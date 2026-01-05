import { useState, useRef, useEffect } from 'react';
import { useAppStore } from '../store';
import {
    Upload, FileText, Copy, Check, Loader2,
    Eye, Download, Search
} from 'lucide-react';
import { ErrorAlert } from './common/ErrorAlert';
import { ModelHelpLink } from './common/ModelHelpLink';
import { DragDropZone } from './common/DragDropZone';
import { JobProgressModal } from './common/JobProgressModal';
import { VisionPreviewModal } from './common/VisionPreviewModal';
import { API_BASE_URL } from '../config';

const isNonPreviewableFormat = (filename: string): boolean => {
    const ext = filename.split('.').pop()?.toLowerCase();
    return ['tiff', 'tif', 'psd', 'raw'].includes(ext || '');
};

export function VisionView() {
    const [file, setFile] = useState<File | null>(null);
    const [inputPreview, setInputPreview] = useState<string | null>(null);
    const [serverFilePath, setServerFilePath] = useState<string | null>(null);
    const [isUploading, setIsUploading] = useState(false);
    const [selectedModel, setSelectedModel] = useState('florence');
    const [isGenerating, setIsGenerating] = useState(false);
    const [isPreviewOpen, setIsPreviewOpen] = useState(false);
    const [result, setResult] = useState<string | null>(null); // Path
    const [resultText, setResultText] = useState<string | null>(null); // Content
    const [copied, setCopied] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [currentJobId, setCurrentJobId] = useState<string | null>(null);

    const { addJob, isConnected } = useAppStore();
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileDrop = async (droppedFile: File) => {
        setFile(droppedFile);
        setResult(null);
        setResultText(null);
        setError(null);
        setCurrentJobId(null);

        if (droppedFile.type.startsWith('image/') && !isNonPreviewableFormat(droppedFile.name)) {
            const reader = new FileReader();
            reader.onload = (ev) => setInputPreview(ev.target?.result as string);
            reader.readAsDataURL(droppedFile);
        } else {
            setInputPreview(null);
        }

        setIsUploading(true);
        const formData = new FormData();
        formData.append('file', droppedFile);

        try {
            const response = await fetch(`${API_BASE_URL()}/api/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error('Upload failed');

            const data = await response.json();
            setServerFilePath(data.path);
        } catch (err: any) {
            console.error("Upload error:", err);
            setError("Failed to upload media");
        } finally {
            setIsUploading(false);
        }
    };

    const startAnalysis = async () => {
        if (!serverFilePath || !isConnected) return;

        const optimisticJobId = crypto.randomUUID();
        setCurrentJobId(optimisticJobId);
        setResult(null);
        setResultText(null);
        setError(null);

        addJob({
            job_id: optimisticJobId,
            type: 'vision',
            status: 'pending',
            progress: 0,
            phase: 'Queued',
            message: 'Waiting for worker...',
            result_path: null,
            error: null,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            params: { model: selectedModel, input: serverFilePath }
        });

        try {
            const response = await fetch(`${API_BASE_URL()}/api/vision/describe`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input_path: serverFilePath,
                    model: selectedModel,
                    job_id: optimisticJobId
                }),
            });

            if (!response.ok) throw new Error('Failed to start analysis job');
            setIsGenerating(true);
        } catch (err: any) {
            setError(err.message);
            setIsGenerating(false);
            setCurrentJobId(null);
        }
    };

    useEffect(() => {
        if (!currentJobId) return;

        let hasSeenJob = false;

        const unsubscribe = useAppStore.subscribe((state) => {
            const job = state.jobs.find(j => j.job_id === currentJobId);
            if (job) {
                hasSeenJob = true;
                if (job.status === 'complete') {
                    setResult(job.result_path || null);
                    setResultText((job as any).result || null);
                    setIsGenerating(false);
                    setCurrentJobId(null); // Clear job ID to close progress modal
                    setIsPreviewOpen(true); // Auto-open preview
                } else if (job.status === 'failed' || job.status === 'cancelled') {
                    setIsGenerating(false);
                    setCurrentJobId(null);
                    if (job.error || job.message) {
                        setError(job.error || job.message || "Generation failed");
                    }
                }
            } else if (hasSeenJob) {
                setIsGenerating(false);
                setCurrentJobId(null);
            }
        });

        return () => unsubscribe();
    }, [currentJobId]);

    const handleCloseModal = () => {
        setCurrentJobId(null);
        setIsGenerating(false);
    };

    const handleViewResult = () => {
        setCurrentJobId(null);
        setIsGenerating(false);
    };

    const copyToClipboard = () => {
        if (resultText) {
            navigator.clipboard.writeText(resultText);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    return (
        <div className="flex flex-col lg:flex-row h-full bg-primary text-primary">
            <div className="w-full lg:w-[500px] border-b lg:border-b-0 lg:border-r border-border p-4 lg:py-6 lg:pr-[27px] lg:pl-1 flex flex-col gap-6 overflow-y-auto shrink-0 h-auto lg:h-full">
                <div>
                    <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
                        <Search className="text-brand-400" size={20} /> Vision
                    </h2>
                    <p className="text-xs text-tertiary">Generate detailed text descriptions from images and videos</p>
                </div>

                <div className="space-y-2">
                    <label className="text-sm font-medium text-secondary">Input File</label>
                    <DragDropZone
                        onFileDrop={handleFileDrop}
                        accept="image/*,video/*,.gif,.tiff,.tif"
                        className={`border-2 border-dashed border-border rounded-lg p-6 flex flex-col items-center justify-center gap-2 hover:border-brand-500/50 hover:bg-secondary/50 transition-colors cursor-pointer relative ${file ? 'bg-brand-500/5' : ''}`}
                        draggingClassName="border-brand-500 bg-brand-500/10"
                    >
                        <input type="file" ref={fileInputRef} className="hidden" accept="image/*,video/*,.gif,.tiff,.tif" onChange={(e) => e.target.files?.[0] && handleFileDrop(e.target.files[0])} />

                        {file && isNonPreviewableFormat(file.name) ? (
                            <div className="flex flex-col items-center gap-2 py-2">
                                <div className="w-16 h-16 bg-tertiary/50 rounded-lg flex items-center justify-center border border-border">
                                    <FileText size={28} className="text-tertiary" />
                                </div>
                                <span className="text-[10px] text-tertiary bg-secondary px-2 py-0.5 rounded uppercase">
                                    {file.name.split('.').pop()} - No Preview
                                </span>
                            </div>
                        ) : inputPreview ? (
                            <img src={inputPreview} alt="Preview" className="max-h-32 object-contain rounded shadow-sm mb-1" />
                        ) : (
                            <div className="flex flex-col items-center gap-2">
                                <Upload size={28} className="text-secondary" />
                                <span className="text-sm text-secondary">Click or Drag & Drop</span>
                            </div>
                        )}

                        <div className="text-center w-full">
                            <p className="font-medium text-primary text-xs truncate max-w-[200px] mx-auto" title={file ? file.name : undefined}>
                                {file ? file.name : "Upload Media"}
                            </p>
                            <p className="text-xs text-secondary mt-0.5">
                                {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : "Image or Video"}
                            </p>
                        </div>

                        {isUploading && (
                            <div className="absolute inset-0 bg-black/60 flex items-center justify-center rounded-lg z-20">
                                <Loader2 className="animate-spin text-brand-400" />
                            </div>
                        )}
                    </DragDropZone>
                </div>

                <div className="space-y-2">
                    <label className="text-sm font-medium text-secondary flex items-center">
                        Vision Model
                        <ModelHelpLink section="vision" />
                    </label>
                    <div className="grid grid-cols-1 gap-2">
                        {/* Florence - Green */}
                        <button
                            onClick={() => setSelectedModel('florence')}
                            className={`flex items-center gap-3 p-3 rounded-lg border text-left transition-all ${selectedModel === 'florence'
                                ? 'bg-emerald-50 dark:bg-emerald-500/10 border-emerald-200 dark:border-emerald-500/50 text-emerald-700 dark:text-emerald-400'
                                : 'bg-primary border-border text-secondary hover:bg-secondary/50'
                                }`}
                        >
                            <FileText size={18} />
                            <div className="flex-1">
                                <div className="font-medium text-sm">Florence-2 (Default, SOTA)</div>
                                <div className="text-[10px] opacity-70">Large model, spatial awareness, rich detail.</div>
                            </div>
                        </button>

                        {/* Qwen3-VL 8B - Purple */}
                        <button
                            onClick={() => setSelectedModel('qwen3-vl-8b')}
                            className={`flex items-center gap-3 p-3 rounded-lg border text-left transition-all ${selectedModel === 'qwen3-vl-8b'
                                ? 'bg-purple-50 dark:bg-purple-500/10 border-purple-200 dark:border-purple-500/50 text-purple-700 dark:text-purple-400'
                                : 'bg-primary border-border text-secondary hover:bg-secondary/50'
                                }`}
                        >
                            <Eye size={18} />
                            <div className="flex-1">
                                <div className="font-medium text-sm">Qwen3-VL 8B (Vision-Language)</div>
                                <div className="text-[10px] opacity-70">Best for complex scene understanding.</div>
                            </div>
                        </button>

                        {/* Qwen3-VL 4B - Purple */}
                        <button
                            onClick={() => setSelectedModel('qwen3-vl-4b')}
                            className={`flex items-center gap-3 p-3 rounded-lg border text-left transition-all ${selectedModel === 'qwen3-vl-4b'
                                ? 'bg-purple-50 dark:bg-purple-500/10 border-purple-200 dark:border-purple-500/50 text-purple-700 dark:text-purple-400'
                                : 'bg-primary border-border text-secondary hover:bg-secondary/50'
                                }`}
                        >
                            <Eye size={18} />
                            <div className="flex-1">
                                <div className="font-medium text-sm">Qwen3-VL 4B (Balanced)</div>
                                <div className="text-[10px] opacity-70">Good balance of quality and speed.</div>
                            </div>
                        </button>

                        {/* Qwen3-VL 2B - Purple */}
                        <button
                            onClick={() => setSelectedModel('qwen3-vl-2b')}
                            className={`flex items-center gap-3 p-3 rounded-lg border text-left transition-all ${selectedModel === 'qwen3-vl-2b'
                                ? 'bg-purple-50 dark:bg-purple-500/10 border-purple-200 dark:border-purple-500/50 text-purple-700 dark:text-purple-400'
                                : 'bg-primary border-border text-secondary hover:bg-secondary/50'
                                }`}
                        >
                            <Eye size={18} />
                            <div className="flex-1">
                                <div className="font-medium text-sm">Qwen3-VL 2B (Lightweight)</div>
                                <div className="text-[10px] opacity-70">Fast, lower memory requirements.</div>
                            </div>
                        </button>

                        {/* BLIP - Blue */}
                        <button
                            onClick={() => setSelectedModel('blip')}
                            className={`flex items-center gap-3 p-3 rounded-lg border text-left transition-all ${selectedModel === 'blip'
                                ? 'bg-blue-50 dark:bg-blue-500/10 border-blue-200 dark:border-blue-500/50 text-blue-700 dark:text-blue-400'
                                : 'bg-primary border-border text-secondary hover:bg-secondary/50'
                                }`}
                        >
                            <Search size={18} />
                            <div className="flex-1">
                                <div className="font-medium text-sm">BLIP (Classic)</div>
                                <div className="text-[10px] opacity-70">Lightweight, fast, standard captions.</div>
                            </div>
                        </button>
                    </div>
                </div>

                <ErrorAlert error={error} onDismiss={() => setError(null)} />

                <div className="mt-auto pt-4">
                    <button
                        disabled={!serverFilePath || isGenerating || isUploading || !isConnected || !!currentJobId}
                        onClick={startAnalysis}
                        className={`w-full py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all ${!serverFilePath || isGenerating || isUploading || !isConnected ? 'bg-secondary text-tertiary cursor-not-allowed opacity-50' : 'bg-gradient-to-r from-brand-600 to-indigo-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-primary shadow-lg shadow-brand-900/20 active:scale-[0.98]'}`}
                    >
                        {isGenerating ? (
                            <><Loader2 className="animate-spin" size={18} /> <span>Analyzing...</span></>
                        ) : (
                            <><FileText size={18} /> <span>Start Analysis</span></>
                        )}
                    </button>
                </div>
            </div>

            <div className="flex-1 p-6 flex items-center justify-center bg-primary/30 relative">
                <div className="max-w-3xl w-full">
                    {result ? (
                        <div className="flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
                            <div
                                className="bg-primary border border-brand-500/20 rounded-lg p-6 relative group cursor-pointer hover:border-brand-500/40 transition-all"
                                onClick={() => setIsPreviewOpen(true)}
                            >
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-2 text-brand-400">
                                        <FileText size={18} />
                                        <span className="text-sm font-bold uppercase tracking-wider">Analysis Result</span>
                                    </div>
                                    <button
                                        onClick={(e) => { e.stopPropagation(); copyToClipboard(); }}
                                        className="p-2 bg-secondary/80 hover:bg-secondary rounded-lg text-secondary hover:text-primary transition-all flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider"
                                    >
                                        {copied ? <Check size={12} className="text-green-500" /> : <Copy size={12} />}
                                        {copied ? 'Copied!' : 'Copy'}
                                    </button>
                                </div>
                                <div className="text-secondary leading-relaxed whitespace-pre-wrap line-clamp-[12] text-sm italic">
                                    {resultText || 'Analysis complete. Click below to view the full detailed description in a scrollable preview.'}
                                </div>
                                <div className="absolute top-4 right-16 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <div className="p-2 bg-tertiary rounded-lg text-primary flex items-center gap-2 text-xs">
                                        <Eye size={14} /> View Full
                                    </div>
                                </div>
                            </div>

                            <div className="flex gap-2">
                                <button onClick={() => setIsPreviewOpen(true)} className="btn-secondary flex-1 py-3 font-bold">
                                    View Full Description
                                </button>
                                <a href={`${API_BASE_URL()}/api/files/${result}?download=true`} className="btn-secondary px-6 shrink-0 flex items-center justify-center">
                                    <Download size={18} />
                                </a>
                            </div>
                        </div>
                    ) : file ? (
                        <div className="flex justify-center animate-in fade-in zoom-in-95 duration-500">
                            {isNonPreviewableFormat(file.name) ? (
                                <div className="relative border-4 border-border rounded-xl overflow-hidden shadow-2xl flex flex-col items-center justify-center p-12 bg-primary/50 min-w-[320px] min-h-[320px]">
                                    <FileText size={64} className="text-tertiary mb-4 opacity-50" />
                                    <div className="flex flex-col items-center text-center">
                                        <span className="text-lg font-bold text-secondary uppercase tracking-tight">{file.name.split('.').pop()} File</span>
                                        <span className="text-[10px] text-tertiary mt-1 uppercase tracking-widest font-mono">No Browser Preview</span>
                                    </div>
                                    <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-md px-3 py-1.5 rounded-lg text-[10px] font-mono text-white border border-white/10 uppercase tracking-tighter">
                                        Original
                                    </div>
                                </div>
                            ) : inputPreview ? (
                                <div className="relative border-4 border-border rounded-xl overflow-hidden shadow-2xl max-w-full">
                                    <img src={inputPreview} className="max-w-full max-h-[70vh] object-contain" alt="Original Content" />
                                    <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-md px-3 py-1.5 rounded-lg text-[10px] font-mono text-white border border-white/10 uppercase tracking-tighter">
                                        Original
                                    </div>
                                </div>
                            ) : file.type.startsWith('video/') && serverFilePath ? (
                                <div className="relative border-4 border-border rounded-xl overflow-hidden shadow-2xl max-w-full">
                                    <video src={`${API_BASE_URL()}/api/files/${serverFilePath}`} className="max-w-full max-h-[70vh]" controls />
                                    <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-md px-3 py-1.5 rounded-lg text-[10px] font-mono text-white border border-white/10 uppercase tracking-tighter">
                                        Original
                                    </div>
                                </div>
                            ) : (
                                <div className="text-center text-tertiary animate-pulse">
                                    <Loader2 className="animate-spin inline-block mb-3" size={32} />
                                    <p className="text-sm font-medium">Processing Preview...</p>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="text-center text-tertiary px-12">
                            <div className="relative inline-block mb-6">
                                <div className="p-6 bg-secondary rounded-full border border-border">
                                    <FileText size={64} className="opacity-20" />
                                </div>
                                <div className="absolute -bottom-2 -right-2 p-3 bg-brand-500 rounded-2xl shadow-lg animate-bounce duration-[2000ms]">
                                    <Upload size={20} className="text-white" />
                                </div>
                            </div>
                            <h3 className="text-2xl font-bold mb-3 text-primary">Ready to Analyze</h3>
                            <p className="text-secondary max-w-sm mx-auto leading-relaxed">
                                Drag and drop media files or use the sidebar to start uncovering details with AI Vision.
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {currentJobId && (
                <JobProgressModal jobId={currentJobId} onClose={handleCloseModal} onViewResult={handleViewResult} />
            )}

            {result && serverFilePath && (
                <VisionPreviewModal
                    isOpen={isPreviewOpen}
                    onClose={() => setIsPreviewOpen(false)}
                    originalPath={serverFilePath}
                    resultPath={result}
                    resultText={resultText}
                    fileName={file?.name || 'analysis'}
                    originalIsVideo={file?.type.startsWith('video/')}
                />
            )}
        </div>
    );
}
