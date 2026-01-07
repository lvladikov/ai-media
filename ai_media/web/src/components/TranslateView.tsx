import { useState, useRef, useEffect } from 'react';
import { useAppStore } from '../store';
import { Upload, Languages, FileType, Loader2, Mic, Square, Play, Trash2, Image as ImageIcon, FileText } from 'lucide-react';
import { API_BASE_URL } from '../config';
import { DragDropZone } from './common/DragDropZone';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { ErrorAlert } from './common/ErrorAlert';
import { ModelHelpLink } from './common/ModelHelpLink';

// Check if a file extension can't be previewed in browsers
const isNonPreviewableFormat = (filename: string): boolean => {
    const ext = filename.split('.').pop()?.toLowerCase();
    return ['tiff', 'tif', 'psd', 'raw', 'docx', 'doc'].includes(ext || '');
};

import { TranslateOptions } from './common/TranslateOptions';
import { ALL_LANGUAGES } from '../data/languages';

// Audio models specific to this view
const AUDIO_MODELS = [
    { value: 'seamless-m4t-v2-large', label: 'SeamlessM4T v2 (Best)' },
    { value: 'pipeline', label: 'Pipeline (Whisper + NLLB + Bark)' },
];

export function TranslateView() {
    const { addJob } = useAppStore();
    // Mode State
    const [mode, setMode] = useState<'text' | 'image' | 'audio'>('text');

    const [file, setFile] = useState<File | null>(null);
    const [serverFilePath, setServerFilePath] = useState<string | null>(null);
    const [targetLanguage, setTargetLanguage] = useState('eng_Latn');

    const [selectedModel, setSelectedModel] = useState('nllb-200-3.3b');
    const [ocrModel, setOcrModel] = useState('florence');

    // Input Source State
    const [inputType, setInputType] = useState<'file' | 'text'>('file');
    const [textInput, setTextInput] = useState('');


    const [isUploading, setIsUploading] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [currentJobId, setCurrentJobId] = useState<string | null>(null);
    const [result, setResult] = useState<string | null>(null);
    const [isPreviewOpen, setIsPreviewOpen] = useState(false);
    const [inputPreviewUrl, setInputPreviewUrl] = useState<string | null>(null);

    // Audio Recording State
    const [isRecording, setIsRecording] = useState(false);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [recordingDuration, setRecordingDuration] = useState(0);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const timerRef = useRef<number | null>(null);



    // Reset model when mode changes
    useEffect(() => {
        if (mode === 'audio') setSelectedModel('seamless-m4t-v2-large');
        else setSelectedModel('nllb-200-3.3b');

        // Clear file when switching modes to avoid confusion
        setFile(null);
        setServerFilePath(null);
        setAudioBlob(null);
        setInputPreviewUrl(null);
        setResult(null);
    }, [mode]);



    const processFile = async (selectedFile: File) => {
        setFile(selectedFile);
        setAudioBlob(null); // Clear audio if file is selected
        setResult(null);

        // Auto-switch mode based on file type if needed, but respect user choice if possible
        const type = selectedFile.type;
        if (type.startsWith('audio/') || type.startsWith('video/')) {
            if (mode !== 'audio') setMode('audio');
        } else if (type.startsWith('image/')) {
            if (mode !== 'image') setMode('image');
        } else {
            if (mode !== 'text') setMode('text');
        }

        // Create local preview
        if (selectedFile.type.startsWith('image/') && !isNonPreviewableFormat(selectedFile.name)) {
            if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
            setInputPreviewUrl(URL.createObjectURL(selectedFile));
        } else {
            if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
            setInputPreviewUrl(null);
        }

        // Upload immediately
        setIsUploading(true);
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(`${API_BASE_URL()}/api/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error('Upload failed');

            const data = await response.json();
            setServerFilePath(data.path);
        } catch (error) {
            console.error("Upload error:", error);
            alert("Failed to upload file");
        } finally {
            setIsUploading(false);
        }
    };

    const startRecording = async () => {
        setMode('audio'); // Ensure audio mode
        setFile(null); // Clear file if recording starts
        setServerFilePath(null);
        setAudioBlob(null);
        setRecordingDuration(0);
        setError(null);

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            const chunks: BlobPart[] = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) chunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(chunks, { type: 'audio/webm' });
                setAudioBlob(blob);
                const url = URL.createObjectURL(blob);
                setInputPreviewUrl(url); // Use preview URL for audio playback

                // Upload the recording immediately
                uploadBlob(blob, 'recording.webm').then(path => setServerFilePath(path));

                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);

            timerRef.current = window.setInterval(() => {
                setRecordingDuration(d => d + 1);
            }, 1000);

        } catch (err) {
            setError("Could not access microphone.");
            console.error(err);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            if (timerRef.current) {
                clearInterval(timerRef.current);
                timerRef.current = null;
            }
        }
    };

    const uploadBlob = async (blob: Blob, filename: string): Promise<string | null> => {
        const formData = new FormData();
        formData.append('file', blob, filename);

        try {
            const response = await fetch(`${API_BASE_URL()}/api/upload`, {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) throw new Error('Upload failed');
            const data = await response.json();
            return data.path;
        } catch (err) {
            console.error("Upload error:", err);
            setError("Failed to upload input.");
            return null;
        }
    };

    const handleTranslate = async () => {
        // Validate input
        let currentServerPath = serverFilePath;

        // If direct text input, upload it first
        if (mode === 'text' && inputType === 'text') {
            if (!textInput.trim()) {
                setError("Please enter some text to translate.");
                return;
            }

            setIsSubmitting(true); // Start loading state early
            const blob = new Blob([textInput], { type: 'text/plain' });
            const path = await uploadBlob(blob, `direct_input_${Date.now()}.txt`);

            if (!path) {
                setIsSubmitting(false);
                return;
            }
            currentServerPath = path;
        }

        if (!currentServerPath || !targetLanguage) {
            // If checking fails and we weren't in text mode (which handles its own error above)
            if (!currentServerPath && !(mode === 'text' && inputType === 'text')) return;
        }

        setIsSubmitting(true);
        setError(null);
        setResult(null);

        // Determine target format
        let targetFormat = 'txt';
        let ocrEnabled = false;

        if (mode === 'audio') {
            targetFormat = 'mp3';
        } else if (mode === 'image') {
            // If translating an image, we want an image back, unless user explicitly wants text (mode separation handles this generally)
            // But for "Translate Image", we want the visual result
            const ext = file?.name.split('.').pop()?.toLowerCase();
            targetFormat = ext || 'png';
            ocrEnabled = true;
        } else if (mode === 'text') {
            // For files, preserve ext. For direct text, default to txt (already handled by blob upload name)
            if (inputType === 'file' && file) {
                const ext = file.name.split('.').pop()?.toLowerCase();
                targetFormat = ext || 'txt';
            }
        }

        try {
            const response = await fetch(`${API_BASE_URL()}/api/convert`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input_path: currentServerPath,
                    target_format: targetFormat,
                    ocr_enabled: ocrEnabled,
                    ocr_model: ocrModel,
                    translate: true,
                    target_language: targetLanguage,
                    translation_model: selectedModel,
                    is_direct_text: mode === 'text' && inputType === 'text',
                }),
            });

            if (!response.ok) throw new Error('Translation request failed');

            const data = await response.json();
            setCurrentJobId(data.job_id);

            addJob({
                job_id: data.job_id,
                type: 'convert',
                status: 'pending',
                progress: 0,
                phase: 'queued',
                message: 'Translation queued',
                result_path: null,
                error: null,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
            });

        } catch (error) {
            console.error(error);
            setError("Failed to start translation");
            setIsSubmitting(false); // Reset if we fail start
        }
    };

    // Watch for job completion
    useEffect(() => {
        if (!currentJobId) return;

        let hasSeenJob = false;

        const unsubscribe = useAppStore.subscribe((state) => {
            const job = state.jobs.find(j => j.job_id === currentJobId);
            if (job) {
                hasSeenJob = true;

                if (job.status === 'complete') {
                    setResult(job.result_path);
                    setIsSubmitting(false);
                } else if (job.status === 'failed') {
                    setIsSubmitting(false);
                    setError(job.error || job.message || "Translation failed");
                } else if (job.status === 'cancelled') {
                    setIsSubmitting(false);
                    setError("Job cancelled.");
                    setTimeout(() => setError(null), 6000);
                }
            } else if (hasSeenJob) {
                setIsSubmitting(false);
                setCurrentJobId(null);
            }
        });
        return () => unsubscribe();
    }, [currentJobId]);

    return (
        <div className="flex flex-col lg:flex-row h-full bg-primary text-primary">
            {/* Parameters Sidebar */}
            <div className="w-full lg:w-[500px] border-b lg:border-b-0 lg:border-r border-border p-4 lg:py-6 lg:pr-[27px] lg:pl-1 flex flex-col gap-6 overflow-y-auto shrink-0 h-auto lg:h-full">
                <div>
                    <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
                        <Languages className="text-blue-400" /> AI Translator
                    </h2>
                    <p className="text-xs text-tertiary">Translate any format using advanced models</p>
                </div>

                {/* Mode Tabs */}
                <div className="flex bg-primary p-1 rounded-lg border border-border">
                    <button
                        className={`flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all flex items-center justify-center gap-2 ${mode === 'text' ? 'bg-indigo-100 dark:bg-secondary text-indigo-900 dark:text-primary shadow-lg' : 'text-slate-600 dark:text-slate-400 hover:text-indigo-700 dark:hover:text-slate-200 hover:bg-indigo-50 dark:hover:bg-tertiary'}`}
                        onClick={() => setMode('text')}
                    >
                        <FileText size={14} />
                        Text & Docs
                    </button>
                    <button
                        className={`flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all flex items-center justify-center gap-2 ${mode === 'image' ? 'bg-indigo-100 dark:bg-secondary text-indigo-900 dark:text-primary shadow-lg' : 'text-slate-600 dark:text-slate-400 hover:text-indigo-700 dark:hover:text-slate-200 hover:bg-indigo-50 dark:hover:bg-tertiary'}`}
                        onClick={() => setMode('image')}
                    >
                        <ImageIcon size={14} />
                        Images
                    </button>
                    <button
                        className={`flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all flex items-center justify-center gap-2 ${mode === 'audio' ? 'bg-indigo-100 dark:bg-secondary text-indigo-900 dark:text-primary shadow-lg' : 'text-slate-600 dark:text-slate-400 hover:text-indigo-700 dark:hover:text-slate-200 hover:bg-indigo-50 dark:hover:bg-tertiary'}`}
                        onClick={() => setMode('audio')}
                    >
                        <Mic size={14} />
                        Audio
                    </button>
                </div>

                {/* Pro Tip - Dynamic based on Mode */}
                <div className="bg-primary border border-border rounded-lg p-3 text-xs leading-relaxed text-secondary/80">
                    <strong className="text-blue-400 flex items-center gap-1 mb-1">
                        <span className="text-yellow-400">💡</span>
                        {mode === 'text' && 'Translate Documents'}
                        {mode === 'image' && 'Translate Images (OCR)'}
                        {mode === 'audio' && 'Translate Speech'}
                    </strong>
                    {mode === 'text' && 'Supports .pdf, .docx, .txt. Choose between NLLB for standard translation or LLMs like Qwen/Llama for nuance.'}
                    {mode === 'image' && 'Extracts text from images using Qwen-VL or Florence-2, translates it, and reconstructs the image with the translated text in place.'}
                    {mode === 'audio' && 'Use SeamlessM4T for direct Speech-to-Speech translation, or the pipeline for more control.'}
                </div>

                {/* Input Source */}
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <label className="text-sm font-medium text-secondary">
                            Input Source
                        </label>

                        {/* Toggle only available for Text mode */}
                        {mode === 'text' && (
                            <div className="flex bg-primary border border-border rounded-lg p-0.5">
                                <button
                                    onClick={() => setInputType('file')}
                                    className={`px-3 py-1 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${inputType === 'file'
                                        ? 'bg-secondary text-primary shadow-sm'
                                        : 'text-tertiary hover:text-secondary'}`}
                                >
                                    File Upload
                                </button>
                                <button
                                    onClick={() => setInputType('text')}
                                    className={`px-3 py-1 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${inputType === 'text'
                                        ? 'bg-secondary text-primary shadow-sm'
                                        : 'text-tertiary hover:text-secondary'}`}
                                >
                                    Direct Text
                                </button>
                            </div>
                        )}
                    </div>

                    {inputType === 'text' && mode === 'text' ? (
                        <div className="relative">
                            <textarea
                                className="w-full h-48 bg-primary border border-border rounded-lg p-4 text-sm focus:outline-none focus:border-blue-500 transition-colors resize-none scrollbar-themed"
                                placeholder="Paste or type text to translate..."
                                value={textInput}
                                onChange={(e) => setTextInput(e.target.value)}
                            />
                            <div className="absolute bottom-3 right-3 text-[10px] text-tertiary">
                                {textInput.length} chars
                            </div>
                        </div>
                    ) : (
                        <DragDropZone
                            onFileDrop={processFile}
                            className={`border border-dashed rounded-lg p-6 text-center transition-all cursor-pointer relative overflow-hidden ${audioBlob ? 'border-blue-500 bg-blue-500/5' : 'border-border hover:border-border hover:bg-primary'}`}
                            draggingClassName="border-blue-400 bg-blue-500/10 scale-[1.02] shadow-xl"
                            rejectClassName="border-red-500 bg-red-500/10"
                            accept={mode === 'text' ? ".pdf,.docx,.txt,.md" : mode === 'image' ? "image/*" : ".mp3,.wav,.m4a,audio/*,video/*"}
                        >
                            {({ isDragging }) => (
                                <>
                                    {file ? (
                                        <div className={`flex flex-col items-center justify-center gap-2 ${isDragging ? 'opacity-50' : ''}`}>
                                            {/* File Preview */}
                                            {inputPreviewUrl && file.type.startsWith('image/') ? (
                                                <div className="relative h-20 w-auto min-w-[5rem] mb-1 rounded overflow-hidden border border-border bg-primary shadow-sm">
                                                    <img src={inputPreviewUrl} alt="Preview" className="h-full w-full object-contain" />
                                                </div>
                                            ) : (
                                                <div className="w-10 h-10 bg-secondary rounded flex items-center justify-center text-primary-400 mb-1">
                                                    <FileType size={20} />
                                                </div>
                                            )}
                                            <div className="text-center w-full">
                                                <p className="font-medium text-primary text-xs truncate max-w-[200px] mx-auto">{file.name}</p>
                                                <p className="text-xs text-secondary">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                            </div>
                                            <div className="text-xs text-primary-400 font-medium">Click to replace</div>
                                        </div>
                                    ) : audioBlob ? (
                                        <div className="flex flex-col items-center justify-center gap-2">
                                            <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center text-blue-600 dark:text-blue-400 mb-1 animate-pulse">
                                                <Mic size={24} />
                                            </div>
                                            <div className="text-center">
                                                <p className="font-medium text-primary text-sm">Audio Recording</p>
                                                <p className="text-xs text-secondary">{(audioBlob.size / 1024).toFixed(1)} KB</p>
                                            </div>
                                            {inputPreviewUrl && (
                                                <audio src={inputPreviewUrl} controls className="h-8 w-48 mt-2" />
                                            )}
                                            <button
                                                onClick={(e) => { e.stopPropagation(); setAudioBlob(null); setInputPreviewUrl(null); }}
                                                className="text-xs text-red-500 hover:underline flex items-center gap-1 mt-1"
                                            >
                                                <Trash2 size={12} /> Remove
                                            </button>
                                        </div>
                                    ) : (
                                        <>
                                            <Upload size={24} className={`mx-auto mb-2 transition-colors ${isDragging ? 'text-blue-400' : 'text-tertiary'}`} />
                                            <p className={`font-medium text-sm transition-colors ${isDragging ? 'text-blue-200' : ''}`}>
                                                Drag & Drop {mode === 'text' ? 'Document' : mode === 'image' ? 'Image' : 'Audio File'}
                                            </p>
                                            <p className="text-[10px] text-tertiary mt-1">or click to browse</p>
                                        </>
                                    )}
                                    {isUploading && <div className="absolute top-2 right-2"><Loader2 className="animate-spin text-tertiary" size={16} /></div>}
                                </>
                            )}
                        </DragDropZone>
                    )}

                    {/* Mic Button - Only show/enable in Audio Mode or show disabled hint */}
                    {mode === 'audio' && (
                        <>
                            <div className="flex items-center gap-3">
                                <div className="h-px bg-border flex-1" />
                                <span className="text-[10px] text-tertiary uppercase font-bold">OR</span>
                                <div className="h-px bg-border flex-1" />
                            </div>

                            <button
                                onClick={isRecording ? stopRecording : startRecording}
                                disabled={isUploading || isSubmitting || (!!file && !isRecording)}
                                className={`w-full py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all ${isRecording
                                    ? 'bg-red-500 hover:bg-red-600 text-white animate-pulse'
                                    : 'bg-primary border border-border hover:bg-primary-hover text-secondary'
                                    }`}
                            >
                                {isRecording ? (
                                    <><Square size={16} fill="currentColor" /> Stop Recording ({recordingDuration}s)</>
                                ) : (
                                    <><Mic size={18} /> Record Microphone</>
                                )}
                            </button>
                        </>
                    )}
                </div>

                {/* Model Selection - Context Aware */}
                <div className="space-y-4 animate-in fade-in slide-in-from-top-1 duration-200">

                    {/* OCR Model - Only for Images */}
                    {mode === 'image' && (
                        <div className="space-y-4">
                            {/* OCR Model */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-secondary flex justify-between items-center">
                                    <span className="flex items-center">OCR Model <ModelHelpLink section="multimedia" /></span>
                                </label>
                                <div className="flex gap-2 p-1 bg-primary rounded-lg border border-border">
                                    <button
                                        onClick={() => setOcrModel('qwen-vl')}
                                        className={`flex-1 py-1.5 px-2 rounded-md text-[10px] font-bold uppercase transition-all ${ocrModel === 'qwen-vl'
                                            ? 'bg-purple-600 text-white shadow-sm'
                                            : 'text-tertiary hover:text-secondary'
                                            }`}
                                    >
                                        Qwen-VL
                                    </button>
                                    <button
                                        onClick={() => setOcrModel('florence')}
                                        className={`flex-1 py-1.5 px-2 rounded-md text-[10px] font-bold uppercase transition-all ${ocrModel === 'florence'
                                            ? 'bg-blue-600 text-white shadow-sm'
                                            : 'text-tertiary hover:text-secondary'
                                            }`}
                                    >
                                        Florence-2
                                    </button>
                                </div>
                            </div>


                        </div>
                    )}

                    {/* Translation Options (Text/Image) */}
                    {(mode === 'text' || mode === 'image') && (
                        <div className="animate-in fade-in slide-in-from-top-1 duration-200">
                            <TranslateOptions
                                enabled={true}
                                onEnabledChange={() => { }} // Always enabled in this view
                                showToggle={false}
                                selectedModel={selectedModel}
                                onModelChange={setSelectedModel}
                                targetLanguage={targetLanguage}
                                onLanguageChange={setTargetLanguage}
                                title={mode === 'image' ? 'Translation' : 'Settings'}
                                infoMessage="Choose NLLB for speed and broad language coverage. Use LLM models for more natural, context-aware translations - especially valuable for professional or creative content."
                            />
                        </div>
                    )}

                    {/* Audio Mode Specifics (Custom Models) */}
                    {mode === 'audio' && (
                        <div className="space-y-2 animate-in fade-in slide-in-from-top-1 duration-200">
                            <label className="text-sm font-medium text-secondary">Translation Model</label>
                            <select
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                className="w-full text-sm p-3 rounded-lg border border-border bg-primary text-secondary focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-shadow"
                            >
                                {AUDIO_MODELS.map(opt => (
                                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                                ))}
                            </select>
                            {/* Target Language for Audio (Reusing TranslateOptions would be nice if it accepted custom models, but it doesn't yet. So manual select here or refactor component to accept models.)
                                Actually, audio translation targets NLLB languages usually (for Seamless text out) or specific ones. Seamless supports many.
                                For now, I'll keep the manual Language select for Audio to avoid breaking it, as `TranslateOptions` enforces derived lists from Text Models.
                             */}
                            <div className="space-y-2 mt-4">
                                <label className="text-sm font-medium text-secondary">Target Language</label>
                                {/* Manual select for Audio to avoid breaking Seamless logic which might differ or just reuse logic?
                                    Seamless uses standard codes? Yes.
                                    But TranslateOptions is tied to TRANSLATE_MODELS.
                                    I'll leave Audio mode distinct for this step to ensure safety, as the component is "Translate Output" (usually text).
                                */}
                            </div>
                        </div>
                    )}
                </div>

                {/* Target Language for Audio Mode (Text/Image handled by component) */}
                {mode === 'audio' && (
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-secondary">Target Language</label>
                        <select
                            value={targetLanguage}
                            onChange={(e) => setTargetLanguage(e.target.value)}
                            className="w-full text-sm p-3 rounded-lg border border-border bg-primary text-secondary focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-shadow"
                        >
                            {ALL_LANGUAGES.sort((a, b) => a.label.localeCompare(b.label)).map(opt => (
                                <option key={opt.value} value={opt.value}>{opt.label}</option>
                            ))}
                        </select>
                    </div>
                )}

                <ErrorAlert error={error} onDismiss={() => setError(null)} />

                {/* Action Button */}
                <div className="mt-auto pt-4">
                    <button
                        onClick={handleTranslate}
                        disabled={
                            (mode === 'text' && inputType === 'text' && !textInput.trim()) ||
                            (mode === 'text' && inputType === 'file' && !file) ||
                            (mode !== 'text' && !file && !audioBlob) ||
                            isUploading ||
                            isSubmitting
                        }
                        className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-white font-bold py-3 rounded-lg shadow-lg shadow-blue-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all"
                    >
                        {isSubmitting ? (
                            <><Loader2 className="animate-spin" size={18} /> Translating...</>
                        ) : (
                            <><Languages size={18} /> {audioBlob ? 'Translate Speech' : 'Translate ' + (mode === 'image' ? 'Image' : 'Document')}</>
                        )}
                    </button>
                </div>
            </div>

            {/* Main Preview Area */}
            <div className="flex-1 p-6 flex items-center justify-center bg-primary/30 min-h-[500px] lg:min-h-0">
                {!result && !file && !audioBlob && (
                    <div className="text-center text-tertiary">
                        <Languages size={48} className="mx-auto mb-4 opacity-20" />
                        <h3 className="text-lg font-medium mb-2">AI Translator</h3>
                        <p className="text-secondary max-w-sm">
                            Select a mode (Text, Image, or Audio) to start translating.
                        </p>
                    </div>
                )}
                {/* Preview for Audio Blob if no result yet */}
                {!result && audioBlob && (
                    <div className="flex flex-col items-center justify-center gap-4 animate-in fade-in">
                        <div className="w-24 h-24 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center text-blue-600">
                            <Mic size={40} />
                        </div>
                        <div className="text-center">
                            <h3 className="text-lg font-bold text-primary">Audio Recorded</h3>
                            <p className="text-sm text-secondary">Ready to translate to {ALL_LANGUAGES.find(l => l.value === targetLanguage)?.label}</p>
                        </div>
                    </div>
                )}

                {result && (
                    <div className="flex flex-col items-center justify-center max-w-full h-full gap-6 animate-in fade-in zoom-in duration-300">
                        <div className="relative border-4 border-brand-500/30 rounded-xl overflow-hidden shadow-2xl flex flex-col items-center justify-center bg-primary/50 min-w-[280px] min-h-[320px] cursor-pointer group" onClick={() => setIsPreviewOpen(true)}>
                            {/* If result is audio, show audio player icon */}
                            {['mp3', 'wav', 'aac'].includes(result.split('.').pop() || '') ? (
                                <div className="flex flex-col items-center">
                                    <div className="w-20 h-20 bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center text-green-600 mb-4">
                                        <Play size={32} className="ml-1" />
                                    </div>
                                    <span className="text-lg font-bold text-secondary">Translated Audio</span>
                                </div>
                            ) : (
                                <FileType size={64} className="text-blue-500 mb-4" />
                            )}

                            <div className="flex flex-col items-center text-center mt-2">
                                <span className="text-sm font-medium text-blue-400 mt-1 uppercase tracking-widest">{targetLanguage.toUpperCase()}</span>
                            </div>

                            <div className="absolute top-2 left-2 bg-blue-600 px-2 py-1 rounded text-xs text-white shadow-lg flex flex-col items-start leading-none gap-0.5">
                                <span className="font-bold uppercase tracking-wider">Translated</span>
                            </div>

                            <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                                <span className="bg-white/90 text-black px-4 py-2 rounded-lg font-bold text-sm">Open Preview</span>
                            </div>
                        </div>

                        <div className="flex gap-3">
                            <button className="btn-secondary text-sm" onClick={() => setIsPreviewOpen(true)}>Preview</button>
                            <a href={`${API_BASE_URL()}/api/files/${result}?download=true`} className="btn-secondary text-sm">Download File</a>
                        </div>
                    </div>
                )}
            </div>

            {currentJobId && (
                <JobProgressModal
                    jobId={currentJobId}
                    title={
                        mode === 'image' ? 'Image Translation' :
                            mode === 'audio' ? 'Audio & Speech Translation' :
                                inputType === 'text' ? 'Text Translation' : 'Document Translation'
                    }
                    onClose={() => setCurrentJobId(null)}
                    onViewResult={() => {
                        setCurrentJobId(null);
                        setIsPreviewOpen(true);
                    }}
                />
            )}

            {result && isPreviewOpen && (
                <PreviewModal
                    isOpen={isPreviewOpen}
                    onClose={() => setIsPreviewOpen(false)}
                    filePath={result}
                    fileName={result.split(/[/\\]/).pop() || 'translated-file'}
                />
            )}
        </div>
    );
}
