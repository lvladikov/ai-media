import { useState, useRef, useMemo, useEffect } from 'react';
import { useAppStore } from '../store';
import { Upload, RefreshCw, FileType, Loader2, Lightbulb } from 'lucide-react';
import { API_BASE_URL } from '../config';
import { DragDropZone } from './common/DragDropZone';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { ComparisonPreviewModal } from './common/ComparisonPreviewModal';
import { ErrorAlert } from './common/ErrorAlert';
import { formatDuration } from '../utils/formatTime';
import { ModelHelpLink } from './common/ModelHelpLink';

// Check if a file extension can't be previewed in browsers
const isNonPreviewableFormat = (filename: string): boolean => {
  const ext = filename.split('.').pop()?.toLowerCase();
  return ['tiff', 'tif', 'psd', 'raw'].includes(ext || '');
};

export function ConvertView() {
  const { addJob } = useAppStore();
  const [file, setFile] = useState<File | null>(null);
  const [serverFilePath, setServerFilePath] = useState<string | null>(null);
  const [targetFormat, setTargetFormat] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [genDuration, setGenDuration] = useState<number | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [inputPreviewUrl, setInputPreviewUrl] = useState<string | null>(null);
  const [ocrModel, setOcrModel] = useState('qwen-vl');

  const resultRef = useRef<HTMLDivElement>(null);

  const availableFormats = useMemo(() => {
    if (!file) return [];

    const ext = file.name.split('.').pop()?.toLowerCase();

    // Logic mirroring server capabilities
    if (['jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif', 'tiff', 'tif'].includes(ext || '')) {
      return ['png', 'jpg', 'webp', 'gif', 'tiff', 'bmp', 'txt (OCR)', 'md (OCR)', 'pdf (OCR)', 'docx (OCR)', 'html (OCR)'];
    }
    if (['mp4', 'mov', 'avi', 'mkv', 'webm'].includes(ext || '')) {
      return ['mp4', 'gif', 'mov', 'webm'];
    }
    if (['mp3', 'wav', 'aac', 'flac', 'ogg', 'm4a'].includes(ext || '')) {
      return ['mp3', 'wav', 'aac', 'flac'];
    }
    if (ext === 'md') {
      return ['html', 'xhtml', 'pdf', 'docx', 'txt'];
    }
    if (ext === 'html') {
      return ['md', 'xhtml', 'pdf', 'docx', 'txt'];
    }
    if (ext === 'docx') {
      return ['md', 'html', 'xhtml', 'pdf', 'txt'];
    }
    if (ext === 'pdf') {
       return ['md', 'txt', 'docx', 'html', 'xhtml', 'rtf', 'json'];
    }
    if (ext === 'txt') {
       return ['pdf', 'md', 'docx', 'html', 'xhtml', 'rtf', 'json'];
    }
    if (ext === 'json') {
       return ['pdf', 'md', 'docx', 'html', 'xhtml', 'rtf', 'txt'];
    }
    if (ext === 'rtf') {
       return ['pdf', 'md', 'docx', 'html', 'xhtml', 'txt', 'json'];
    }
    if (ext === 'xhtml') {
       return ['pdf', 'md', 'docx', 'html', 'txt', 'rtf', 'json'];
    }

    return [];
  }, [file]);
  
  const processFile = async (selectedFile: File) => {
      setFile(selectedFile);
      setTargetFormat(''); // Reset format selection
      setResult(null); // Reset previous result

      // Create local preview if it's an image and not a non-previewable format
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

  const handleConvert = async () => {
    if (!serverFilePath || !targetFormat) return;

    setIsSubmitting(true);
    setError(null);
    setResult(null);
    setGenDuration(null);

    try {
      const response = await fetch(`${API_BASE_URL()}/api/convert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_path: serverFilePath,
          target_format: targetFormat.includes('(OCR)') ? targetFormat.split(' ')[0] : targetFormat,
          ocr_enabled: targetFormat.includes('(OCR)'),
          ocr_model: ocrModel,
        }),
      });

      if (!response.ok) throw new Error('Convert request failed');

      const data = await response.json();

      setCurrentJobId(data.job_id);

      addJob({
        job_id: data.job_id,
        type: 'convert',
        status: 'pending',
        progress: 0,
        phase: 'queued',
        message: 'Job queued',
        result_path: null,
        error: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });

    } catch (error) {
      console.error(error);
      setError("Failed to start conversion");
    } finally {
      setIsSubmitting(false);
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

          // Calculate duration
          if (job.generation_started_at && job.updated_at) {
            const start = new Date(job.generation_started_at).getTime();
            const end = new Date(job.updated_at).getTime();
            const seconds = Math.round((end - start) / 1000);
            setGenDuration(seconds > 0 ? seconds : 1);
          } else if (job.created_at && job.updated_at) {
            const start = new Date(job.created_at).getTime();
            const end = new Date(job.updated_at).getTime();
            setGenDuration(Math.round((end - start) / 1000));
          }

          setIsSubmitting(false);
        } else if (job.status === 'failed') {
          setIsSubmitting(false);
          setError(job.error || job.message || "Conversion failed");
        } else if (job.status === 'cancelled') {
          setIsSubmitting(false);
          setError("Job cancelled.");
          setTimeout(() => setError(null), 6000);
        }
      } else if (hasSeenJob) {
        // Job was in store but now removed (e.g., cancelled)
        setIsSubmitting(false);
        setCurrentJobId(null);
      }
    });
    return () => unsubscribe();
  }, [currentJobId]);

  const handleCloseModal = () => {
    setCurrentJobId(null);
    if (result) setIsPreviewOpen(true);
  };
  
  const handleViewResult = () => {
    setTimeout(() => {
        resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  };

  return (
    <div className="flex flex-col lg:flex-row h-full bg-primary text-primary">
      {/* Parameters Sidebar */}
      <div className="w-full lg:w-[500px] border-b lg:border-b-0 lg:border-r border-border p-4 lg:py-6 lg:pr-[27px] lg:pl-1 flex flex-col gap-6 overflow-y-auto shrink-0 h-auto lg:h-full">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
            <RefreshCw className="text-blue-400" /> Media Converter
          </h2>
          <p className="text-xs text-tertiary">Convert images, video, audio, and documents</p>
        </div>

        {/* Info Block */}
        <div className="bg-blue-50 dark:bg-blue-500/10 border border-blue-200 dark:border-blue-500/20 p-3 rounded-lg">
          <p className="text-[11px] leading-relaxed text-blue-800 dark:text-blue-300">
            <span className="font-bold flex items-center gap-1 mb-1">
              <Lightbulb size={12} className="text-amber-600 dark:text-amber-400 fill-amber-600/20 dark:fill-amber-400/20" /> Pro Tip:
            </span>
            You can also upload <span className="text-secondary font-medium">Images</span> or <span className="text-secondary font-medium">Scanned PDFs</span> to extract text using <span className="text-green-700 dark:text-green-400 font-bold">High-Precision OCR</span>.
          </p>
        </div>

        {/* File Upload */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-secondary">Input File</label>
          <DragDropZone
            onFileDrop={processFile}
            className="border border-dashed rounded-lg p-6 text-center transition-all cursor-pointer relative overflow-hidden border-border hover:border-border hover:bg-primary"
            draggingClassName="border-blue-400 bg-blue-500/10 scale-[1.02] shadow-xl"
            rejectClassName="border-red-500 bg-red-500/10"
            // Simple generic accept string for now
            accept="image/*,video/*,audio/*,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain,text/markdown,.pdf,.docx,.txt,.md"
          >
            {({ isDragging, isDragReject }) => (
               // Wrapper renders children, state is passed
               <>
                 {file ? (
                   <div className={`flex flex-col items-center justify-center gap-2 ${isDragging ? 'opacity-50' : ''}`}>
                       {inputPreviewUrl ? (
                        <div className="relative h-20 w-auto min-w-[5rem] mb-1 rounded overflow-hidden border border-border bg-primary shadow-sm group-hover:border-slate-500 transition-colors">
                           <img src={inputPreviewUrl} alt="Preview" className="h-full w-full object-contain" />
                        </div>
                      ) : (
                         <div className="flex flex-col items-center gap-1">
                            <div className="w-10 h-10 bg-secondary rounded flex items-center justify-center text-primary-400 mb-1">
                              <FileType size={20} />
                            </div>
                            {file && isNonPreviewableFormat(file.name) && (
                              <span className="text-[9px] text-tertiary font-mono uppercase bg-primary px-1 py-0.5 rounded border border-border mt.05">No Preview</span>
                            )}
                         </div>
                      )}
                     <div className="text-center w-full">
                       <p className="font-medium text-primary text-xs truncate max-w-[200px] mx-auto" title={file.name}>{file.name}</p>
                       <p className="text-xs text-secondary">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                     </div>
                     <div className="text-xs text-primary-400 font-medium">
                        {isDragReject ? 'Format Not Supported' : (isDragging ? 'Drop to replace' : 'Click to change')}
                     </div>
                     {isUploading && <div className="absolute top-2 right-2"><Loader2 className="animate-spin text-tertiary" size={16} /></div>}
                   </div>
                 ) : (
                   <>
                     {isDragReject ? (
                        <>
                            <Upload size={24} className="mx-auto mb-2 text-red-400 transition-colors" />
                            <p className="font-medium text-sm text-red-200 transition-colors">Format Not Supported</p>
                        </>
                     ) : (
                        <>
                            <Upload size={24} className={`mx-auto mb-2 transition-colors ${isDragging ? 'text-blue-400' : 'text-tertiary'}`} />
                            <p className={`font-medium text-sm transition-colors ${isDragging ? 'text-blue-200' : ''}`}>Click or Drag & Drop</p>
                        </>
                     )}
                     <p className="text-[10px] text-tertiary mt-1">Supports img, vid, audio, docs</p>
                   </>
                 )}
               </>
            )}
          </DragDropZone>
        </div>

        {/* Target Format */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-secondary">Target Format</label>
          {file && availableFormats.length > 0 ? (
            <div className="grid grid-cols-3 gap-2">
              {availableFormats.map(fmt => (
                <button
                  key={fmt}
                  onClick={() => setTargetFormat(fmt)}
                  className={`py-2 px-2 rounded-md font-medium border text-xs uppercase transition-all ${targetFormat === fmt
                    ? 'bg-blue-600 border-blue-500 text-white shadow-md shadow-blue-900/20'
                    : 'bg-primary border-border text-secondary hover:border-border hover:text-primary'
                    }`}
                >
                  {fmt}
                </button>
              ))}
            </div>
          ) : (
            <div className="p-4 bg-primary border border-border rounded-lg text-center text-tertiary text-xs italic">
                {file ? "No compatible formats found." : "Upload a file to see options."}
            </div>
          )}
        </div>
        
        {/* OCR Model Selection */}
        {targetFormat.includes('(OCR)') && (
          <div className="space-y-2 animate-in fade-in slide-in-from-top-1 duration-200">
            <label className="text-sm font-medium text-secondary flex justify-between items-center">
              <span className="flex items-center">
                OCR Model
                <ModelHelpLink section="multimedia" />
              </span>
              <span className="text-[10px] text-tertiary font-normal italic">
                {ocrModel === 'qwen-vl' ? '~30GB RAM usage' : 'Fast & lightweight'}
              </span>
            </label>
            <div className="flex gap-2 p-1 bg-primary rounded-lg border border-border">
              <button
                onClick={() => setOcrModel('qwen-vl')}
                className={`flex-1 py-1.5 px-2 rounded-md text-[10px] font-bold uppercase transition-all ${
                  ocrModel === 'qwen-vl'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-tertiary hover:text-secondary'
                }`}
              >
                Qwen-VL
              </button>
              <button
                onClick={() => setOcrModel('florence')}
                className={`flex-1 py-1.5 px-2 rounded-md text-[10px] font-bold uppercase transition-all ${
                  ocrModel === 'florence'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-tertiary hover:text-secondary'
                }`}
              >
                Florence-2
              </button>
            </div>
            {ocrModel === 'qwen-vl' && (
              <p className="text-[10px] text-indigo-800 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-500/5 p-2 rounded border border-indigo-200 dark:border-indigo-500/20 italic">
                Best for high-precision code, paths, and emojis.
              </p>
            )}
          </div>
        )}

        <ErrorAlert error={error} onDismiss={() => setError(null)} />

        {/* Action Button */}
        <div className="mt-auto pt-4">
            <button
            onClick={handleConvert}
            disabled={!targetFormat || isUploading || isSubmitting}
            className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-white font-bold py-3 rounded-lg shadow-lg shadow-blue-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all"
            >
            {isSubmitting ? (
                <><Loader2 className="animate-spin" size={18} /> Converting...</>
            ) : (
                <><RefreshCw size={18} /> Convert Now</>
            )}
            </button>
        </div>
      </div>

      {/* Main Preview Area */}
      <div ref={resultRef} className="flex-1 p-6 flex items-center justify-center bg-primary/30 min-h-[500px] lg:min-h-0">
        {!result && !file && (
          <div className="text-center text-tertiary">
            <RefreshCw size={48} className="mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-medium mb-2">Ready to Convert</h3>
            <p className="text-secondary max-w-sm">
              Upload a file from the <span className="lg:hidden">controls above</span><span className="hidden lg:inline">sidebar</span> to see conversion options.
            </p>
          </div>
        )}

        {!result && file && inputPreviewUrl && (
          <div className="flex flex-col items-center justify-center max-w-full h-full gap-4">
            <div className="relative border-4 border-border rounded-xl overflow-hidden shadow-2xl max-h-[70vh]">
              <img src={inputPreviewUrl} alt="Original" className="max-h-[70vh] object-contain" />
              <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-md px-3 py-1.5 rounded-lg text-[10px] font-mono text-white border border-white/10 uppercase tracking-tighter">
                Original
              </div>
            </div>
            <p className="text-sm text-secondary">Select a format and click convert to process this file.</p>
          </div>
        )}
        
        {!result && file && !inputPreviewUrl && (
          <div className="relative border-4 border-border rounded-xl overflow-hidden shadow-2xl flex flex-col items-center justify-center p-8 bg-primary/50 min-w-[280px] min-h-[320px]">
            <FileType size={64} className="text-tertiary mb-4" />
            <div className="flex flex-col items-center text-center">
              <span className="text-lg font-bold text-secondary">{file.name.split('.').pop()?.toUpperCase()} File</span>
              <span className="text-xs text-tertiary mt-1 uppercase tracking-widest font-mono">No Browser Preview</span>
            </div>
            <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-md px-3 py-1.5 rounded-lg text-[10px] font-mono text-white border border-white/10 uppercase tracking-tighter">
              Original
            </div>
          </div>
        )}

        {result && (
           <div className="flex flex-col items-center justify-center max-w-full h-full gap-6 animate-in fade-in zoom-in duration-300">
              <div className="relative border-4 border-brand-500/30 rounded-xl overflow-hidden shadow-2xl flex flex-col items-center justify-center bg-primary/50 min-w-[280px] min-h-[320px] cursor-pointer group" onClick={() => setIsPreviewOpen(true)}>
                {/* Result Preview Logic */}
                {['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp'].includes(result.split('.').pop()?.toLowerCase() || '') ? (
                  <img src={`${API_BASE_URL()}/api/files/${result}`} alt="Converted Result" className="max-h-[70vh] object-contain" />
                ) : ['mp4', 'webm', 'mov', 'mkv'].includes(result.split('.').pop()?.toLowerCase() || '') ? (
                  <video src={`${API_BASE_URL()}/api/files/${result}`} controls className="max-h-[70vh]" />
                ) : (
                  <>
                    <FileType size={64} className="text-tertiary mb-4" />
                    <div className="flex flex-col items-center text-center">
                      <span className="text-lg font-bold text-secondary">{result.split('.').pop()?.toUpperCase()} File</span>
                      <span className="text-xs text-tertiary mt-1 uppercase tracking-widest font-mono">No Browser Preview</span>
                    </div>
                  </>
                )}

               <div className="absolute top-2 left-2 bg-blue-600 px-2 py-1 rounded text-xs text-white shadow-lg flex flex-col items-start leading-none gap-0.5">
                   <span className="font-bold uppercase tracking-wider">Converted</span>
                   {genDuration && <span className="opacity-80 font-medium">in {formatDuration(genDuration * 1000)}</span>}
               </div>

                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                    <span className="bg-white/90 text-black px-4 py-2 rounded-lg font-bold text-sm">Open Full Preview</span>
                </div>
             </div>

             <div className="flex gap-3">
               <button className="btn-secondary text-sm" onClick={() => setIsPreviewOpen(true)}>Full Preview</button>
               <a href={`${API_BASE_URL()}/api/files/${result}?download=true`} className="btn-secondary text-sm">Download File</a>
             </div>
           </div>
        )}
      </div>

      {currentJobId && (
        <JobProgressModal
          jobId={currentJobId}
          onClose={handleCloseModal}
          onViewResult={handleViewResult}
        />
      )}

      {result && (isPreviewOpen || isSubmitting) && (
        ['jpg', 'jpeg', 'png', 'webp', 'gif', 'tiff', 'tif', 'bmp'].includes(result.split('.').pop()?.toLowerCase() || '') && (inputPreviewUrl || serverFilePath) ? (
          <ComparisonPreviewModal
            isOpen={isPreviewOpen}
            onClose={() => setIsPreviewOpen(false)}
            originalPath={inputPreviewUrl || serverFilePath || ''}
            resultPath={result}
            fileName={result.split('/').pop() || 'converted-file'}
            resultLabel="Converted"
            originalFormat={file?.name.split('.').pop()}
            resultFormat={result.split('.').pop()}
          />
        ) : (
          <PreviewModal
            isOpen={isPreviewOpen}
            onClose={() => setIsPreviewOpen(false)}
            filePath={result}
            fileName={result.split('/').pop() || 'converted-file'}
          />
        )
      )}
    </div>
  );
}
