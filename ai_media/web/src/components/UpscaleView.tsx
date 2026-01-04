import { useState, useRef, useEffect } from 'react';
import { useAppStore } from '../store';
import { TrendingUp, TrendingDown, Zap, Loader2, Wand2, Image as ImageIcon, ArrowRight, ChevronsUp, ChevronsDown, Upload } from 'lucide-react';
import { API_BASE_URL } from '../config';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { ComparisonPreviewModal } from './common/ComparisonPreviewModal';
import { ErrorAlert } from './common/ErrorAlert';
import { ResourceWarningModal } from './common/ResourceWarningModal';
import { NumberInput } from './common/NumberInput';
import { DragDropZone } from './common/DragDropZone';
import { formatDuration } from '../utils/formatTime';

// Check if a file extension can't be previewed in browsers
const isNonPreviewableFormat = (filename: string): boolean => {
  const ext = filename.split('.').pop()?.toLowerCase();
  return ['tiff', 'tif', 'psd', 'raw'].includes(ext || '');
};

export function UpscaleView() {
  const { addJob } = useAppStore();
  const [file, setFile] = useState<File | null>(null);
  const [inputPreview, setInputPreview] = useState<string | null>(null);
  const [serverFilePath, setServerFilePath] = useState<string | null>(null);

  const [factor, setFactor] = useState(2.0); // 2 or 4
  const [method, setMethod] = useState('fast'); // fast, ai, simple
  const [strength, setStrength] = useState(0.3); // 0.0 - 1.0 (denoising)
  const [isUploading, setIsUploading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [genDuration, setGenDuration] = useState<number | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  const [showResourceWarning, setShowResourceWarning] = useState(false);
  const [resourceWarningData, setResourceWarningData] = useState<any>(null);

  const [isCustomFactor, setIsCustomFactor] = useState(false);
  const [processedFactor, setProcessedFactor] = useState<number | null>(null);
  const isDownscale = factor < 1.0;
  const isResultDownscale = processedFactor !== null && processedFactor < 1.0;

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setResult(null);
      setError(null);
      setCurrentJobId(null);
      setShowResourceWarning(false);

      // Preview if image
      if (selectedFile.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (ev) => setInputPreview(ev.target?.result as string);
        reader.readAsDataURL(selectedFile);
      } else {
        setInputPreview(null);
      }

      // Upload
      setIsUploading(true);
      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const response = await fetch(`${API_BASE_URL}/api/upload`, {
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
    }
  };

  // Handler for DragDropZone - converts File to ChangeEvent-like structure
  const handleFileDrop = async (droppedFile: File) => {
    setFile(droppedFile);
    setResult(null);
    setError(null);
    setCurrentJobId(null);
    setShowResourceWarning(false);

    // Preview if image and not non-previewable
    if (droppedFile.type.startsWith('image/') && !isNonPreviewableFormat(droppedFile.name)) {
      const reader = new FileReader();
      reader.onload = (ev) => setInputPreview(ev.target?.result as string);
      reader.readAsDataURL(droppedFile);
    } else {
      setInputPreview(null);
    }

    // Upload
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', droppedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
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

  const startUpscaleJob = async () => {
    setIsSubmitting(true);
    setError(null);
    setResult(null);
    setGenDuration(null);
    setShowResourceWarning(false);
    setProcessedFactor(factor);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upscale`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_path: serverFilePath,
          factor: factor,
          method: method,
          strength: strength,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Upscale request failed');
      }

      const data = await response.json();
      setCurrentJobId(data.job_id);

      addJob({
        job_id: data.job_id,
        type: 'upscale',
        status: 'pending',
        progress: 0,
        phase: 'queued',
        message: 'Job queued',
        result_path: null,
        error: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });

    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to start upscale");
      setIsSubmitting(false);
    }
  };

  const handleUpscaleClick = async () => {
    if (!serverFilePath) return;

    // Validate first
    setIsSubmitting(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/upscale/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_path: serverFilePath,
          factor: factor,
          method: method,
          strength: strength
        })
      });

      const data = await res.json();

      if (data.status === 'warning' || data.status === 'critical') {
        setResourceWarningData({
          warning: data.warning,
          type: data.warning_type,
          details: data.details
        });
        setShowResourceWarning(true);
        setIsSubmitting(false); // Stop loader while waiting for user
      } else {
        // Safe to proceed
        startUpscaleJob();
      }
    } catch (e) {
      console.error("Validation failed, proceeding anyway", e);
      startUpscaleJob();
    }
  };

  // Watch for job completion
  useEffect(() => {
    if (!currentJobId) return;

    const unsubscribe = useAppStore.subscribe((state) => {
      const job = state.jobs.find(j => j.job_id === currentJobId);
      if (job) {
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
          setError(job.error || job.message || "Upscale failed");
        } else if (job.status === 'cancelled') {
          setIsSubmitting(false);
          setError("Job cancelled.");
          setTimeout(() => setError(null), 6000);
        }
      } else {
        // Job not found in store (removed on cancellation)
        setIsSubmitting(false);
        setCurrentJobId(null);
      }
    });
    return () => unsubscribe();
  }, [currentJobId]);

  return (
    <div className="flex flex-col lg:flex-row h-full bg-slate-900 text-slate-200">
      {/* Sidebar Params */}
      <div className="w-full lg:w-[500px] border-b lg:border-b-0 lg:border-r border-slate-800 p-4 lg:py-6 lg:pr-[27px] lg:pl-1 flex flex-col gap-6 overflow-y-auto shrink-0 h-auto lg:h-full">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
            {isDownscale ? <TrendingDown className="text-emerald-400" /> : <TrendingUp className="text-emerald-400" />} {isDownscale ? "Downscale" : "Upscale"}
          </h2>
          <p className="text-xs text-slate-500">Enhance resolution and details</p>
        </div>

        {/* Upload */}
        <DragDropZone
          className="border-2 border-dashed border-slate-700 rounded-lg p-6 flex flex-col items-center justify-center gap-2 hover:border-emerald-500/50 hover:bg-slate-800/50 transition-colors cursor-pointer relative"
          draggingClassName="border-emerald-500 bg-emerald-500/10"
          onFileDrop={handleFileDrop}
          accept="image/*,video/*,.tiff,.tif"
        >
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept="image/*,video/*,.tiff,.tif"
            onChange={handleFileSelect}
          />

          {file && isNonPreviewableFormat(file.name) ? (
            <div className="flex flex-col items-center gap-2 py-2">
              <div className="w-16 h-16 bg-slate-700/50 rounded-lg flex items-center justify-center border border-slate-600">
                <ImageIcon size={28} className="text-slate-500" />
              </div>
              <span className="text-[10px] text-slate-500 bg-slate-800 px-2 py-0.5 rounded">
                {file.name.split('.').pop()?.toUpperCase()} - No Preview
              </span>
            </div>
          ) : inputPreview ? (
            <img src={inputPreview} alt="Preview" className="max-h-32 object-contain rounded" />
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload size={28} className="text-slate-400" />
              <span className="text-sm text-slate-400">Click or Drag & Drop</span>
            </div>
          )}

          <div className="text-center">
            <p className="text-sm font-medium">{file ? file.name : "Upload Media"}</p>
            <p className="text-xs text-slate-500">Image or Video</p>
          </div>

          {isUploading && (
            <div className="absolute inset-0 bg-black/60 flex items-center justify-center rounded-lg">
              <Loader2 className="animate-spin text-emerald-400" />
            </div>
          )}
        </DragDropZone>

        {/* Controls */}
        <div className="space-y-6">

          {/* Factor */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-400">Scale Factor</label>
            <div className="flex flex-wrap gap-2 bg-slate-950 p-1 rounded-lg border border-slate-800">
              {[2.0, 4.0, 8.0].map(f => (
                <button
                  key={f}
                  className={`flex-1 py-1.5 px-2 text-sm font-medium rounded transition-all ${!isCustomFactor && factor === f ? 'bg-slate-800 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
                  onClick={() => {
                    setFactor(f);
                    setIsCustomFactor(false);
                  }}
                >
                  {f}x
                </button>
              ))}
              <button
                className={`flex-1 py-1.5 px-2 text-sm font-medium rounded transition-all ${isCustomFactor ? 'bg-primary-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
                onClick={() => setIsCustomFactor(true)}
                title="Custom factor (allows higher numbers and < 1.0 effectively for downscaling)"
              >
                Custom
              </button>
            </div>

            {isCustomFactor && (
              <div className="animate-in fade-in slide-in-from-top-1">
                <NumberInput
                  value={factor}
                  onChange={setFactor}
                  min={0.1}
                  max={method === 'ai' ? 16.0 : 128.0}
                  step={0.1}
                  allowFloat={true}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary-500 transition-colors"
                  placeholder="Enter scale (e.g. 0.5, 2x)"
                  title={`Enter value (0.1 - ${method === 'ai' ? '16.0' : '128.0'}). Use values < 1.0 to downscale (e.g. 0.5)`}
                />
                {method === 'ai' && (
                  <p className="text-[10px] text-slate-500 mt-1 text-right">Max recommended for AI models: 8x</p>
                )}
              </div>
            )}
          </div>

          {/* Method */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-slate-400">Method</label>

            {/* AI Group */}
            <div className="space-y-2">
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider pl-1">AI Upscaling</p>
              <div className="grid grid-cols-1 gap-2">
                <button
                  onClick={() => setMethod('fast')}
                  className={`flex items-center gap-3 p-3 rounded-lg border text-left transition-all ${method === 'fast'
                      ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-900'
                    }`}
                >
                  <Zap size={18} />
                  <div>
                    <div className="font-medium text-sm">Fast</div>
                    <div className="text-[10px] opacity-70">Real-ESRGAN (Best for clean styles)</div>
                  </div>
                </button>

                <button
                  onClick={() => setMethod('ai')}
                  className={`flex items-center gap-3 p-3 rounded-lg border text-left transition-all ${method === 'ai'
                      ? 'bg-purple-500/10 border-purple-500/50 text-purple-400'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-900'
                    }`}
                >
                  <Wand2 size={18} />
                  <div>
                    <div className="font-medium text-sm">Creative</div>
                    <div className="text-[10px] opacity-70">Latent Upscale (SD x2/x4)</div>
                  </div>
                </button>
              </div>
            </div>

            {/* Traditional Group */}
            <div className="space-y-2">
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider pl-1">Traditional</p>
              <button
                onClick={() => setMethod('simple')}
                className={`w-full flex items-center gap-3 p-3 rounded-lg border text-left transition-all ${method === 'simple'
                    ? 'bg-blue-500/10 border-blue-500/50 text-blue-400'
                    : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-900'
                  }`}
              >
                <ImageIcon size={18} />
                <div>
                  <div className="font-medium text-sm">Simple</div>
                  <div className="text-[10px] opacity-70">Lanczos interpolation (No AI)</div>
                </div>
              </button>
            </div>
          </div>

          {/* Denoise Strength (Only for AI) */}
          {method === 'ai' && (
            <div className="space-y-2 animate-in fade-in slide-in-from-top-2 duration-200">
              <label className="text-sm font-medium text-slate-400 flex justify-between">
                Creativity / Denoise
                <span className="text-white">{strength}</span>
              </label>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={strength}
                onChange={(e) => setStrength(parseFloat(e.target.value))}
                className="w-full accent-emerald-500 h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-slate-500">Higher = more imagined details (modifies original)</p>
            </div>
          )}
        </div>

        <button
          onClick={handleUpscaleClick}
          disabled={!serverFilePath || isUploading || isSubmitting}
          className={`mt-auto w-full relative overflow-hidden bg-[length:100%_200%] bg-[linear-gradient(to_top,#10b981,#14b8a6,#10b981,#14b8a6,#10b981)] text-white font-bold py-3 rounded-lg shadow-lg shadow-emerald-900/20 disabled:opacity-50 disabled:grayscale flex items-center justify-center gap-2 transition-all group hover:shadow-emerald-900/40 ${serverFilePath && !isUploading && !isSubmitting ? (isDownscale ? 'animate-gradient-y-reverse' : 'animate-gradient-y') : ''
            }`}
        >
          {/* Decorative Animated Arrows */}
          {!isSubmitting && !(!serverFilePath || isUploading) && (
            <>
              <div className="absolute left-3 inset-y-0 flex items-center justify-center">
                {isDownscale ? (
                  <ChevronsDown className="text-white blur-[1px] animate-shimmer-down" size={24} />
                ) : (
                  <ChevronsUp className="text-white blur-[1px] animate-shimmer-up" size={24} />
                )}
              </div>
              <div className="absolute right-3 inset-y-0 flex items-center justify-center">
                {isDownscale ? (
                  <ChevronsDown className="text-white blur-[1px] animate-shimmer-down" size={24} />
                ) : (
                  <ChevronsUp className="text-white blur-[1px] animate-shimmer-up" size={24} />
                )}
              </div>
            </>
          )}

          <div className="relative z-10 flex items-center gap-2">
            {isSubmitting ? (
              <><Loader2 className="animate-spin" size={18} /> {isDownscale ? "Downscaling..." : "Upscaling..."}</>
            ) : (
              <>{isDownscale ? <TrendingDown size={18} /> : <TrendingUp size={18} />} {isDownscale ? "Start Downscale" : "Start Upscale"}</>
            )}
          </div>
        </button>

        <ErrorAlert error={error} onDismiss={() => setError(null)} />
      </div>

      <div className="flex-1 p-8 flex items-center justify-center bg-slate-950/30">
        {file && isNonPreviewableFormat(file.name) ? (
          /* TIFF/Non-Previewable File Placeholder */
          <div className="flex flex-col md:flex-row items-stretch gap-6 max-w-full">
            <div className="relative border-4 border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col items-center justify-center p-8 bg-slate-900/50 min-w-[280px] min-h-[320px]">
              <ImageIcon size={64} className="text-slate-600 mb-4" />
              <div className="flex flex-col items-center text-center">
                <span className="text-lg font-bold text-slate-300">{file.name.split('.').pop()?.toUpperCase()} File</span>
                <span className="text-xs text-slate-500 mt-1 uppercase tracking-widest font-mono">No Browser Preview</span>
              </div>
              <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-md px-3 py-1.5 rounded-lg text-[10px] font-mono text-white border border-white/10 uppercase tracking-tighter">
                Original
              </div>
            </div>

            {/* Arrow */}
            {result && <ArrowRight className="hidden md:block text-slate-600" size={32} />}

            {/* Upscaled Result */}
            {result && (
              <div className="relative border-4 border-brand-500/30 rounded-xl overflow-hidden shadow-2xl flex flex-col items-center justify-center bg-slate-900/50 min-w-[280px] min-h-[320px] cursor-pointer group" onClick={() => setIsPreviewOpen(true)}>
                {isNonPreviewableFormat(result) ? (
                  <>
                    <ImageIcon size={64} className="text-slate-600 mb-4" />
                    <div className="flex flex-col items-center text-center">
                      <span className="text-lg font-bold text-slate-300">{result.split('.').pop()?.toUpperCase()} File</span>
                      <span className="text-xs text-slate-500 mt-1 uppercase tracking-widest font-mono">No Browser Preview</span>
                    </div>
                  </>
                ) : (
                  <img src={`http://localhost:8000/api/files/${result}`} className="max-w-full max-h-[70vh] object-contain" />
                )}
                <div className="absolute top-4 left-4 bg-brand-600 backdrop-blur-md px-3 py-1.5 rounded-lg border border-brand-400/20 shadow-lg flex flex-col items-start leading-none gap-1">
                  <span className="text-[10px] font-mono text-white uppercase tracking-tighter">
                    {isResultDownscale ? "Downscaled" : "Upscaled"} {processedFactor}x
                  </span>
                  {genDuration && <span className="text-[9px] text-brand-200 font-medium opacity-80 uppercase tracking-widest">in {formatDuration(genDuration * 1000)}</span>}
                </div>
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                  <span className="bg-white/90 text-black px-4 py-2 rounded-lg font-bold text-sm">Open Full Preview</span>
                </div>
              </div>
            )}
          </div>
        ) : inputPreview ? (
          <div className="flex flex-col md:flex-row items-center gap-6 max-w-full">
            {/* Original */}
            <div className="relative border-4 border-slate-800 rounded-xl overflow-hidden shadow-2xl max-w-full">
              <img src={inputPreview} className="max-w-full max-h-[70vh] object-contain" />
              <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-md px-3 py-1.5 rounded-lg text-[10px] font-mono text-white border border-white/10 uppercase tracking-tighter">
                Original
              </div>
            </div>

            {/* Arrow */}
            {result && <ArrowRight className="hidden md:block text-slate-600" size={32} />}

            {/* Upscaled Result */}
            {result && (
              <div className="relative border-4 border-brand-500/30 rounded-xl overflow-hidden shadow-2xl max-w-full cursor-pointer group" onClick={() => setIsPreviewOpen(true)}>
                <img src={`http://localhost:8000/api/files/${result}`} className="max-w-full max-h-[70vh] object-contain" />
                <div className="absolute top-4 left-4 bg-brand-600 backdrop-blur-md px-3 py-1.5 rounded-lg border border-brand-400/20 shadow-lg flex flex-col items-start leading-none gap-1">
                  <span className="text-[10px] font-mono text-white uppercase tracking-tighter">
                    {isResultDownscale ? "Downscaled" : "Upscaled"} {processedFactor}x
                  </span>
                  {genDuration && <span className="text-[9px] text-brand-200 font-medium opacity-80 uppercase tracking-widest">in {formatDuration(genDuration * 1000)}</span>}
                </div>
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                  <span className="bg-white/90 text-black px-4 py-2 rounded-lg font-bold text-sm">Open Full Preview</span>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center text-slate-500">
            <TrendingUp size={48} className="mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-medium mb-2">Ready to Upscale</h3>
            <p className="text-slate-400 max-w-sm">
              Upload an image/video from the <span className="lg:hidden">controls above</span><span className="hidden lg:inline">sidebar</span> to start upscaling.
            </p>
          </div>
        )}
      </div>

      {currentJobId && (
        <JobProgressModal
          jobId={currentJobId}
          onClose={() => {
            setCurrentJobId(null);
            if (result) setIsPreviewOpen(true);
          }}
        />
      )}

      {result && (isPreviewOpen || isSubmitting) && (
        ['jpg', 'jpeg', 'png', 'webp', 'gif', 'tiff', 'tif', 'bmp'].includes(result.split('.').pop()?.toLowerCase() || '') && (inputPreview || serverFilePath) ? (
          <ComparisonPreviewModal
            isOpen={isPreviewOpen}
            onClose={() => setIsPreviewOpen(false)}
            originalPath={inputPreview || serverFilePath || ''}
            resultPath={result}
            fileName={result.split('/').pop() || (isResultDownscale ? 'downscaled-file' : 'upscaled-file')}
            resultLabel={`${isResultDownscale ? "Downscaled" : "Upscaled"} ${processedFactor || factor}x`}
            factor={processedFactor || factor}
            originalFormat={file?.name.split('.').pop()}
            resultFormat={result.split('.').pop()}
          />
        ) : (
          <PreviewModal
            isOpen={isPreviewOpen}
            onClose={() => setIsPreviewOpen(false)}
            filePath={result}
            fileName={result.split('/').pop() || 'upscaled.png'}
          />
        )
      )}

      {showResourceWarning && resourceWarningData && (
        <ResourceWarningModal
          isOpen={showResourceWarning}
          onConfirm={startUpscaleJob}
          onCancel={() => setShowResourceWarning(false)}
          warning={resourceWarningData.warning}
          type={resourceWarningData.type}
          details={resourceWarningData.details}
        />
      )}
    </div>
  );
}
