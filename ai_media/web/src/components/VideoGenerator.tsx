import { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../store';
import { generateVideo, uploadFile, useModels } from '../hooks/useApi';
import { API_BASE_URL } from '../config';
import { Film, Loader2, AlertTriangle, Info, Upload, X } from 'lucide-react';
import { NumberInput } from './common/NumberInput';
import { Tooltip } from './common/Tooltip';
import { ResolutionSelector } from './common/ResolutionSelector';
import { ValidationTooltip } from './common/ValidationTooltip';
import { RandomPrompt } from './common/RandomPrompt';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { ErrorAlert } from './common/ErrorAlert';
import { ModelHelpLink } from './common/ModelHelpLink';
import { formatDuration } from '../utils/formatTime';
import { HelpCircle } from 'lucide-react';
import { getDynamicRam } from '../utils/modelResources';
import { ResourceWarningModal } from './common/ResourceWarningModal';



// Display info matching CLI
const MODEL_DISPLAY_INFO: Record<string, { label: string }> = {
  'zeroscope': { label: 'Zeroscope (Fast, 576x320)' },
  'zeroscope-xl': { label: 'Zeroscope XL (Higher Res)' },
  'ms-1.7b': { label: 'ModelScope 1.7B (Watermark Issues)' },
  'cogvideox': { label: 'CogVideoX-5b (High Qual, 38GB VRAM!)' },
  'wan-2.2-5b': { label: 'Wan 2.2 (5B)' },
  'wan-2.2': { label: 'Wan 2.2 (14B)' },
  'ltx-video': { label: 'LTX-Video (Fast, High Res)' },
  'mochi-1': { label: 'Mochi-1 (Physics SOTA)' },
  'hunyuan': { label: 'HunyuanVideo (13B, Cinematic)' },
  'svd': { label: 'Stable Video Diffusion (Image-to-Video)' },
};

const MODEL_ORDER = [
  'zeroscope', 'zeroscope-xl', 'ms-1.7b', 'cogvideox', 'wan-2.2-5b', 'wan-2.2',
  'ltx-video', 'mochi-1', 'hunyuan', 'svd'
];

export function VideoGenerator() {
  const { addJob } = useAppStore();
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState('zeroscope');
  const [width, setWidth] = useState(576);
  const [height, setHeight] = useState(320);
  const [duration, setDuration] = useState(4);
  const [fps, setFps] = useState(24);
  const [framework, setFramework] = useState(navigator.userAgent.toLowerCase().includes('mac') ? 'mlx' : 'auto');
  const [precision, setPrecision] = useState("auto");
  const [format, setFormat] = useState("mp4");

  const [isLoading, setIsLoading] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [genDuration, setGenDuration] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [showWarning, setShowWarning] = useState(false);
  const [pendingGeneration, setPendingGeneration] = useState<boolean>(false);

  // High resource text for the model that triggered the warning
  const [warningDetails, setWarningDetails] = useState<{
    message: string;
    details?: any;
    critical?: boolean;
  } | null>(null);

  // Image Input State (for I2V)
  const [inputImage, setInputImage] = useState<File | null>(null);
  const [inputImagePreview, setInputImagePreview] = useState<string | null>(null);
  const [inputImagePath, setInputImagePath] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  // Use global models cache
  const { models } = useModels();
  const availableModels = models?.video || [];


  // Update defaults when model changes
  useEffect(() => {
    // Use CLI defaults for everything: 720p (1280x720) and 2s duration
    // This triggers auto-upscaling for zeroscope in the backend, matching CLI behavior.
    setWidth(1280);
    setHeight(720);
    setDuration(2);

    // Initial FPS estimation (backend handles actual consistency)
    if (model === 'zeroscope' || model === 'zeroscope-xl' || model === 'svd') {
      setFps(8);
    } else if (model === 'wan-2.2' || model === 'wan-2.2-5b') {
      setFps(15);
    } else if (model === 'cogvideox') {
      setFps(8);
    } else {
      setFps(24);
    }
  }, [model]);

  const supportsImageInput = ['wan-2.2', 'wan-2.2-5b', 'svd', 'cogvideox', 'hunyuan'].includes(model);

  const handleImageUpload = async (file: File) => {
    setIsUploading(true);
    setError(null);
    try {
      // Create local preview
      const previewUrl = URL.createObjectURL(file);
      setInputImagePreview(previewUrl);
      setInputImage(file);

      // Upload to server
      const result = await uploadFile(file);
      setInputImagePath(result.path);
    } catch (err) {
      console.error("Upload failed", err);
      setError("Failed to upload input image");
      setInputImage(null);
      setInputImagePreview(null);
    } finally {
      setIsUploading(false);
    }
  };

  const handleRemoveImage = () => {
    setInputImage(null);
    setInputImagePath(null);
    if (inputImagePreview) URL.revokeObjectURL(inputImagePreview);
    setInputImagePreview(null);
  };

  const executeGeneration = async (force: boolean = false) => {
    setIsLoading(true);
    setResult(null);
    setGenDuration(null);
    setError(null);
    setShowWarning(false);
    setPendingGeneration(false);

    try {
      const response = await generateVideo({
        prompt,
        model,
        width,
        height,
        duration,
        input_image: inputImagePath || undefined,
        framework: framework === 'auto' ? undefined : framework,
        precision: precision === 'auto' ? undefined : precision,
        format,
        force,
      });

      setCurrentJobId(response.job_id);

      addJob({
        job_id: response.job_id,
        type: 'video',
        status: 'pending',
        progress: 0,
        phase: 'queued',
        message: 'Job queued',
        result_path: response.output_path,
        error: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });

    } catch (err) {
      console.error('Generation failed:', err);
      setIsLoading(false);
      setCurrentJobId(null);
      setError("Failed to start generation job");
    }
  };

  const handleGenerate = async () => {
    // Some models like SVD allow image only, but we usually want a prompt too or default one
    if (!prompt.trim() && !inputImagePath) return;

    // Check for High Resource usage
    const ramStr = getDynamicRam(model, precision, framework);
    // Parse "~14GB" -> 14
    const estimatedRam = parseInt(ramStr.replace(/[^0-9.]/g, '') || "0");

    const sys = useAppStore.getState().systemInfo;
    const resources = useAppStore.getState().resources;
    const totalRam = sys?.ram_total_gb || 0;
    const usedRam = resources?.global.ram_used_gb || 0;
    const freeRam = resources ? (totalRam - usedRam) : (totalRam * 0.5);

    // Warn logic: >85% of total, or > free RAM, or > 32GB absolute
    const needsWarning = (totalRam > 0 && estimatedRam > totalRam * 0.85) || estimatedRam > freeRam || estimatedRam > 32;

    if (needsWarning) {
      setWarningDetails({
        message: `This model configuration requires ~${estimatedRam}GB of RAM. It may exceed your system's available memory (${Math.round(freeRam)}GB free / ${Math.round(totalRam)}GB total).`,
        critical: totalRam > 0 && estimatedRam > totalRam,
        details: {
          estimated_ram_gb: estimatedRam,
          available_ram_gb: Math.round(freeRam * 10) / 10,
          total_ram_gb: totalRam
        }
      });
      setShowWarning(true);
      setPendingGeneration(true);
      return;
    }

    executeGeneration(false);
  };

  // Watch job status via store subscription
  useEffect(() => {
    if (!currentJobId) return;

    let hasSeenJob = false;

    const unsubscribe = useAppStore.subscribe((state) => {
      const updatedJob = state.jobs.find(j => j.job_id === currentJobId);
      if (updatedJob) {
        hasSeenJob = true;

        if (updatedJob.status === 'complete') {
          setResult(updatedJob.result_path);

          // Calculate generation duration
          if (updatedJob.generation_started_at && updatedJob.updated_at) {
            const start = new Date(updatedJob.generation_started_at).getTime();
            const end = new Date(updatedJob.updated_at).getTime();
            const seconds = Math.round((end - start) / 1000);
            setGenDuration(seconds > 0 ? seconds : 1);
          } else if (updatedJob.created_at && updatedJob.updated_at) {
            const start = new Date(updatedJob.created_at).getTime();
            const end = new Date(updatedJob.updated_at).getTime();
            setGenDuration(Math.round((end - start) / 1000));
          }

          setIsLoading(false);
        } else if (updatedJob.status === 'failed') {
          setIsLoading(false);
          setError(updatedJob.error || updatedJob.message || "Generation failed");
        } else if (updatedJob.status === 'cancelled') {
          setIsLoading(false);
          setError("Job cancelled.");
          setTimeout(() => setError(null), 6000);
        }
      } else if (hasSeenJob) {
        // Job was in store but now removed (e.g., cancelled)
        setIsLoading(false);
        setCurrentJobId(null);
      }
    });
    return () => unsubscribe();
  }, [currentJobId]);

  const handleCloseModal = () => {
    setCurrentJobId(null);
    setIsLoading(false);
  };

  // Sort models
  const sortedModels = MODEL_ORDER.filter(name =>
    availableModels.some(m => m.name === name)
  );

  const resultRef = useRef<HTMLDivElement>(null);

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
            <Film className="text-brand-400" /> Video Gen
          </h2>
          <p className="text-xs text-tertiary">Generate videos from text descriptions</p>
        </div>

        {/* Prompt */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="label">Prompt</label>
            <RandomPrompt type="video" onPromptSelect={setPrompt} />
          </div>
          <textarea
            className="w-full bg-primary border border-border rounded-lg p-3 text-sm focus:outline-none focus:border-brand-500 resize-y min-h-[120px]"
            placeholder="Enter your video prompt or use the Random Prompt tool..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        {/* Framework Selector (Mac only) */}
        <div className={`space-y-1 ${!navigator.userAgent.toLowerCase().includes('mac') ? 'hidden' : ''}`}>
          <label className="label">Platform</label>
          <select
            className="select w-full bg-primary border-border text-sm focus:border-brand-500"
            value={framework}
            onChange={(e) => setFramework(e.target.value)}
            disabled={isLoading}
            title="Inference Framework - Use MLX for best performance on Mac"
          >
            <option value="mlx">MLX (Native Mac)</option>
            <option value="torch">PyTorch (MPS)</option>
          </select>
        </div>

        {/* Precision Selector */}
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <label className="label">Precision</label>
            <button
              onClick={() => useAppStore.getState().openHelpSection('precision')}
              className="text-tertiary hover:text-brand-500 transition-colors"
              title="Learn about precision options"
            >
              <HelpCircle size={14} />
            </button>
          </div>
          <select
            className="select w-auto bg-primary border-border text-sm focus:border-brand-500 max-w-full"
            value={precision}
            onChange={(e) => setPrecision(e.target.value)}
            disabled={isLoading}
            title="Model precision - affects speed and memory usage"
          >
            <option value="auto">
              {(() => {
                const isMac = navigator.userAgent.toLowerCase().includes('mac');
                const isMlx = framework === 'mlx' || (framework === 'auto' && isMac);
                return `Auto (${isMlx ? 'int4 - MLX Default' : 'float16 - Default'})`;
              })()}
            </option>
            <option value="int4">int4 (4-bit, Fast)</option>
            <option value="int6">int6 (6-bit, Balanced Speed)</option>
            <option value="int8">int8 (8-bit, Balanced Quality)</option>
            <option value="float16">float16 (Standard)</option>
            <option value="bfloat16">bfloat16 (Brain Float)</option>
            <option value="float32">float32 (Slow, Max Quality)</option>
          </select>
        </div>

        {/* Model Selector */}
        <div className="space-y-2">
          <label className="label flex items-center">
            Model
            <ModelHelpLink section="video" />
          </label>
          <select
            className="select w-full bg-primary border-border text-sm focus:border-brand-500"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            {sortedModels.map((name) => {
              const info = MODEL_DISPLAY_INFO[name];
              const vram = getDynamicRam(name, precision, framework);
              const isHighRam = parseInt(vram.replace(/[^0-9]/g, '')) > 24;

              return (
                <option key={name} value={name}>
                  {info ? `${isHighRam ? '⚠️ ' : ''}${info.label} (${vram})` : name}
                </option>
              );
            })}
          </select>
          {/* Warnings */}
          {model === 'cogvideox' && (
            <div className="flex items-center gap-2 text-red-600 dark:text-red-400 text-xs mt-1">
              <AlertTriangle size={12} />
              <span>Extremely high VRAM (38GB+)</span>
            </div>
          )}
          {(model === 'wan-2.2' || model === 'wan-2.2-5b') && (
            <div className="flex items-center gap-2 text-amber-600 dark:text-amber-400 text-xs mt-1">
              <AlertTriangle size={12} />
              <span>Requires HF Login</span>
            </div>
          )}
          {model === 'svd' && (
            <div className="flex items-center gap-2 text-blue-600 dark:text-blue-400 text-xs mt-1">
              <Info size={12} />
              <span>Requires input image (not supported in text mode)</span>
            </div>
          )}
          {(model === 'wan-2.2' || model === 'wan-2.2-5b') && (
            <div className="flex items-center gap-2 mt-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
              <div className="text-xl">⚠️</div>
              <div className="text-sm text-blue-200">
                <strong>Wan 2.2 Recommendation:</strong> 720p is the native training resolution.
                <br />
                {model === 'wan-2.2-5b' ? '5B Model runs on consumer GPUs (12GB+)' : '14B Model requires 24GB+ VRAM'}
              </div>
            </div>
          )}
        </div>

        {/* Image Input (Conditional) */}
        {supportsImageInput && (
          <div className="space-y-2">
            <label className="label flex items-center justify-between">
              Initial Image (Optional)
              <span className="text-[10px] bg-secondary px-1.5 py-0.5 rounded text-tertiary">I2V Mode</span>
            </label>

            {!inputImagePreview ? (
              <div
                className={`border-2 border-dashed border-border rounded-lg p-6 flex flex-col items-center justify-center text-tertiary transition-colors ${isUploading ? 'opacity-50' : 'hover:border-brand-500 hover:bg-secondary/30'}`}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const file = e.dataTransfer.files[0];
                  if (file && file.type.startsWith('image/')) handleImageUpload(file);
                }}
              >
                <input
                  type="file"
                  className="hidden"
                  id="video-input-image"
                  accept="image/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleImageUpload(file);
                  }}
                />
                <label htmlFor="video-input-image" className="cursor-pointer flex flex-col items-center gap-2">
                  {isUploading ? <Loader2 className="animate-spin" /> : <Upload size={24} />}
                  <span className="text-xs">
                    {isUploading ? "Uploading..." : "Click or Drop Image"}
                  </span>
                </label>
              </div>
            ) : (
              <div className="relative rounded-lg overflow-hidden border border-border group bg-black/20">
                <img src={inputImagePreview} alt="Input" className="w-full h-32 object-contain" />
                <button
                  onClick={handleRemoveImage}
                  className="absolute top-1 right-1 p-1 bg-black/50 hover:bg-red-500/80 rounded-full text-white transition-colors"
                >
                  <X size={14} />
                </button>
                <div className="absolute bottom-0 left-0 right-0 bg-black/60 p-1 text-[10px] text-white truncate px-2">
                  {inputImage?.name}
                </div>
              </div>
            )}

            {(model === 'wan-2.2' || model === 'wan-2.2-5b') && inputImagePreview && (
              <div className="text-[10px] text-emerald-400 flex items-center gap-1">
                <Info size={10} />
                <span>Wan 2.2 will preserve mostly structure & motion</span>
              </div>
            )}
          </div>
        )}

        {/* Resolution */}
        <div className="space-y-2">
          <ResolutionSelector
            width={width}
            height={height}
            onChange={(w, h) => { setWidth(w); setHeight(h); }}
          />
        </div>

        {/* Duration/FPS */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="label flex items-center gap-1">
              Duration (s)
              <Tooltip content="Length of video in seconds. Default 2s. Longer videos take significantly more VRAM/time." align="left" />
            </label>
            <NumberInput value={duration} onChange={setDuration} min={1} max={10} step={0.1} allowFloat={true} />
          </div>
          <div className="space-y-1">
            <label className="label flex items-center gap-1">
              FPS (Read Only)
              <Tooltip content="Frames Per Second. Determined by model (e.g. 8, 24). Cannot be manually changed." align="right" />
            </label>
            <NumberInput value={fps} onChange={() => { }} disabled={true} />
          </div>
        </div>

        {/* Output Format */}
        <div className="space-y-2">
          <label className="label">Output Format</label>
          <select
            className="select w-full bg-primary border-border text-sm focus:border-brand-500"
            value={format}
            onChange={(e) => setFormat(e.target.value)}
            disabled={isLoading}
          >
            <option value="mp4">MP4 (H.264, Universal)</option>
            <option value="webm">WebM (VP9, Web)</option>
            <option value="mov">MOV (QuickTime)</option>
            <option value="mkv">MKV (Matroska)</option>
            <option value="avi">AVI (Legacy)</option>
            <option value="flv">FLV (Flash)</option>
            <option value="ts">TS (MPEG-TS)</option>
            <option value="gif">GIF (Animated, No Audio)</option>
          </select>
        </div>

        <ErrorAlert error={error} onDismiss={() => setError(null)} />

        <ValidationTooltip error={(!prompt.trim() && !inputImagePath) ? "Please enter a prompt or image" : null} className="w-full mt-auto pt-4">
          <button
            className="w-full bg-gradient-to-r from-brand-600 to-pink-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-primary font-bold py-3 rounded-lg shadow-lg shadow-brand-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all"
            onClick={handleGenerate}
            disabled={isLoading || pendingGeneration || (!prompt.trim() && !inputImagePath)}
          >
            {isLoading ? (<><Loader2 className="animate-spin" size={18} /> Generating...</>) : (<><Film size={18} /> Generate Video</>)}
          </button>
        </ValidationTooltip>
      </div>

      {/* Main Preview */}
      <div ref={resultRef} className="flex-1 p-6 flex items-center justify-center bg-primary/30 min-h-[500px] lg:min-h-0 scroll-mt-4">
        {result ? (
          <div className="flex flex-col items-center justify-center max-w-full h-full gap-4">
            <div
              className="relative group rounded-lg overflow-hidden border border-brand-500/30 shadow-2xl max-h-[85vh] cursor-pointer"
              onClick={() => setIsPreviewOpen(true)}
            >
              <video src={`${API_BASE_URL()}/api/files/${result}`} controls className="max-h-[85vh] object-contain" />
              <div className="absolute top-2 left-2 bg-brand-600 px-2 py-1 rounded text-[10px] sm:text-xs text-primary shadow-lg flex flex-col items-start leading-none gap-0.5">
                <span className="font-bold uppercase tracking-wider">Result</span>
                {genDuration && <span className="opacity-80 font-medium">in {formatDuration(genDuration * 1000)}</span>}
              </div>
              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity pointer-events-none">
                <span className="bg-white/90 text-black px-4 py-2 rounded-lg font-bold text-sm">Open Full Preview</span>
              </div>
            </div>

            <div className="flex gap-2">
              <button className="btn-secondary text-sm" onClick={() => setIsPreviewOpen(true)}>Full Screen</button>
              <a href={`${API_BASE_URL()}/api/files/${result}?download=true`} className="btn-secondary text-sm">Download</a>
            </div>
          </div>
        ) : (
          <div className="text-center text-tertiary">
            <Film size={48} className="mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-medium mb-2">Ready to Generate</h3>
            <p className="text-secondary max-w-sm">
              Enter a prompt <span className="lg:hidden">controls above</span><span className="hidden lg:inline">sidebar</span> to start creating videos.
            </p>
          </div>
        )}
      </div>

      <ResourceWarningModal
        isOpen={showWarning}
        warning={warningDetails?.message || "High resource usage warning"}
        type={warningDetails?.critical ? 'critical' : 'warning'}
        details={warningDetails?.details}
        onConfirm={() => executeGeneration(true)}
        onCancel={() => {
          setShowWarning(false);
          setPendingGeneration(false);
        }}
      />

      {currentJobId && (
        <JobProgressModal
          jobId={currentJobId}
          onClose={handleCloseModal}
          onViewResult={handleViewResult}
        />
      )}

      {result && (
        <PreviewModal
          isOpen={isPreviewOpen}
          onClose={() => setIsPreviewOpen(false)}
          filePath={result}
          fileName={result.split('/').pop() || 'video.mp4'}
        />
      )}
    </div>
  );
}
