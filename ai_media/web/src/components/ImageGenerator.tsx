import { useState, useEffect } from 'react';
import { useAppStore } from '../store';
import { generateImage, fetchModels } from '../hooks/useApi';
import { API_BASE_URL } from '../config';
import { Sparkles, Loader2, AlertTriangle } from 'lucide-react';
import { Tooltip } from './common/Tooltip';
import { ResolutionSelector } from './common/ResolutionSelector';
import { ValidationTooltip } from './common/ValidationTooltip';
import { RandomPrompt } from './common/RandomPrompt';
import { JobProgressModal } from './common/JobProgressModal';
import { ResourceWarningModal } from './common/ResourceWarningModal';
import { PreviewModal } from './PreviewModal';
import { ErrorAlert } from './common/ErrorAlert';
import { ModelHelpLink } from './common/ModelHelpLink';
import { formatDuration } from '../utils/formatTime';

interface ModelInfo {
  name: string;
  model_id: string;
  vram_required: number | null;
  ram_required: number | null;
  max_resolution: [number, number] | null;
}

// Display names and info matching CLI exactly
const MODEL_DISPLAY_INFO: Record<string, { label: string; vram: string; note?: string }> = {
  'sd3.5-turbo': { label: 'SD 3.5 Turbo (Default, Fast 4 Steps, 🔒 Gated)', vram: '~19GB' },
  'sdxl': { label: 'SDXL Turbo (Fast, no login)', vram: '~8GB' },
  'sd-1.5': { label: 'SD 1.5 (Lightweight, Negative Prompt)', vram: '~4GB' },
  'sd3.5-medium': { label: 'SD 3.5 Medium (High Quality, Negative Prompt, 🔒 Gated)', vram: '~10GB' },
  'sd3.5-large': { label: 'SD 3.5 Large (Best Quality, Negative Prompt, 🔒 Gated)', vram: '~19GB' },

  'qwen-image-auto': { label: 'Qwen 2.5 Image (High Quality, Negative Prompt)', vram: '~20-40GB' },
  'qwen-image-lightning': { label: 'Qwen 2.5 Image (Lightning, Fast)', vram: '~40GB' },
  'qwen-image-4bit': { label: 'Qwen 2.5 Image (4-bit Lite, Negative Prompt, CUDA only)', vram: '~20GB' },
  'flux': { label: 'Flux Schnell (High Quality, Slow on Mac, 🔒 Gated)', vram: '~12GB' },
  'flux-dev': { label: 'Flux Dev (Professional, Very Slow on Mac, 🔒 Gated)', vram: '~16GB' },
  'flux2': { label: 'FLUX.2 (4-bit quantized, CUDA only, 🔒 Gated)', vram: '~12GB' },
  'flux2-full': { label: 'FLUX.2 Full (SOTA 2025, ⚠️ 128GB+ RAM!, 🔒 Gated)', vram: '~65GB' },
};

const MODEL_ORDER = [
  'sd3.5-turbo', 'sdxl', 'sd-1.5', 'sd3.5-medium', 'sd3.5-large',
  'qwen-image-auto', 'qwen-image-lightning', 'qwen-image-4bit', 'flux', 'flux-dev', 'flux2', 'flux2-full'
];

export function ImageGenerator() {
  const { addJob, systemInfo } = useAppStore();
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState('sd3.5-turbo'); // Default matching CLI
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [steps, setSteps] = useState(4); // SD 3.5 Turbo uses 4 steps by default
  const [guidanceScale, setGuidanceScale] = useState(0); // SD 3.5 Turbo uses 0
  const [negativePrompt, setNegativePrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);

  const [duration, setDuration] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [showWarning, setShowWarning] = useState(false);
  const [pendingGeneration, setPendingGeneration] = useState<boolean>(false);

  // High resource text for the model that triggered the warning
  const [warningDetails, setWarningDetails] = useState<{
    message: string;
    details?: any;
    critical?: boolean;
  } | null>(null);

  // Fetch models on mount
  useEffect(() => {
    fetchModels()
      .then((data) => {
        if (data.image) {
          setAvailableModels(data.image);
        }
      })
      .catch((err) => console.error('Failed to fetch models:', err));
  }, []);

  // Update defaults when model changes
  useEffect(() => {
    // Set model-specific defaults
    if (model.includes('sd-1.5')) {
      setWidth(512);
      setHeight(512);
    } else {
      // CLI Default is 720p (1280x720) for all other models
      setWidth(1280);
      setHeight(720);
    }

    // Steps defaults
    if (model.includes('turbo') || model.includes('flux')) {
      setSteps(4);
      setGuidanceScale(0);
    } else if (model === 'sd-1.5') {
      setSteps(30);
      setGuidanceScale(7.5);
    } else {
      setSteps(30);
      setGuidanceScale(7.5);
    }
  }, [model]);



  // Actual execution logic, moved from handleGenerate
  const executeGeneration = async (force: boolean = false) => {
    setIsLoading(true);
    setResult(null);
    setDuration(null);
    setError(null);
    setShowWarning(false);
    setPendingGeneration(false);

    try {
      // Resolve model alias for Qwen
      let selectedModel = model;
      if (model === 'qwen-image-auto') {
        const has2512 = availableModels.some(m => m.name === 'qwen-image-2512');
        const hasCUDA = availableModels.some(m => m.name === 'qwen-image');
        const sys = useAppStore.getState().systemInfo;

        if (sys?.mps_available && has2512) {
          selectedModel = 'qwen-image-2512';
        } else if (hasCUDA) {
          selectedModel = 'qwen-image';
        } else if (has2512) {
          selectedModel = 'qwen-image-2512';
        }
        // Fallback if neither found is unlikely due to availability check, but keep original if so
      } else if (model === 'qwen-image-4bit' || model === 'qwen-image-lightning') {
        selectedModel = model; // Explicit selection
      }

      const response = await generateImage({
        prompt,
        model: selectedModel,
        width,
        height,
        steps,
        guidance_scale: guidanceScale,
        negative_prompt: negativePrompt,
        force,
      });

      setCurrentJobId(response.job_id);

      addJob({
        job_id: response.job_id,
        type: 'image',
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
    if (!prompt.trim()) return;

    // Check for High Resource models that block the CLI
    // Check for High Resource models that block the CLI
    // "qwen-image-auto" (resolves to heavy models), "flux2-full", "flux-dev" often trigger high-RAM/VRAM warnings
    const highResourceModels = ['qwen-image-auto', 'qwen-image-4bit', 'qwen-image-lightning', 'flux2-full'];

    // Qwen2512 is notably heavy (40GB) and almost always warns on consumer hardware
    // Flux2 Full is 128GB+

    if (highResourceModels.includes(model)) {
      let warningMsg = "";
      let critical = false;

      const sys = useAppStore.getState().systemInfo;
      const resources = useAppStore.getState().resources;
      const totalRam = sys?.ram_total_gb || 0;
      const usedRam = resources?.global.ram_used_gb || 0;
      const freeRam = resources ? (totalRam - usedRam) : (totalRam * 0.5); // Default to 50% free if stats not yet polled

      const estimatedRam = model === 'flux2-full' ? 128 : 40;

      // Logic: Only show the blocking modal if:
      // 1. The model literally won't fit in total RAM (Critical)
      // 2. The model uses > 85% of total RAM (High risk of swap)
      // 3. Current available RAM is less than the model requirement (High risk of immediate swap)

      const needsWarning = estimatedRam > totalRam || estimatedRam > (totalRam * 0.85) || estimatedRam > freeRam;

      if (needsWarning) {
        if (model === 'flux2-full') {
          warningMsg = "This model (FLUX.2 Full) requires over 128GB of RAM. It will almost certainly crash or freeze average consumer hardware (laptops/desktops) unless you have a workstation class machine. Proceed only if sure.";
          critical = estimatedRam > totalRam;
        } else if (model.includes('qwen')) {
          warningMsg = "Qwen-Image models are very resource intensive (~20-40GB RAM). This may cause system slowdowns or swapping on your current setup. The process normally asks for confirmation in CLI - clicking Proceed here will auto-confirm it.";
          critical = estimatedRam > totalRam;
        }

        setWarningDetails({
          message: warningMsg,
          critical: critical,
          details: {
            target_resolution: `${width}x${height}`,
            megapixels: Math.round((width * height) / 10000) / 100,
            estimated_ram_gb: estimatedRam,
            available_ram_gb: Math.round(freeRam * 10) / 10
          }
        });

        setShowWarning(true);
        setPendingGeneration(true);
        return;
      }
    }

    // Standard path (or high-resource model that fits comfortably)
    executeGeneration(false);
  };

  // Effect to watch the current job for completion/failure using store data
  // This replaces the poll interval
  useEffect(() => {
    if (!currentJobId) return;

    // Track if we've ever seen this job in the store (to detect removal vs not-yet-added)
    let hasSeenJob = false;

    // Use a subscription to the store to react to updates for this specific job
    const unsubscribe = useAppStore.subscribe((state) => {
      const updatedJob = state.jobs.find(j => j.job_id === currentJobId);
      if (updatedJob) {
        hasSeenJob = true; // Mark that we've seen the job
        
        if (updatedJob.status === 'complete') {
          setResult(updatedJob.result_path);

          // Calculate duration
          if (updatedJob.generation_started_at && updatedJob.updated_at) {
            const start = new Date(updatedJob.generation_started_at).getTime();
            const end = new Date(updatedJob.updated_at).getTime();
            const seconds = Math.round((end - start) / 1000);
            setDuration(seconds > 0 ? seconds : 1);
          } else if (updatedJob.created_at && updatedJob.updated_at) {
            const start = new Date(updatedJob.created_at).getTime();
            const end = new Date(updatedJob.updated_at).getTime();
            setDuration(Math.round((end - start) / 1000));
          }

          setIsLoading(false);
        } else if (updatedJob.status === 'failed') {
          setIsLoading(false);
          setError(updatedJob.error || updatedJob.message || "Generation failed");
        } else if (updatedJob.status === 'cancelled') {
          setIsLoading(false);
          setError("Job cancelled. The model may continue running briefly until it reaches a checkpoint.");
          setTimeout(() => setError(null), 8000);
        }
      } else if (hasSeenJob) {
        // Job was in store but now removed (e.g., cancelled - server cleaned up)
        setIsLoading(false);
        setCurrentJobId(null);
      }
      // If !updatedJob && !hasSeenJob, the job simply hasn't appeared in the store yet - do nothing
    });

    return () => unsubscribe();
  }, [currentJobId]);

  // Sort models in CLI order and filter based on system capabilities
  const sortedModels = MODEL_ORDER.filter(name => {
    // Hide CUDA-only models on non-CUDA systems (like Mac/MPS)
    const isNoCuda = systemInfo && !systemInfo.cuda_available;
    if (isNoCuda) {
      if (name === 'qwen-image-4bit') return false;
      if (name === 'flux2') return false;
    }

    if (name === 'qwen-image-auto' || name === 'qwen-image-4bit' || name === 'qwen-image-lightning') {
      return availableModels.some(m => m.name === 'qwen-image' || m.name === 'qwen-image-2512');
    }
    return availableModels.some(m => m.name === name);
  });

  // Check if current model supports Negative Prompt (Lightning models don't support CFG)
  const supportsNegativePrompt = !model.includes('turbo') && !model.includes('flux') && !model.includes('lightning') && model !== 'sdxl';

  const handleCloseModal = () => {
    setCurrentJobId(null);
    setIsLoading(false);
  };

  return (
    <div className="flex flex-col lg:flex-row h-full bg-primary text-primary">
      {/* Parameters Sidebar */}
      <div className="w-full lg:w-[500px] border-b lg:border-b-0 lg:border-r border-border p-4 lg:py-6 lg:pr-[27px] lg:pl-1 flex flex-col gap-6 overflow-y-auto shrink-0 h-auto lg:h-full">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
            <Sparkles className="text-brand-400" /> Image Gen
          </h2>
          <p className="text-xs text-tertiary">Generate images from text descriptions</p>
        </div>

        {/* Prompt */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-secondary">Prompt</label>
            <RandomPrompt type="image" onPromptSelect={setPrompt} />
          </div>
          <textarea
            className="w-full bg-primary border border-border rounded-lg p-3 text-sm focus:outline-none focus:border-brand-500 resize-y min-h-[100px]"
            placeholder="A majestic mountain landscape at sunset with dramatic clouds..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        {/* Negative Prompt */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className={`text-sm font-medium ${supportsNegativePrompt ? 'text-secondary' : 'text-tertiary'} flex items-center gap-1`}>
              Negative Prompt (Optional)
              <Tooltip content="List items to exclude (e.g., 'blur, text'). Do NOT use 'no' or 'without'. Note: For Lightning/Turbo models, using this will force standard speed (~2x slower)." />
            </label>
          </div>
          <input
            type="text"
            className={`w-full bg-primary border border-border rounded-lg p-3 text-sm focus:outline-none focus:border-brand-500 ${!supportsNegativePrompt ? 'opacity-50 cursor-not-allowed text-tertiary' : ''}`}
            placeholder={supportsNegativePrompt ? "e.g. blur, text, watermark (NOT 'no text')" : "Not supported by this model"}
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            disabled={!supportsNegativePrompt}
          />
        </div>

        {/* Model Selector */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-secondary flex items-center">
            Model
            <ModelHelpLink section="image" />
          </label>
          <select
            className="select w-full bg-primary border-border text-sm focus:border-brand-500"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            {sortedModels.map((name) => {
              const info = MODEL_DISPLAY_INFO[name];
              return (
                <option key={name} value={name}>
                  {info ? `${info.label} ${info.vram}` : name}
                </option>
              );
            })}
          </select>

          {/* Warnings */}
          {(model.includes('sd3.5') || model === 'stable-audio') && (
            <div className="flex items-center gap-2 text-amber-600 dark:text-amber-400 text-xs mt-1">
              <AlertTriangle size={12} />
              <span>Requires HF Login</span>
            </div>
          )}
          {model === 'flux2-full' && (
            <div className="flex items-center gap-2 text-red-600 dark:text-red-400 text-xs mt-1">
              <AlertTriangle size={12} />
              <span>Requires 128GB+ RAM</span>
            </div>
          )}
        </div>


        {/* Resolution */}
        <div className="space-y-2">
          <ResolutionSelector
            width={width}
            height={height}
            onChange={(w, h) => { setWidth(w); setHeight(h); }}
          />
        </div>

        {/* Advanced Settings */}
        <div className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-secondary flex items-center gap-1">
                Steps: <span className="text-brand-400 font-bold">{steps}</span>
                <Tooltip content="Number of refinement passes. Turbo models need 4. Standard models work best at 20-30. Higher is slower but more detailed." align="left" />
              </label>
            </div>
            <input
              type="range"
              min="1"
              max="100"
              step="1"
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value))}
              className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-brand-500"
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-secondary flex items-center gap-1">
                Text Guidance (CFG): <span className="text-brand-400 font-bold">{guidanceScale}</span>
                <Tooltip content="How closely to follow your prompt. 0 is distilled (Turbo/Flux), 5-8 is standard. High values (>12) on Mac can sometimes cause black images." align="left" />
              </label>
            </div>
            <input
              type="range"
              min="0"
              max="20"
              step="0.5"
              value={guidanceScale}
              onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
              className="w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-brand-500"
            />
          </div>


        </div>

        <ErrorAlert error={error} onDismiss={() => setError(null)} />

        {/* Generate Action */}
        <ValidationTooltip error={!prompt.trim() ? "Please enter a prompt" : null} className="w-full mt-auto pt-4">
          <button
            className="w-full bg-gradient-to-r from-brand-600 to-indigo-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-primary font-bold py-3 rounded-lg shadow-lg shadow-brand-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all"
            onClick={handleGenerate}
            disabled={isLoading || !prompt.trim() || pendingGeneration}
          >
            {isLoading ? (
              <><Loader2 className="animate-spin" size={18} /> Generating...</>
            ) : (
              <><Sparkles size={18} /> Generate Image</>
            )}
          </button>
        </ValidationTooltip>

      </div>

      {/* Main Preview Area */}
      <div className="flex-1 p-6 flex items-center justify-center bg-primary/30">
        {result ? (
          <div className="flex flex-col items-center justify-center max-w-full h-full gap-4">
            <div
              className="relative group rounded-lg overflow-hidden border border-brand-500/30 shadow-2xl max-h-[85vh] cursor-pointer"
              onClick={() => setIsPreviewOpen(true)}
            >
              <img
                src={`${API_BASE_URL()}/api/files/${result}`}
                alt="Generated Image"
                className="max-h-[85vh] object-contain"
              />
              <div className="absolute top-2 left-2 bg-brand-600 px-2 py-1 rounded text-[10px] sm:text-xs text-primary shadow-lg flex flex-col items-start leading-none gap-0.5">
                <span className="font-bold uppercase tracking-wider">Result</span>
                {duration && <span className="opacity-80 font-medium">in {formatDuration(duration * 1000)}</span>}
              </div>
              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
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
            <Sparkles size={48} className="mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-medium mb-2">Ready to Imagine</h3>
            <p className="text-secondary max-w-sm">
              Enter a prompt in the sidebar and click Generate to create an image.
            </p>
          </div>
        )}
      </div>

      {/* Modals */}
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
        <JobProgressModal jobId={currentJobId} onClose={handleCloseModal} />
      )}

      {result && (
        <PreviewModal
          isOpen={isPreviewOpen}
          onClose={() => setIsPreviewOpen(false)}
          filePath={result}
          fileName={result.split('/').pop() || 'image.png'}
        />
      )}
    </div>
  );
}
