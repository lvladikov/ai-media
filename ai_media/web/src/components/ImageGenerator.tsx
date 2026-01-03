import { useState, useEffect } from 'react';
import { useAppStore } from '../store';
import { generateImage, fetchModels } from '../hooks/useApi';
import { Sparkles, Loader2, AlertTriangle } from 'lucide-react';
import { NumberInput } from './common/NumberInput';
import { Tooltip } from './common/Tooltip';
import { ResolutionSelector } from './common/ResolutionSelector';
import { ValidationTooltip } from './common/ValidationTooltip';
import { RandomPrompt } from './common/RandomPrompt';
import { JobProgressModal } from './common/JobProgressModal';
import { ResourceWarningModal } from './common/ResourceWarningModal';
import { PreviewModal } from './PreviewModal';
import { ErrorAlert } from './common/ErrorAlert';

// ... (keep interfaces)
// ...

// Note: I am not including the entire file here, just fixing the import and then targeting the specific body part separately?
// No, replace_file_content replaces a contiguous block. I cannot edit top and bottom easily in one go unless I include everything between.
// I will use multi_replace for this, or two replace calls.
// Let's use two replace calls. First imports.

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
  'sd-1.5': { label: 'SD 1.5 (Lightweight)', vram: '~4GB' },
  'sd3.5-medium': { label: 'SD 3.5 Medium (High Quality, 🔒 Gated)', vram: '~10GB' },
  'sd3.5-large': { label: 'SD 3.5 Large (Best Quality, 🔒 Gated)', vram: '~19GB' },
  
  'qwen-image-auto': { label: 'Qwen 2.5 Image (High Quality)', vram: '~20-40GB' },
  'qwen-image-lightning': { label: 'Qwen 2.5 Image (Lightning, Fast)', vram: '~40GB' },
  'qwen-image-4bit': { label: 'Qwen 2.5 Image (4-bit Lite, CUDA only)', vram: '~20GB' },
  'flux': { label: 'Flux Schnell (High Quality, Slow on Mac)', vram: '~12GB' },
  'flux-dev': { label: 'Flux Dev (Professional, Very Slow on Mac)', vram: '~16GB' },
  'flux2': { label: 'FLUX.2 (4-bit quantized, CUDA only)', vram: '~12GB' },
  'flux2-full': { label: 'FLUX.2 Full (SOTA 2025, ⚠️ 128GB+ RAM!)', vram: '~65GB' },
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

  // ... (keep useEffects for fetchModels and defaults)

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
      setGuidanceScale(0); // Turbo/Flux models often use 0 guidance
    } else if (model === 'sd-1.5') {
      setSteps(30);
      setGuidanceScale(7.5);
    } else {
      // Default for other models
      setSteps(30);
      setGuidanceScale(7.5);
    }
  }, [model]);

  // Actual execution logic, moved from handleGenerate
  const executeGeneration = async (force: boolean = false) => {
    setIsLoading(true);
    setResult(null);
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
        negative_prompt: negativePrompt, // Pass negative prompt
        force, // Pass force flag to API
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
        
        if (model === 'flux2-full') {
            warningMsg = "This model (FLUX.2 Full) requires over 128GB of RAM. It will almost certainly crash or freeze average consumer hardware (laptops/desktops) unless you have a workstation class machine. Proceed only if sure.";
            critical = true;
        } else if (model.includes('qwen')) {
            warningMsg = "Qwen-Image models are very resource intensive (~20-40GB RAM). This may cause system slowdowns or swapping. The process normally asks for confirmation in CLI - clicking Proceed here will auto-confirm it.";
            critical = false;
        }

        setWarningDetails({
            message: warningMsg,
            critical: critical,
            details: {
                target_resolution: `${width}x${height}`,
                megapixels: Math.round((width * height) / 10000) / 100,
                // Rough estimates
                estimated_ram_gb: model === 'flux2-full' ? 128 : 40, 
                available_ram_gb: useAppStore.getState().systemInfo?.ram_total_gb || 0
            }
        });
        
        setShowWarning(true);
        setPendingGeneration(true);
        return;
    }

    // Standard path
    executeGeneration(false);
  };
  
  // Effect to watch the current job for completion/failure using store data
  // This replaces the poll interval
  useEffect(() => {
    if (!currentJobId) return;
    
    // Find the job in the global store
    const job = useAppStore.getState().jobs.find(j => j.job_id === currentJobId);
    if (!job) return;

    // Use a subscription to the store to react to updates for this specific job
    const unsubscribe = useAppStore.subscribe((state) => {
        const updatedJob = state.jobs.find(j => j.job_id === currentJobId);
        if (updatedJob) {
            if (updatedJob.status === 'complete') {
                setResult(updatedJob.result_path);
                setIsLoading(false);
            } else if (updatedJob.status === 'failed') {
                setIsLoading(false);
                setError(updatedJob.error || updatedJob.message || "Generation failed");
            } else if (updatedJob.status === 'cancelled') {
                setIsLoading(false);
                setError("Job cancelled. The model may continue running briefly until it reaches a checkpoint.");
                setTimeout(() => setError(null), 8000);
            }
        } else {
            // Job not found in store (removed on cancellation)
            setIsLoading(false);
            setCurrentJobId(null);
        }
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
    <div className="flex flex-col lg:flex-row h-full bg-slate-900 text-slate-200">
      {/* Parameters Sidebar */}
      <div className="w-full lg:w-[500px] border-b lg:border-b-0 lg:border-r border-slate-800 p-4 lg:py-6 lg:pr-[27px] lg:pl-0 flex flex-col gap-6 overflow-y-auto shrink-0 h-auto lg:h-full">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
            <Sparkles className="text-brand-400" /> Image Gen
          </h2>
          <p className="text-xs text-slate-500">Generate images from text descriptions</p>
        </div>

        {/* Prompt */}
        <div className="space-y-2">
           <div className="flex items-center justify-between">
             <label className="text-sm font-medium text-slate-400">Prompt</label>
             <RandomPrompt type="image" onPromptSelect={setPrompt} />
           </div>
           <textarea
             className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm focus:outline-none focus:border-brand-500 resize-y min-h-[100px]"
             placeholder="A majestic mountain landscape at sunset with dramatic clouds..."
             value={prompt}
             onChange={(e) => setPrompt(e.target.value)}
           />
        </div>
        
        {/* Negative Prompt */}
        <div className="space-y-2">
           <div className="flex items-center justify-between">
             <label className={`text-sm font-medium ${supportsNegativePrompt ? 'text-slate-400' : 'text-slate-600'} flex items-center gap-1`}>
                Negative Prompt (Optional)
                <Tooltip content="List items to exclude (e.g., 'blur, text'). Do NOT use 'no' or 'without'. Note: For Lightning/Turbo models, using this will force standard speed (~2x slower)." />
             </label>
           </div>
           <input
             type="text"
             className={`w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm focus:outline-none focus:border-brand-500 ${!supportsNegativePrompt ? 'opacity-50 cursor-not-allowed text-slate-500' : ''}`}
             placeholder={supportsNegativePrompt ? "e.g. blur, text, watermark (NOT 'no text')" : "Not supported by this model"}
             value={negativePrompt}
             onChange={(e) => setNegativePrompt(e.target.value)}
             disabled={!supportsNegativePrompt}
           />
        </div>

        {/* Model Selector */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-400">Model</label>
          <select 
            className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-sm focus:outline-none focus:border-brand-500"
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
            <div className="flex items-center gap-2 text-amber-400 text-xs mt-1">
              <AlertTriangle size={12} />
              <span>Requires HF Login</span>
            </div>
          )}
          {model === 'flux2-full' && (
            <div className="flex items-center gap-2 text-red-400 text-xs mt-1">
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
        <div className="grid grid-cols-2 gap-4">
           <div className="space-y-1">
             <label className="text-xs font-medium text-slate-400 flex items-center gap-1">
                Steps
                <Tooltip content="Inference steps. SD 3.5 Turbo needs 4. Flux Schnell needs 4. Others usually 20-30." />
             </label>
             <NumberInput value={steps} onChange={setSteps} min={1} max={100} />
           </div>
           <div className="space-y-1">
             <label className="text-xs font-medium text-slate-400 flex items-center gap-1">
                Guidance
                <Tooltip content="CFG Scale. SD 3.5 Turbo/Flux Schnell use 0 (distilled). Others use 5-7." />
             </label>
             <NumberInput value={guidanceScale} onChange={setGuidanceScale} min={0} max={20} step={0.5} allowFloat={true} />
           </div>
        </div>
        
        <ErrorAlert error={error} onDismiss={() => setError(null)} />

        {/* Generate Action */}
        <ValidationTooltip error={!prompt.trim() ? "Please enter a prompt" : null} className="w-full mt-auto pt-4">
          <button 
            className="w-full bg-gradient-to-r from-brand-600 to-indigo-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-white font-bold py-3 rounded-lg shadow-lg shadow-brand-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all"
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
      <div className="flex-1 p-6 flex items-center justify-center bg-slate-950/30">
        {result ? (
           <div className="flex flex-col items-center justify-center max-w-full h-full gap-4">
               <div 
                 className="relative group rounded-lg overflow-hidden border border-brand-500/30 shadow-2xl max-h-[85vh] cursor-pointer" 
                 onClick={() => setIsPreviewOpen(true)}
               >
                 <img 
                   src={`http://localhost:8000/api/files/${result}`} 
                   alt="Generated Image" 
                   className="max-h-[85vh] object-contain" 
                 />
                 <div className="absolute top-2 left-2 bg-brand-600 px-2 py-1 rounded text-xs text-white shadow-lg">Result</div>
                 <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                    <span className="bg-white/90 text-black px-4 py-2 rounded-lg font-bold text-sm">Open Full Preview</span>
                 </div>
               </div>
               
               <div className="flex gap-2">
                 <button className="btn-secondary text-sm" onClick={() => setIsPreviewOpen(true)}>Full Screen</button>
                 <a href={`http://localhost:8000/api/files/${result}`} target="_blank" rel="noreferrer" className="btn-secondary text-sm">Download</a>
               </div>
           </div>
        ) : (
          <div className="text-center text-slate-500">
            <Sparkles size={48} className="mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-medium mb-2">Ready to Imagine</h3>
            <p className="text-slate-400 max-w-sm">
                Upload an image from the <span className="lg:hidden">controls above</span><span className="hidden lg:inline">sidebar</span> to start editing with AI instructions.
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
