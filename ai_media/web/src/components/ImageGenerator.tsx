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
  'qwen-image': { label: 'Qwen-Image (CUDA, 4-bit)', vram: '~20GB' },
  'qwen-image-mps': { label: 'Qwen-Image (Best Text, Mac Full)', vram: '~40GB' },
  'flux': { label: 'Flux Schnell (High Quality, Slow on Mac)', vram: '~12GB' },
  'flux-dev': { label: 'Flux Dev (Professional, Very Slow on Mac)', vram: '~16GB' },
  'flux2': { label: 'FLUX.2 (4-bit quantized)', vram: '~12GB' },
  'flux2-full': { label: 'FLUX.2 Full (SOTA 2025, ⚠️ 128GB+ RAM!)', vram: '~65GB' },
};

const MODEL_ORDER = [
  'sd3.5-turbo', 'sdxl', 'sd-1.5', 'sd3.5-medium', 'sd3.5-large',
  'qwen-image', 'qwen-image-mps', 'flux', 'flux-dev', 'flux2', 'flux2-full'
];

export function ImageGenerator() {
  const { addJob } = useAppStore();
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState('sd3.5-turbo'); // Default matching CLI
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [steps, setSteps] = useState(4); // SD 3.5 Turbo uses 4 steps by default
  const [guidanceScale, setGuidanceScale] = useState(0); // SD 3.5 Turbo uses 0
  const [isLoading, setIsLoading] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

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

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    setIsLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await generateImage({ 
        prompt, 
        model, 
        width, 
        height, 
        steps,
        guidance_scale: guidanceScale,
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

      // No longer polling manually - monitoring via WebSocket in App root
    } catch (err) {
      console.error('Generation failed:', err);
      setIsLoading(false);
      setCurrentJobId(null);
      setError("Failed to start generation job");
    }
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

  // Sort models in CLI order
  const sortedModels = MODEL_ORDER.filter(name => 
    availableModels.some(m => m.name === name)
  );
  
  const handleCloseModal = () => {
    setCurrentJobId(null);
    setIsLoading(false);
  };

  return (
    <div className="w-full max-w-none px-4 mx-auto relative">
      <div className="card p-6 mb-8">
        <h1 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Sparkles className="text-primary-400" />
          Image Generation
        </h1>
      </div>

      <div className="card space-y-4">
        {/* Prompt */}
        <div>
          <div className="flex items-center justify-between mb-2">
             <label className="label mb-0">Prompt</label>
             <RandomPrompt type="image" onPromptSelect={setPrompt} />
          </div>
          <textarea
            className="input min-h-[150px] resize-y"
            placeholder="A majestic mountain landscape at sunset with dramatic clouds..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        {/* Model Selector - Full list matching CLI */}
        <div>
          <label className="label">Model</label>
          <select 
            className="select" 
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
          
          {/* Gated model warning */}
          {(model.includes('sd3.5') || model === 'stable-audio') && (
            <div className="mt-2 flex items-center gap-2 text-yellow-400 text-sm">
              <AlertTriangle size={16} />
              <span>Gated model - requires Hugging Face login</span>
               <button 
                onClick={() => useAppStore.getState().toggleHelp()} 
                className="underline hover:text-yellow-300 ml-1"
              >
                (See Help)
              </button>
            </div>
          )}
          {/* High RAM warning */}
          {model === 'flux2-full' && (
            <div className="mt-2 flex items-center gap-2 text-red-400 text-sm">
              <AlertTriangle size={16} />
              <span>This model requires 128GB+ RAM!</span>
            </div>
          )}
        </div>


        {/* Resolution Selector */}
        <ResolutionSelector 
           width={width} 
           height={height} 
           onChange={(w, h) => { setWidth(w); setHeight(h); }} 
        />

        {/* Steps/Guidance */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
           <div>
             <label className="label flex items-center">
                Steps
                <Tooltip content="Inference steps. SD 3.5 Turbo needs 4. Flux Schnell needs 4. Others usually 20-30." />
             </label>
             <NumberInput value={steps} onChange={setSteps} min={1} max={100} />
           </div>
           <div>
             <label className="label flex items-center">
                Guidance
                <Tooltip content="CFG Scale. SD 3.5 Turbo/Flux Schnell use 0 (distilled). Others use 5-7. Higher = follows prompt more strictly." />
             </label>
             <NumberInput value={guidanceScale} onChange={setGuidanceScale} min={0} max={20} step={0.5} allowFloat={true} />
           </div>
        </div>
        

        <ValidationTooltip error={!prompt.trim() ? "Please enter a prompt to generate an image" : null} className="w-full">
          <button 
            className="btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed" 
            onClick={handleGenerate} 
            disabled={isLoading || !prompt.trim()}
          >
            {isLoading ? (<><Loader2 className="animate-spin" size={18} />Generating...</>) : (<><Sparkles size={18} />Generate Image</>)}
          </button>
        </ValidationTooltip>
      </div>

      <ErrorAlert error={error} onDismiss={() => setError(null)} />
      
      {/* Job Progress Modal */}
      {currentJobId && (
        <JobProgressModal jobId={currentJobId} onClose={handleCloseModal} />
      )}

      {/* Result Preview */}
      {result && (
        <div className="mt-6 card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-primary">Result</h2>
            <div className="flex gap-2">
               <button 
                  className="btn-primary text-sm"
                  onClick={() => setIsPreviewOpen(true)}
                >
                  Preview
                </button>
                <a href={`http://localhost:8000/api/files/${result}`} target="_blank" rel="noreferrer" className="btn-secondary text-sm">Download</a>
            </div>
          </div>
          <div className="cursor-pointer group relative rounded-lg overflow-hidden border border-border" onClick={() => setIsPreviewOpen(true)}>
            <img
              src={`http://localhost:8000/api/files/${result}`}
              alt="Generated image"
              className="max-w-full rounded-lg"
            />
            <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
               <span className="bg-white/90 text-black px-4 py-2 rounded-lg font-bold text-sm">Open Full Preview</span>
            </div>
          </div>
        </div>
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
