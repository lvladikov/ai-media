import { useState, useEffect } from 'react';
import { useAppStore } from '../store';
import { generateVideo, fetchModels } from '../hooks/useApi';
import { Film, Loader2, AlertTriangle, Info } from 'lucide-react';
import { NumberInput } from './common/NumberInput';
import { Tooltip } from './common/Tooltip';
import { ResolutionSelector } from './common/ResolutionSelector';
import { ValidationTooltip } from './common/ValidationTooltip';
import { RandomPrompt } from './common/RandomPrompt';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';

interface ModelInfo {
  name: string;
  model_id: string;
  vram_required: number | null;
  ram_required: number | null;
  max_resolution: [number, number] | null;
}

// Display info matching CLI
const MODEL_DISPLAY_INFO: Record<string, { label: string; vram: string }> = {
  'zeroscope': { label: 'Zeroscope (Fast, 576x320)', vram: '~4GB' },
  'zeroscope-xl': { label: 'Zeroscope XL (Higher Res)', vram: '~8GB' },
  'ms-1.7b': { label: 'ModelScope 1.7B (Watermark Issues)', vram: '~12GB' },
  'cogvideox': { label: 'CogVideoX-5b (High Qual, 38GB VRAM!)', vram: '~38GB' },
  'wan-2.2': { label: 'Wan 2.2 (14B, 🔒 Gated)', vram: '~24GB' },
  'ltx-video': { label: 'LTX-Video (Fast, High Res)', vram: '~16GB' },
  'mochi-1': { label: 'Mochi-1 (Physics SOTA)', vram: '~20GB' },
  'hunyuan': { label: 'HunyuanVideo (13B, Cinematic)', vram: '~24GB' },
  'svd': { label: 'Stable Video Diffusion (Image-to-Video)', vram: '~16GB' },
};

const MODEL_ORDER = [
  'zeroscope', 'zeroscope-xl', 'ms-1.7b', 'cogvideox', 'wan-2.2', 
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
  const [isLoading, setIsLoading] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  // Fetch models on mount
  useEffect(() => {
    fetchModels()
      .then((data) => {
        if (data.video) {
          setAvailableModels(data.video);
        }
      })
      .catch((err) => console.error('Failed to fetch models:', err));
  }, []);

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
    } else if (model === 'wan-2.2') {
      setFps(15);
    } else if (model === 'cogvideox') {
      setFps(8);
    } else {
      setFps(24);
    }
  }, [model]);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setIsLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await generateVideo({ 
        prompt, 
        model, 
        width, 
        height, 
        duration, // Note: backend uses frames/fps usually but duration is safer abstraction
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

  // Watch job status via store subscription
  useEffect(() => {
      if (!currentJobId) return;

      const unsubscribe = useAppStore.subscribe((state) => {
          const updatedJob = state.jobs.find(j => j.job_id === currentJobId);
          if (updatedJob) {
              if (updatedJob.status === 'complete') {
                  setResult(updatedJob.result_path);
                  setIsLoading(false);
              } else if (updatedJob.status === 'failed') {
                  setIsLoading(false);
              }
          }
      });
      return () => unsubscribe();
  }, [currentJobId]);

  const handleCloseModal = () => {
      setCurrentJobId(null);
  };

  // Sort models
  const sortedModels = MODEL_ORDER.filter(name => 
    availableModels.some(m => m.name === name)
  );

  return (
    <div className="max-w-4xl relative">
      <h1 className="text-2xl font-bold mb-6 flex items-center gap-2">
        <Film className="text-primary-400" />
        Video Generation
      </h1>

      <div className="card space-y-4">
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="label mb-0">Prompt</label>
            <RandomPrompt type="video" onPromptSelect={setPrompt} />
          </div>
          <textarea
            className="input min-h-[100px] resize-y"
            placeholder="A serene forest with sunlight filtering through the trees..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        <div>
          <label className="label">Model</label>
          <select className="select" value={model} onChange={(e) => setModel(e.target.value)}>
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
          {model === 'cogvideox' && (
            <div className="mt-2 flex items-center gap-2 text-red-400 text-sm">
              <AlertTriangle size={16} />
              <span>Extremely high VRAM usage (38GB+) - likely to fail on consumer GPUs</span>
            </div>
          )}
          {model === 'wan-2.2' && (
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
          {model === 'svd' && (
            <div className="mt-2 flex items-center gap-2 text-blue-400 text-sm">
              <Info size={16} />
              <span>SVD requires an input image (not supported in text-to-video mode yet)</span>
            </div>
          )}
        </div>


        {/* Resolution */}
        <ResolutionSelector 
            width={width} 
            height={height} 
            onChange={(w, h) => { setWidth(w); setHeight(h); }} 
        />

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label flex items-center">
               Duration (s)
               <Tooltip content="Length of video in seconds. Default 2s. Longer videos take significantly more VRAM/time." />
            </label>
            <NumberInput value={duration} onChange={setDuration} min={1} max={10} step={0.1} allowFloat={true} />
          </div>
             <div>
            <label className="label flex items-center">
               FPS
               <Tooltip content="Frames Per Second. Determined by model (e.g. 8, 24). Cannot be manually changed." />
            </label>
            <NumberInput value={fps} onChange={() => {}} disabled={true} />
          </div>
        </div>


        <ValidationTooltip error={!prompt.trim() ? "Please enter a prompt to generate a video" : null} className="w-full">
          <button 
            className="btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed" 
            onClick={handleGenerate} 
            disabled={isLoading || !prompt.trim()}
          >
            {isLoading ? (<><Loader2 className="animate-spin" size={18} />Generating...</>) : (<><Film size={18} />Generate Video</>)}
          </button>
        </ValidationTooltip>
      </div>
      
      {error && (
        <div className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-200 flex items-start gap-2">
           <AlertTriangle className="shrink-0 mt-0.5" size={18} />
           <div>
             <p className="font-semibold">Generation Failed</p>
             <p className="text-sm opacity-90">{error}</p>
           </div>
        </div>
      )}

      {/* Job Info Modal */}
      {currentJobId && (
          <JobProgressModal jobId={currentJobId} onClose={handleCloseModal} />
      )}

      {result && (
        <div className="mt-6 card">
          <div className="flex items-center justify-between mb-4 text-primary">
            <h2 className="text-lg font-semibold">Result</h2>
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
            <video src={`http://localhost:8000/api/files/${result}`} controls className="max-w-full rounded-lg" />
            <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity pointer-events-none">
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
          fileName={result.split('/').pop() || 'video.mp4'}
        />
      )}
    </div>
  );
}
