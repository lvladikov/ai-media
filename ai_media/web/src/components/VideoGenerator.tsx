import { useState, useEffect, useRef } from 'react';
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
import { ErrorAlert } from './common/ErrorAlert';
import { formatDuration } from '../utils/formatTime';

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
  const [genDuration, setGenDuration] = useState<number | null>(null);
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
    setGenDuration(null);
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
      } else {
        // Job not found in store (removed on cancellation)
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
    <div className="flex flex-col lg:flex-row h-full bg-slate-900 text-slate-200">
      {/* Parameters Sidebar */}
      <div className="w-full lg:w-[500px] border-b lg:border-b-0 lg:border-r border-slate-800 p-4 lg:py-6 lg:pr-[27px] lg:pl-0 flex flex-col gap-6 overflow-y-auto shrink-0 h-auto lg:h-full">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
            <Film className="text-brand-400" /> Video Gen
          </h2>
          <p className="text-xs text-slate-500">Generate videos from text descriptions</p>
        </div>

        {/* Prompt */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-slate-400">Prompt</label>
            <RandomPrompt type="video" onPromptSelect={setPrompt} />
          </div>
          <textarea
            className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm focus:outline-none focus:border-brand-500 resize-y min-h-[120px]"
            placeholder="A serene forest with sunlight filtering through the trees..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        {/* Model Selector */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-400">Model</label>
          <select
            className="select w-full bg-slate-950 border-slate-700 text-sm focus:border-brand-500"
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
          {model === 'cogvideox' && (
            <div className="flex items-center gap-2 text-red-400 text-xs mt-1">
              <AlertTriangle size={12} />
              <span>Extremely high VRAM (38GB+)</span>
            </div>
          )}
          {model === 'wan-2.2' && (
            <div className="flex items-center gap-2 text-amber-400 text-xs mt-1">
              <AlertTriangle size={12} />
              <span>Requires HF Login</span>
            </div>
          )}
          {model === 'svd' && (
            <div className="flex items-center gap-2 text-blue-400 text-xs mt-1">
              <Info size={12} />
              <span>Requires input image (not supported in text mode)</span>
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

        {/* Duration/FPS */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="text-xs font-medium text-slate-400 flex items-center gap-1">
              Duration (s)
              <Tooltip content="Length of video in seconds. Default 2s. Longer videos take significantly more VRAM/time." />
            </label>
            <NumberInput value={duration} onChange={setDuration} min={1} max={10} step={0.1} allowFloat={true} />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium text-slate-400 flex items-center gap-1">
              FPS (Read Only)
              <Tooltip content="Frames Per Second. Determined by model (e.g. 8, 24). Cannot be manually changed." />
            </label>
            <NumberInput value={fps} onChange={() => { }} disabled={true} />
          </div>
        </div>

        <ErrorAlert error={error} onDismiss={() => setError(null)} />

        <ValidationTooltip error={!prompt.trim() ? "Please enter a prompt" : null} className="w-full mt-auto pt-4">
          <button
            className="w-full bg-gradient-to-r from-brand-600 to-pink-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-white font-bold py-3 rounded-lg shadow-lg shadow-brand-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all"
            onClick={handleGenerate}
            disabled={isLoading || !prompt.trim()}
          >
            {isLoading ? (<><Loader2 className="animate-spin" size={18} /> Generating...</>) : (<><Film size={18} /> Generate Video</>)}
          </button>
        </ValidationTooltip>
      </div>

      {/* Main Preview */}
      <div ref={resultRef} className="flex-1 p-6 flex items-center justify-center bg-slate-950/30 min-h-[500px] lg:min-h-0 scroll-mt-4">
        {result ? (
          <div className="flex flex-col items-center justify-center max-w-full h-full gap-4">
            <div
              className="relative group rounded-lg overflow-hidden border border-brand-500/30 shadow-2xl max-h-[85vh] cursor-pointer"
              onClick={() => setIsPreviewOpen(true)}
            >
              <video src={`http://localhost:8000/api/files/${result}`} controls className="max-h-[85vh] object-contain" />
              <div className="absolute top-2 left-2 bg-brand-600 px-2 py-1 rounded text-[10px] sm:text-xs text-white shadow-lg flex flex-col items-start leading-none gap-0.5">
                <span className="font-bold uppercase tracking-wider">Result</span>
                {genDuration && <span className="opacity-80 font-medium">in {formatDuration(genDuration * 1000)}</span>}
              </div>
              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity pointer-events-none">
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
            <Film size={48} className="mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-medium mb-2">Ready to Generate</h3>
            <p className="text-slate-400 max-w-sm">
              Enter a prompt in the <span className="lg:hidden">controls above</span><span className="hidden lg:inline">sidebar</span> to start creating videos.
            </p>
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
