
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

// Display info matching CLI exactly
const MODEL_DISPLAY_INFO: Record<string, { label: string; vram: string }> = {
  'musicgen-small': { label: 'MusicGen Small (Fast, Good for Music)', vram: '~4GB' },
  'musicgen-medium': { label: 'MusicGen Medium (Better Quality)', vram: '~8GB' },
  'musicgen-large': { label: 'MusicGen Large (Best Quality)', vram: '~14GB' },
  'audioldm2': { label: 'AudioLDM2 (Sound Effects & Audio)', vram: '~6GB' },
  'stable-audio': { label: 'Stable Audio (High Quality, 🔒 Gated)', vram: '~12GB' },
  'bark': { label: 'Bark (TTS & Audio - Transformer)', vram: '~12GB' },
};

const MODEL_ORDER = [
  'musicgen-small', 'musicgen-medium', 'musicgen-large', 
  'audioldm2', 'stable-audio', 'bark'
];

import { useState, useEffect } from 'react';
import { useAppStore } from '../store';
import { generateAudio, fetchModels } from '../hooks/useApi';
import { FileAudio, Loader2, AlertTriangle, AlertCircle } from 'lucide-react';
import { NumberInput } from './common/NumberInput';
import { Tooltip } from './common/Tooltip';
import { ValidationTooltip } from './common/ValidationTooltip';

export function AudioGenerator() {
  const { addJob } = useAppStore();
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState('musicgen-medium');
  const [duration, setDuration] = useState(10);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  // Fetch models on mount
  useEffect(() => {
    fetchModels()
      .then((data) => {
        if (data.audio) {
          setAvailableModels(data.audio);
        }
      })
      .catch((err) => console.error('Failed to fetch models:', err));
  }, []);

  // Update defaults when model changes
  useEffect(() => {
    if (model === 'bark') {
      // Bark duration is determined by text length, not parameter
      setDuration(0); 
    } else {
      setDuration(10);
    }
  }, [model]);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setIsLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await generateAudio({ prompt, model, duration });
      setCurrentJobId(response.job_id);
      addJob({
        job_id: response.job_id,
        type: 'audio',
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
            } else if (updatedJob.status === 'failed' || updatedJob.status === 'cancelled') {
                setIsLoading(false);
                setError(updatedJob.error || updatedJob.message || "Generation failed");
            }
        } else {
            // Job not found in store (removed on cancellation)
            setIsLoading(false);
            setCurrentJobId(null);
        }
    });
    return () => unsubscribe();
  }, [currentJobId]);

  // Sort models
  const sortedModels = MODEL_ORDER.filter(name => 
    availableModels.some(m => m.name === name)
  );

  return (
    <div className="w-full max-w-none px-4 mx-auto">
      <div className="card p-6 mb-8">
        <h1 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <FileAudio className="text-primary-400" />
          Audio Generation
        </h1>
      </div>

      <div className="card space-y-4">
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="label mb-0">Prompt</label>
            <RandomPrompt type="audio" onPromptSelect={setPrompt} />
          </div>
          <textarea
            className="input min-h-[150px] resize-y"
            placeholder="Upbeat electronic music with a driving beat..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
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
            {model === 'stable-audio' && (
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
            {model === 'bark' && (
              <div className="mt-2 flex items-center gap-2 text-blue-400 text-sm">
                <AlertCircle size={16} />
                <span>Duration depends on text length (approx 14s). Use [laughter], [music] tags.</span>
              </div>
            )}
          </div>
          <div>
            <label className="label flex items-center">
              Duration (seconds)
              <Tooltip content="Audio duration in seconds. Ignored for Bark (text-based length) and MusicGen-Small/Medium (fixed 30s tokens often). Default 10s." />
            </label>
            <NumberInput 
              value={duration} 
              onChange={setDuration} 
              min={1} 
              max={120} 
              disabled={model === 'bark'}
              placeholder={model === 'bark' ? 'Auto-determined' : '10'}
            />
          </div>
        </div>


        <ValidationTooltip error={!prompt.trim() ? "Please enter a prompt to generate audio" : null} className="w-full">
          <button 
            className="btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed" 
            onClick={handleGenerate} 
            disabled={isLoading || !prompt.trim()}
          >
            {isLoading ? (<><Loader2 className="animate-spin" size={18} />Generating...</>) : (<><FileAudio size={18} />Generate Audio</>)}
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
          <div className="p-4 bg-slate-900/50 rounded-lg border border-border">
            <audio src={`http://localhost:8000/api/files/${result}`} controls className="w-full" />
          </div>
        </div>
      )}

      {currentJobId && (
        <JobProgressModal jobId={currentJobId} onClose={() => {
          setCurrentJobId(null);
          setIsLoading(false);
        }} />
      )}

      {result && (
        <PreviewModal 
          isOpen={isPreviewOpen}
          onClose={() => setIsPreviewOpen(false)}
          filePath={result}
          fileName={result.split('/').pop() || 'audio.mp3'}
        />
      )}
    </div>
  );
}
