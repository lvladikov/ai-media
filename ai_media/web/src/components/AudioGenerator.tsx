
import { RandomPrompt } from './common/RandomPrompt';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { ErrorAlert } from './common/ErrorAlert';

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

import { useState, useEffect, useRef } from 'react';
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
            <FileAudio className="text-brand-400" /> Audio Gen
          </h2>
          <p className="text-xs text-slate-500">Generate sound effects and music</p>
        </div>

        {/* Prompt */}
        <div className="space-y-2">
           <div className="flex items-center justify-between">
             <label className="text-sm font-medium text-slate-400">Prompt</label>
             <RandomPrompt type="audio" onPromptSelect={setPrompt} />
           </div>
           <textarea
             className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm focus:outline-none focus:border-brand-500 resize-y min-h-[120px]"
             placeholder="Upbeat electronic music with a driving beat..."
             value={prompt}
             onChange={(e) => setPrompt(e.target.value)}
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
           {model === 'stable-audio' && (
             <div className="flex items-center gap-2 text-amber-400 text-xs mt-1">
               <AlertTriangle size={12} />
               <span>Requires HF Login</span>
             </div>
           )}
           {model === 'bark' && (
             <div className="flex items-center gap-2 text-blue-400 text-xs mt-1">
               <AlertCircle size={12} />
               <span>Duration based on text length (~14s)</span>
             </div>
           )}
        </div>

        {/* Duration */}
        <div className="space-y-2">
           <label className="text-xs font-medium text-slate-400 flex items-center gap-1">
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

        <ErrorAlert error={error} onDismiss={() => setError(null)} />

        <ValidationTooltip error={!prompt.trim() ? "Please enter a prompt" : null} className="w-full mt-auto pt-4">
          <button 
            className="w-full bg-gradient-to-r from-brand-600 to-purple-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-white font-bold py-3 rounded-lg shadow-lg shadow-brand-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all" 
            onClick={handleGenerate} 
            disabled={isLoading || !prompt.trim()}
          >
            {isLoading ? (<><Loader2 className="animate-spin" size={18} /> Generating...</>) : (<><FileAudio size={18} /> Generate Audio</>)}
          </button>
        </ValidationTooltip>
      </div>

      {/* Main Preview */}
      <div ref={resultRef} className="flex-1 p-6 flex items-center justify-center bg-slate-950/30 min-h-[500px] lg:min-h-0 scroll-mt-4">
        {result ? (
           <div className="flex flex-col items-center justify-center max-w-2xl w-full gap-6">
                <div className="w-full p-8 bg-slate-900/80 backdrop-blur-sm rounded-2xl border border-brand-500/30 shadow-2xl flex flex-col items-center gap-6">
                   <div className="w-24 h-24 rounded-full bg-brand-500/10 flex items-center justify-center text-brand-400 animate-pulse">
                      <FileAudio size={48} />
                   </div>
                   <audio src={`http://localhost:8000/api/files/${result}`} controls className="w-full" />
                   <p className="text-sm text-slate-400">Generated Audio Result</p>
                </div>
                
                <div className="flex gap-2">
                   <a href={`http://localhost:8000/api/files/${result}`} target="_blank" rel="noreferrer" className="btn-secondary text-sm">Download</a>
                </div>
           </div>
        ) : (
          <div className="text-center text-slate-500">
            <FileAudio size={48} className="mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-medium mb-2">Ready to Compose</h3>
            <p className="max-w-sm mx-auto">Describe the sound or music you want to create.</p>
          </div>
        )}
      </div>

      {currentJobId && (
        <JobProgressModal 
          jobId={currentJobId} 
          onClose={() => {
            setCurrentJobId(null);
            setIsLoading(false);
          }} 
          onViewResult={handleViewResult}
        />
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
