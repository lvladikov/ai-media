import { useState, useEffect } from 'react';
import { useAppStore } from '../store';
import { generateArticle, fetchModels } from '../hooks/useApi';
import { FileText, Loader2, Globe, AlertTriangle } from 'lucide-react';
import { Tooltip } from './common/Tooltip';
import { ValidationTooltip } from './common/ValidationTooltip';
import { RandomPrompt } from './common/RandomPrompt';

interface ModelInfo {
  name: string;
}

import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { formatDuration } from '../utils/formatTime';

// Display names matching CLI
const MODEL_DISPLAY_INFO: Record<string, { label: string; vram: string }> = {
  'deepseek-r1-qwen-7b': { label: 'DeepSeek R1 7B (Reasoning)', vram: '~7GB' },
  'deepseek-r1-qwen-14b': { label: 'DeepSeek R1 14B (Reasoning)', vram: '~14GB' },
  'deepseek-r1-qwen-32b': { label: 'DeepSeek R1 32B (Reasoning)', vram: '~24GB' },
  'deepseek-r1-llama-8b': { label: 'DeepSeek R1 Llama 8B', vram: '~8GB' },
  'deepseek-r1-llama-70b': { label: 'DeepSeek R1 Llama 70B (High VRAM)', vram: '~40GB' },
  'llama-3.1-8b': { label: 'Llama 3.1 8B (Fast & Stable)', vram: '~8GB' },
  'mistral-nemo-12b': { label: 'Mistral Nemo 12B', vram: '~12GB' },
  'qwen-2.5-14b': { label: 'Qwen 2.5 14B Instruct', vram: '~14GB' },
  'default': { label: 'Default (Llama 3.1 8B)', vram: '~8GB' },
};

const MODEL_ORDER = [
  'default',
  'deepseek-r1-qwen-7b', 'deepseek-r1-qwen-14b', 'deepseek-r1-qwen-32b',
  'deepseek-r1-llama-8b', 'deepseek-r1-llama-70b',
  'llama-3.1-8b', 'mistral-nemo-12b', 'qwen-2.5-14b'
];

export function ArticleGenerator() {
  const { addJob } = useAppStore();
  const [topic, setTopic] = useState('');
  const [model, setModel] = useState('default');
  const [format, setFormat] = useState('md');
  const [length, setLength] = useState('quick');
  const [online, setOnline] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [reasoning, setReasoning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  // Fetch models on mount
  useEffect(() => {
    fetchModels()
      .then((data) => {
        if (data.text) {
          setAvailableModels(data.text);
        }
      })
      .catch((err) => console.error('Failed to fetch models:', err));
  }, []);

  const handleGenerate = async () => {
    if (!topic.trim()) return;
    setIsLoading(true);
    setResult(null);
    setDuration(null);
    setReasoning(null);
    setError(null);

    try {
      const response = await generateArticle({ 
        topic, 
        model, 
        format, 
        online, 
        length
      });
      
      setCurrentJobId(response.job_id);
      
      addJob({
        job_id: response.job_id,
        type: 'article',
        status: 'pending',
        progress: 0,
        phase: 'queued',
        message: 'Job queued',
        result_path: response.output_path,
        error: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });
      
      // Monitoring handled by WebSocket via JobProgressModal
    } catch (err) {
      console.error('Generation failed:', err);
      setIsLoading(false);
      setError("Failed to start generation job");
    }
  };

  // Watch for job completion to show result inline after modal closes
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
                    setDuration(seconds > 0 ? seconds : 1);
                } else if (job.created_at && job.updated_at) {
                    // Fallback
                    const start = new Date(job.created_at).getTime();
                    const end = new Date(job.updated_at).getTime();
                    setDuration(Math.round((end - start) / 1000));
                }
                
                setReasoning(job.reasoning || null);
                setIsLoading(false);
            } else if (job.status === 'failed' || job.status === 'cancelled') {
                setIsLoading(false);
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
    name === 'default' || availableModels.some(m => m.name === name)
  );

  return (
    <div className="w-full max-w-none px-4 mx-auto">
      <div className="card p-6 mb-8">
        <h1 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <FileText className="text-primary-400" />
          Article Generation
        </h1>
      </div>

      <div className="card space-y-4">
        <div>
          <div className="flex items-center justify-between mb-2">
             <label className="label flex items-center mb-0">
                Topic
                <Tooltip content="Main subject of the article. Be specific for better results." />
             </label>
             <RandomPrompt type="article" onPromptSelect={setTopic} />
          </div>
          <textarea
            className="input min-h-[150px] resize-y"
            placeholder="The future of renewable energy technologies..."
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="label flex items-center">
               Model
               <Tooltip content="LLM for generation. DeepSeek R1 for reasoning, Llama 3.1 for general speed." />
            </label>
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
            {model === 'deepseek-r1-llama-70b' && (
              <div className="mt-2 flex items-center gap-2 text-yellow-400 text-sm">
                <AlertTriangle size={16} />
                <span>Requires 40GB+ VRAM (Dual 3090/4090 or Mac Studio Ultra)</span>
              </div>
            )}
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label flex items-center">
                 Format
                 <Tooltip content="Output file format. Markdown is best for editing, HTML for web, PDF for sharing." />
              </label>
              <select className="select" value={format} onChange={(e) => setFormat(e.target.value)}>
                <option value="md">Markdown</option>
                <option value="html">HTML</option>
                <option value="pdf">PDF</option>
              </select>
            </div>
             <div>
              <label className="label flex items-center">
                 Length
                 <Tooltip content="Target word count. Quick ~500, Standard ~1500, Detailed ~3000 words." />
              </label>
              <select className="select" value={length} onChange={(e) => setLength(e.target.value)}>
                <option value="quick">Quick (~500 words)</option>
                <option value="standard">Standard (~1500 words)</option>
                <option value="detailed">Detailed (~3000 words)</option>
              </select>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input 
            type="checkbox" 
            id="online" 
            className="checkbox"
            checked={online}
            onChange={(e) => setOnline(e.target.checked)}
          />
          <label htmlFor="online" className="label cursor-pointer flex items-center gap-2">
            <Globe size={16} className={online ? 'text-green-400' : 'text-slate-500'} />
            Enable Online Research (DuckDuckGo Search)
            <Tooltip content="If enabled, the model will search the web for real-time information to include in the article." />
          </label>
        </div>


        <ValidationTooltip error={!topic.trim() ? "Please enter a topic for the article" : null} className="w-full">
          <button 
            className="btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed" 
            onClick={handleGenerate} 
            disabled={isLoading || !topic.trim()}
          >
            {isLoading ? (<><Loader2 className="animate-spin" size={18} />Generating...</>) : (<><FileText size={18} />Generate Article</>)}
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
          <div className="flex items-end justify-between mb-4">
             <h2 className="text-lg font-semibold text-primary">Result</h2>
             {duration && <span className="text-xs text-secondary mb-1">Generated in {formatDuration(duration * 1000)}</span>}
          </div>
          <div className="p-4 bg-tertiary rounded-lg flex items-center justify-between border border-border">
             <span className="truncate text-primary">{result.split('/').pop()}</span>
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
          
          {/* Reasoning / Thinking Block */}
          {reasoning && (
             <div className="mt-4 p-4 bg-slate-900/50 rounded-lg border border-slate-700/50">
               <div className="flex items-center gap-2 mb-2 text-primary-400">
                  <span className="text-xs font-semibold uppercase tracking-wider">Reasoning Process</span>
               </div>
               <div className="text-xs text-secondary font-mono whitespace-pre-wrap max-h-48 overflow-y-auto italic">
                  {reasoning}
               </div>
             </div>
          )}
        </div>
      )}
      {currentJobId && (
        <JobProgressModal 
          jobId={currentJobId} 
          onClose={() => {
            handleCloseModal();
            if (result) setIsPreviewOpen(true);
          }} 
        />
      )}
      {result && (
        <PreviewModal 
          isOpen={isPreviewOpen}
          onClose={() => setIsPreviewOpen(false)}
          filePath={result}
          fileName={result.split('/').pop() || 'article.md'}
        />
      )}
    </div>
  );
}
