
import { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../store';
import { generateArticle, fetchModels, type ModelInfo } from '../hooks/useApi';
import { FileText, Loader2, Globe, AlertTriangle } from 'lucide-react';

import { ValidationTooltip } from './common/ValidationTooltip';
import { RandomPrompt } from './common/RandomPrompt';
import { NumberInput } from './common/NumberInput';
import { ErrorAlert } from './common/ErrorAlert';


import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { formatDuration } from '../utils/formatTime';

// Display names matching CLI
const MODEL_DISPLAY_INFO: Record<string, { label: string; vram: string }> = {
  'deepseek-r1-qwen-7b': { label: 'DeepSeek R1 Qwen 7B (Reasoning)', vram: '~7GB' },
  'deepseek-r1-qwen-14b': { label: 'DeepSeek R1 Qwen 14B (Reasoning)', vram: '~14GB' },
  'deepseek-r1-qwen-32b': { label: 'DeepSeek R1 Qwen 32B (Reasoning)', vram: '~24GB' },
  'deepseek-r1-llama-8b': { label: 'DeepSeek R1 Llama 8B (Reasoning)', vram: '~8GB' },
  'deepseek-r1-llama-70b': { label: 'DeepSeek R1 Llama 70B (Reasoning)', vram: '~40GB' },
  'llama-3.1-8b': { label: 'Llama 3.1 8B (Fast & Stable)', vram: '~8GB' },
  'mistral-nemo-12b': { label: 'Mistral Nemo 12B', vram: '~12GB' },
  'qwen-2.5-14b': { label: 'Qwen 2.5 14B Instruct', vram: '~14GB' },
  'qwen3-coder-30b': { label: 'Qwen3 Coder 30B (MoE, 3.3B active)', vram: '~10GB' },
  'qwen-coder-32b': { label: 'Qwen 2.5 Coder 32B (⚠️ 120GB RAM)', vram: '~24GB' },
};

const MODEL_ORDER = [
  'deepseek-r1-qwen-7b', 'deepseek-r1-qwen-14b', 'deepseek-r1-qwen-32b',
  'deepseek-r1-llama-8b', 'deepseek-r1-llama-70b',
  'llama-3.1-8b', 'mistral-nemo-12b', 'qwen-2.5-14b', 'qwen3-coder-30b', 'qwen-coder-32b'
];

export function ArticleGenerator() {
  const { addJob } = useAppStore();
  const [topic, setTopic] = useState('');
  const [model, setModel] = useState('');
  const [format, setFormat] = useState('md');
  const [length, setLength] = useState("quick");
  const [online, setOnline] = useState(false);
  const [researchIterations, setResearchIterations] = useState(3);
  const [maxImages, setMaxImages] = useState(5);
  const [filename, setFilename] = useState('');
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
          
          // Find default model based on backend flag
          let initialModel = '';
          const defaultModel = data.text.find((m: ModelInfo) => m.is_default);
          
          if (defaultModel) {
             initialModel = defaultModel.name;
          } else if (data.text.length > 0) {
             initialModel = data.text[0].name;
          }
          
          setModel(initialModel);
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
        model: model || undefined, 
        format, 
        online, 
        length,
        research_iterations: researchIterations,
        max_images: maxImages,
        output_filename: filename || undefined
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
            } else if (job.status === 'failed') {
                setIsLoading(false);
            } else if (job.status === 'cancelled') {
                setIsLoading(false);
                setError("Job cancelled.");
                
                // Auto-dismiss after 6 seconds
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

  // Sort models based on fixed order, filtering valid ones from API
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
            <FileText className="text-brand-400" /> Article Gen
          </h2>
          <p className="text-xs text-slate-500">Generate articles, blogs and essays</p>
        </div>

        {/* Topic */}
        <div className="space-y-2">
           <div className="flex items-center justify-between">
             <label className="text-sm font-medium text-slate-400">Topic</label>
             <RandomPrompt type="article" onPromptSelect={setTopic} />
           </div>
           <textarea
             className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm focus:outline-none focus:border-brand-500 resize-y min-h-[120px]"
             placeholder="The future of renewable energy technologies..."
             value={topic}
             onChange={(e) => setTopic(e.target.value)}
           />
        </div>

        {/* Filename */}
        <div className="space-y-2">
           <label className="text-xs font-medium text-slate-400">
              Filename (Optional)
           </label>
           <input
             type="text"
             className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-sm focus:outline-none focus:border-brand-500"
             placeholder="Auto-generated if empty"
             value={filename}
             onChange={(e) => setFilename(e.target.value)}
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
               {!model && <option value="">Loading...</option>}
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
              <div className="flex items-center gap-2 text-amber-400 text-xs mt-1">
                <AlertTriangle size={12} />
                <span>Requires 40GB+ VRAM</span>
              </div>
            )}
        </div>

        {/* Format & Length */}
        <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="text-xs font-medium text-slate-400">Format</label>
              <select 
                className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-sm focus:outline-none focus:border-brand-500" 
                value={format} 
                onChange={(e) => setFormat(e.target.value)}
              >
                <option value="md">Markdown</option>
                <option value="html">HTML</option>
                <option value="pdf">PDF</option>
                <option value="docx">Word</option>
                <option value="txt">Text</option>
                <option value="json">JSON</option>
              </select>
            </div>
             <div className="space-y-1">
              <label className="text-xs font-medium text-slate-400">Length</label>
              <select 
                className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-sm focus:outline-none focus:border-brand-500" 
                value={length} 
                onChange={(e) => setLength(e.target.value)}
              >
                <option value="quick">Quick</option>
                <option value="standard">Standard</option>
                <option value="detailed">Detailed</option>
              </select>
            </div>
        </div>

        {/* Online Research */}
         <div className="space-y-3 pt-2 border-t border-slate-800">
            <label className="flex items-center gap-2 cursor-pointer">
               <input 
                 type="checkbox" 
                 className="checkbox checkbox-xs checkbox-primary"
                 checked={online}
                 onChange={(e) => setOnline(e.target.checked)}
               />
               <span className="text-sm font-medium text-slate-300 flex items-center gap-2">
                 <Globe size={14} className="text-brand-400" />
                 Online Research
               </span>
            </label>

            {online && (
              <div className="grid grid-cols-2 gap-4 pl-6">
                <div>
                  <label className="text-xs text-slate-400">Sources</label>
                  <NumberInput
                    value={researchIterations}
                    onChange={setResearchIterations}
                    min={1}
                    max={10}
                    className="w-full h-8 text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-400">Max Images</label>
                  <NumberInput
                    value={maxImages}
                    onChange={setMaxImages}
                    min={0}
                    max={20}
                    className="w-full h-8 text-sm"
                  />
                </div>
              </div>
            )}
        </div>

        <ErrorAlert error={error} onDismiss={() => setError(null)} />

        <ValidationTooltip error={!topic.trim() ? "Please enter a topic" : null} className="w-full mt-auto pt-4">
          <button 
            className="w-full bg-gradient-to-r from-brand-600 to-cyan-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-white font-bold py-3 rounded-lg shadow-lg shadow-brand-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all" 
            onClick={handleGenerate} 
            disabled={isLoading || !topic.trim()}
          >
            {isLoading ? (<><Loader2 className="animate-spin" size={18} /> Generating...</>) : (<><FileText size={18} /> Generate Article</>)}
          </button>
        </ValidationTooltip>
      </div>

      {/* Main Result Area */}
      <div ref={resultRef} className="flex-1 p-6 flex items-center justify-center bg-slate-950/30 scroll-mt-4">
        {result ? (
           <div className="flex flex-col items-center justify-center max-w-3xl w-full gap-6 h-full">
                <div className="w-full p-8 bg-slate-900/80 backdrop-blur-sm rounded-2xl border border-brand-500/30 shadow-2xl flex flex-col gap-6 relative overflow-hidden">
                   <div className="flex items-center justify-between border-b border-slate-700/50 pb-4">
                      <div className="flex items-center gap-3">
                         <div className="w-10 h-10 rounded-lg bg-brand-500/20 flex items-center justify-center text-brand-400">
                            <FileText size={24} />
                         </div>
                         <div>
                            <h3 className="font-medium text-slate-200">{result?.split('/').pop()}</h3>
                            {duration && <p className="text-xs text-slate-500">Generated in {formatDuration(duration * 1000)}</p>}
                         </div>
                      </div>
                      <div className="flex gap-2">
                         <button className="btn-secondary text-sm" onClick={() => setIsPreviewOpen(true)}>Preview</button>
                         <a href={`http://localhost:8000/api/files/${result}`} target="_blank" rel="noreferrer" className="btn-secondary text-sm">Download</a>
                      </div>
                   </div>

                   {/* Reasoning / Thinking Block */}
                   {reasoning && (
                      <div className="p-4 bg-slate-950/50 rounded-lg border border-slate-800/50 max-h-[40vh] overflow-y-auto">
                        <div className="flex items-center gap-2 mb-2 text-brand-400">
                           <span className="text-xs font-bold uppercase tracking-wider opacity-70">Reasoning Process</span>
                        </div>
                        <div className="text-xs text-slate-400 font-mono whitespace-pre-wrap italic leading-relaxed">
                           {reasoning}
                        </div>
                      </div>
                   )}
                   
                   {!reasoning && (
                       <div className="text-center py-12 text-slate-500">
                          <p>Article generated successfully.</p>
                       </div>
                   )}
                </div>
           </div>
        ) : (
          <div className="text-center text-slate-500">
            <FileText size={48} className="mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-medium mb-2">Ready to Write</h3>
            <p className="max-w-sm mx-auto">Enter a topic to generate a comprehensive article, blog post, or essay.</p>
          </div>
        )}
      </div>

      {currentJobId && (
        <JobProgressModal 
          jobId={currentJobId} 
          onClose={() => {
            handleCloseModal();
            if (result) setIsPreviewOpen(true);
          }} 
          onViewResult={handleViewResult}
        />
      )}
      {result && (
        <PreviewModal 
          isOpen={isPreviewOpen}
          onClose={() => setIsPreviewOpen(false)}
          filePath={result}
          fileName={result?.split('/').pop() || 'article.md'}
        />
      )}
    </div>
  );
}
