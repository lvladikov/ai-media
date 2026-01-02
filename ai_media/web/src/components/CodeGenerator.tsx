import { useState, useEffect } from 'react';
import { useAppStore } from '../store';
import { generateCode, fetchModels } from '../hooks/useApi';
import { Code, Loader2, AlertTriangle, Download, FolderArchive, X } from 'lucide-react';
import { RandomPrompt } from './common/RandomPrompt';
import { Tooltip } from './common/Tooltip';
import { ValidationTooltip } from './common/ValidationTooltip';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { ProjectPreviewModal } from './ProjectPreviewModal';
import { formatDuration } from '../utils/formatTime';
import { API_BASE_URL } from '../config';

interface ModelInfo {
  name: string;
}

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

export function CodeGenerator() {
  const { addJob } = useAppStore();
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState('default');
  const [outputName, setOutputName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [isMultiFile, setIsMultiFile] = useState(false);
  const [generatedFiles, setGeneratedFiles] = useState<string[]>([]);
  const [reasoning, setReasoning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [showProjectPreview, setShowProjectPreview] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(256);

  // Fetch models on mount, use WebSocket for cleanup (same as Chat)
  useEffect(() => {
    fetchModels()
      .then((data) => {
        if (data.text) {
          setAvailableModels(data.text);
        }
      })
      .catch((err) => console.error('Failed to fetch models:', err));
    
    // Connect to a lightweight endpoint for cleanup on disconnect (like Chat)
    const ws = new WebSocket(`ws://localhost:8000/ws/code`);
    
    // Cleanup: Close socket when navigating away - server unloads model on disconnect
    return () => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    };
  }, []);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setIsLoading(true);
    setResult(null);
    setDuration(null);
    setReasoning(null);
    setIsMultiFile(false);
    setError(null);

    try {
      const response = await generateCode({ 
        prompt, 
        model, 
        output_name: outputName || undefined,
      });
      
      setCurrentJobId(response.job_id);
      
      addJob({
        job_id: response.job_id,
        type: 'code',
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

  // Watch for job completion
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
                    // Fallback to total time if start time missing
                    const start = new Date(job.created_at).getTime();
                    const end = new Date(job.updated_at).getTime();
                    setDuration(Math.round((end - start) / 1000));
                }
                
                // Check if multi-file by looking at is_multi_file flag
                const multiFile = job.is_multi_file || false;
                setIsMultiFile(multiFile);
                setGeneratedFiles(job.generated_files || []);
                setReasoning(job.reasoning || null);
                setIsLoading(false);
            } else if (job.status === 'failed' || job.status === 'cancelled') {
                setIsLoading(false);
                
                let msg = job.error || job.message || "Generation failed";
                if (job.status === 'cancelled') {
                   msg += " (Note: If model was downloading, it may finish before cleanup)";
                }
                setError(msg);
                
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

  // Sort models - exclude llama-3.1-8b since it's the same as 'default'
  const sortedModels = MODEL_ORDER.filter(name => 
    (name === 'default' || availableModels.some(m => m.name === name)) && name !== 'llama-3.1-8b'
  );

  // Get display name for result
  const resultDisplayName = result ? result.split('/').pop() : '';

  return (
    <div className="w-full max-w-none px-4 mx-auto">
      <div className="card p-6 mb-8">
        <h1 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Code className="text-primary-400" />
          Code Generation
        </h1>
      </div>

      <div className="card space-y-4">
        <div>
          <div className="flex items-center justify-between mb-2">
          <label className="label flex items-center mb-0">
             Instructions
             <Tooltip content="Describe the code you want to generate. Be specific about libraries, functionality, and input/output. The model will determine the appropriate language and file structure." />
          </label>
          <RandomPrompt type="code" onPromptSelect={setPrompt} />
          </div>
          <textarea
            className="input min-h-[180px] resize-y"
            placeholder="Create a To-Do list app with HTML, CSS, and JavaScript..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="label flex items-center">
               Model
               <Tooltip content="LLM for coding. DeepSeek R1 models are recommended for best logic." />
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
             {/* Recommendation for Code */}
             {(model.includes('deepseek') || model.includes('qwen')) && (
              <div className="mt-1 text-green-400 text-xs text-right">
                Recommended for coding
              </div>
            )}
          </div>
          
          <div>
            <label className="label flex items-center">
               Output
               <Tooltip content="Optional output name. Leave empty for auto-generated names based on content. For multi-file projects, this becomes the zip filename." />
            </label>
            <input 
              type="text" 
              className="input" 
              placeholder="Leave empty for auto-generated name"
              value={outputName} 
              onChange={(e) => setOutputName(e.target.value)}
            />
          </div>
        </div>


        <ValidationTooltip error={!prompt.trim() ? "Please enter instructions for the code" : null} className="w-full">
          <button 
            className="btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed" 
            onClick={handleGenerate} 
            disabled={isLoading || !prompt.trim()}
          >
            {isLoading ? (<><Loader2 className="animate-spin" size={18} />Generating...</>) : (<><Code size={18} />Generate Code</>)}
          </button>
        </ValidationTooltip>
      </div>

      {error && (
        <div className="mt-6 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-200 flex items-start gap-2 relative group">
           <AlertTriangle className="shrink-0 mt-0.5" size={18} />
           <div className="flex-1 pr-6">
             <p className="font-semibold">Generation Failed</p>
             <p className="text-sm opacity-90">{error}</p>
           </div>
           <button 
             onClick={() => setError(null)}
             className="absolute top-4 right-4 text-red-300 hover:text-white transition-colors p-1"
             aria-label="Dismiss error"
           >
             <X size={16} />
           </button>
        </div>
      )}

      {result && (
        <div className="mt-6 card">
          <div className="flex items-end justify-between mb-4">
             <h2 className="text-lg font-semibold text-primary">Result</h2>
             {duration && <span className="text-xs text-secondary mb-1">Generated in {formatDuration(duration * 1000)}</span>}
          </div>
          <div className="p-4 bg-tertiary rounded-lg flex items-center justify-between border border-border">
             <span className="truncate text-primary flex items-center gap-2">
               {isMultiFile ? <FolderArchive size={18} className="text-brand-400" /> : <Code size={18} className="text-brand-400" />}
               {resultDisplayName}{isMultiFile && '.zip'}
             </span>
             <div className="flex gap-2">
                {!isMultiFile ? (
                  <button 
                    className="btn-primary text-sm"
                     onClick={() => setIsPreviewOpen(true)}
                  >
                    Preview
                  </button>
                ) : (
                  <button 
                    className="btn-primary text-sm"
                     onClick={() => setShowProjectPreview(true)}
                  >
                    Preview Project
                  </button>
                )}
                <a 
                  href={isMultiFile 
                    ? `${API_BASE_URL}/api/files/zip?path=${encodeURIComponent(result)}` 
                    : `${API_BASE_URL}/api/files/${result}`
                  } 
                  target="_blank" 
                  rel="noreferrer" 
                  className="btn-secondary text-sm flex items-center gap-1"
                >
                  <Download size={14} />
                  {isMultiFile ? 'Download ZIP' : 'Download'}
                </a>
             </div>
          </div>
          
          {/* Multi-file project explanation */}
          {isMultiFile && (
            <div className="mt-4 p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
              <div className="flex items-start gap-3">
                <FolderArchive size={20} className="text-brand-400 mt-0.5 shrink-0" />
                <div className="flex-1">
                  <p className="text-sm text-primary font-medium mb-2">
                    Multi-File Project ({generatedFiles.length} files)
                  </p>
                  <p className="text-xs text-secondary mb-3">
                     Click "Preview Project" above to browse all files in a VS Code-like interface.
                  </p>
                </div>
              </div>
            </div>
          )}

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
            if (result) {
              if (isMultiFile) setShowProjectPreview(true);
              else setIsPreviewOpen(true);
            }
          }} 
        />
      )}
      {result && isMultiFile && (
        <ProjectPreviewModal
            isOpen={showProjectPreview}
            onClose={() => setShowProjectPreview(false)}
            resultPath={result}
            sidebarWidth={sidebarWidth}
            onSidebarWidthChange={setSidebarWidth}
        />
      )}
      {result && !isMultiFile && (
        <PreviewModal 
          isOpen={isPreviewOpen}
          onClose={() => setIsPreviewOpen(false)}
          filePath={result || ''}
          fileName={resultDisplayName || 'code'}
        />
      )}
    </div>
  );
}
