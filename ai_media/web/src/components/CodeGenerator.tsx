import { useState, useEffect } from 'react';
import { useAppStore } from '../store';
import { generateCode, fetchModels } from '../hooks/useApi';
import { Code, Loader2, AlertTriangle } from 'lucide-react';
import { RandomPrompt } from './common/RandomPrompt';
import { Tooltip } from './common/Tooltip';
import { ValidationTooltip } from './common/ValidationTooltip';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';

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
  const [filename, setFilename] = useState('script.py');
  const [language, setLanguage] = useState('python');
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
        if (data.text) {
          setAvailableModels(data.text);
        }
      })
      .catch((err) => console.error('Failed to fetch models:', err));
  }, []);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setIsLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await generateCode({ 
        prompt, 
        model, 
        filename,
        language
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
                setIsLoading(false);
            } else if (job.status === 'failed') {
                setIsLoading(false);
                setError(job.error || job.message || "Generation failed");
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
    name === 'default' || availableModels.some(m => m.name === name)
  );

  return (
    <div className="max-w-4xl">
      <h1 className="text-2xl font-bold mb-6 flex items-center gap-2">
        <Code className="text-primary-400" />
        Code Generator
      </h1>

      <div className="card space-y-4">
        <div>
          <div className="flex items-center justify-between mb-2">
          <label className="label flex items-center mb-0">
             Instructions
             <Tooltip content="Describe the code you want to generate. Be specific about libraries, functionality, and input/output." />
          </label>
          <RandomPrompt type="code" onPromptSelect={setPrompt} />
          </div>
          <textarea
            className="input min-h-[100px] resize-y"
            placeholder="Write a Python script to scrape a website..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="label flex items-center">
               Model
               <Tooltip content="LLM for coding. DeepSeek R1 / Qwen-2.5-Coder are recommended for best logic." />
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
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label flex items-center">
                 Language
                 <Tooltip content="Programming language for syntax highlighting and file extension." />
              </label>
              <select className="select" value={language} onChange={(e) => setLanguage(e.target.value)}>
                <option value="python">Python</option>
                <option value="javascript">JavaScript</option>
                <option value="typescript">TypeScript</option>
                <option value="html">HTML</option>
                <option value="css">CSS</option>
                <option value="bash">Bash</option>
                <option value="sql">SQL</option>
                <option value="go">Go</option>
                <option value="rust">Rust</option>
              </select>
            </div>
             <div>
              <label className="label flex items-center">
                 Filename
                 <Tooltip content="Output filename. Ensure extension matches language (e.g. .py, .js)." />
              </label>
              <input 
                type="text" 
                className="input" 
                value={filename} 
                onChange={(e) => setFilename(e.target.value)}
              />
            </div>
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
          <h2 className="text-lg font-semibold mb-4 text-primary">Result</h2>
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
          fileName={result.split('/').pop() || filename}
        />
      )}
    </div>
  );
}
