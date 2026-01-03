import { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../store';
import { generateCode, fetchModels } from '../hooks/useApi';
import { Code, Loader2, Download, FolderArchive, AlertTriangle } from 'lucide-react';
import { RandomPrompt } from './common/RandomPrompt';

import { ValidationTooltip } from './common/ValidationTooltip';
import { ErrorAlert } from './common/ErrorAlert';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { ProjectPreviewModal } from './ProjectPreviewModal';
import { formatDuration } from '../utils/formatTime';
import { API_BASE_URL } from '../config';

interface ModelInfo {
  name: string;
  is_default?: boolean;
}

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
  'qwen3-coder-30b': { label: 'Qwen3 Coder 30B MoE (⚠️ 64GB RAM)', vram: '~10GB' },
  'qwen-coder-32b': { label: 'Qwen 2.5 Coder 32B (⚠️ 120GB RAM)', vram: '~24GB' },
  'qwen-coder-14b': { label: 'Qwen 2.5 Coder 14B', vram: '~12GB' },
  'qwen-coder-7b': { label: 'Qwen 2.5 Coder 7B', vram: '~6GB' },
};

const MODEL_ORDER = [
  'deepseek-r1-qwen-7b', 'deepseek-r1-qwen-14b', 'deepseek-r1-qwen-32b',
  'deepseek-r1-llama-8b', 'deepseek-r1-llama-70b',
  'llama-3.1-8b', 'mistral-nemo-12b', 'qwen-2.5-14b',
  'qwen3-coder-30b', 'qwen-coder-32b', 'qwen-coder-14b', 'qwen-coder-7b'
];

export function CodeGenerator() {
  const { addJob } = useAppStore();
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState('');
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

          const msg = job.status === 'cancelled'
            ? "Job cancelled."
            : (job.error || job.message || "Generation failed");
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

  // Sort models based on fixed order, filtering valid ones from API
  const sortedModels = MODEL_ORDER.filter(name =>
    availableModels.some(m => m.name === name)
  );

  // Get display name for result
  const resultDisplayName = result ? result.split('/').pop() : '';

  const resultRef = useRef<HTMLDivElement>(null);

  const handleViewResult = () => {
    setTimeout(() => {
      resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  };

  return (
    <div className="flex flex-col lg:flex-row h-full bg-slate-900 text-slate-200">
      {/* Parameters Sidebar */}
      <div className="w-full lg:w-[500px] border-b lg:border-b-0 lg:border-r border-slate-800 p-4 lg:py-6 lg:pr-[27px] lg:pl-1 flex flex-col gap-6 overflow-y-auto shrink-0 h-auto lg:h-full">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
            <Code className="text-brand-400" /> Code Gen
          </h2>
          <p className="text-xs text-slate-500">Generate applications, scripts and modules</p>
        </div>

        {/* Instructions */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-slate-400">Instructions</label>
            <RandomPrompt type="code" onPromptSelect={setPrompt} />
          </div>
          <textarea
            className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm focus:outline-none focus:border-brand-500 resize-y min-h-[160px]"
            placeholder="Create a To-Do list app with HTML, CSS, and JavaScript..."
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
          {/* Recommendation for Code */}
          {(model.includes('deepseek') || model.includes('qwen')) && (
            <div className="text-green-500 text-xs text-right mt-1">
              Recommended for coding
            </div>
          )}
        </div>

        {/* Output Name */}
        <div className="space-y-2">
          <label className="text-xs font-medium text-slate-400">
            Output Name (Optional)
          </label>
          <input
            type="text"
            className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-sm focus:outline-none focus:border-brand-500"
            placeholder="Auto-generated if empty"
            value={outputName}
            onChange={(e) => setOutputName(e.target.value)}
          />
        </div>

        <ErrorAlert error={error} onDismiss={() => setError(null)} />

        <ValidationTooltip error={!prompt.trim() ? "Please enter instructions" : null} className="w-full mt-auto pt-4">
          <button
            className="w-full bg-gradient-to-r from-brand-600 to-cyan-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-white font-bold py-3 rounded-lg shadow-lg shadow-brand-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all"
            onClick={handleGenerate}
            disabled={isLoading || !prompt.trim()}
          >
            {isLoading ? (<><Loader2 className="animate-spin" size={18} /> Generating...</>) : (<><Code size={18} /> Generate Code</>)}
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
                    {isMultiFile ? <FolderArchive size={24} /> : <Code size={24} />}
                  </div>
                  <div className="overflow-hidden">
                    <h3 className="font-medium text-slate-200 truncate max-w-md">{resultDisplayName}{isMultiFile && '.zip'}</h3>
                    {duration && <p className="text-xs text-slate-500">Generated in {formatDuration(duration * 1000)}</p>}
                  </div>
                </div>
                <div className="flex gap-2">
                  {!isMultiFile ? (
                    <button className="btn-secondary text-sm" onClick={() => setIsPreviewOpen(true)}>Preview</button>
                  ) : (
                    <button className="btn-secondary text-sm" onClick={() => setShowProjectPreview(true)}>Preview Project</button>
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
                <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
                  <div className="flex items-start gap-3">
                    <FolderArchive size={20} className="text-brand-400 mt-0.5 shrink-0" />
                    <div className="flex-1">
                      <p className="text-sm text-slate-300 font-medium mb-1">
                        Multi-File Project ({generatedFiles.length} files)
                      </p>
                      <p className="text-xs text-slate-500">
                        Click "Preview Project" above to browse the file structure.
                      </p>
                    </div>
                  </div>
                </div>
              )}

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
                  <p>Code generated successfully.</p>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="text-center text-slate-500">
            <Code size={48} className="mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-medium mb-2">Ready to Code</h3>
            <p className="max-w-sm mx-auto">Generate scripts, modules, or full applications by describing your requirements.</p>
          </div>
        )}
      </div>

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
          onViewResult={handleViewResult}
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
