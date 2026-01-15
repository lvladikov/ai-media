import { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../store';
import { generateCode, useModels } from '../hooks/useApi';
import { Code, Loader2, Download, FolderArchive, AlertTriangle, HelpCircle } from 'lucide-react';
import { getDynamicRam } from '../utils/modelResources';
import { RandomPrompt } from './common/RandomPrompt';

import { ValidationTooltip } from './common/ValidationTooltip';
import { ErrorAlert } from './common/ErrorAlert';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { ProjectPreviewModal } from './ProjectPreviewModal';
import { formatDuration } from '../utils/formatTime';
import { ModelHelpLink } from './common/ModelHelpLink';
import { API_BASE_URL } from '../config';



// Display names matching CLI
const MODEL_DISPLAY_INFO: Record<string, { label: string }> = {
  'deepseek-r1-qwen-7b': { label: 'DeepSeek R1 Qwen 7B (Reasoning)' },
  'deepseek-r1-qwen-14b': { label: 'DeepSeek R1 Qwen 14B (Reasoning)' },
  'deepseek-r1-qwen-32b': { label: 'DeepSeek R1 Qwen 32B (Reasoning)' },
  'deepseek-r1-llama-8b': { label: 'DeepSeek R1 Llama 8B (Reasoning)' },
  'deepseek-r1-llama-70b': { label: 'DeepSeek R1 Llama 70B (Reasoning)' },
  'llama-3.1-8b': { label: 'Llama 3.1 8B (Fast & Stable, 🔒 Gated)' },
  'mistral-nemo-12b': { label: 'Mistral Nemo 12B' },
  'qwen3-8b': { label: 'Qwen 3 8B (Reasoning)' },
  'qwen3-14b': { label: 'Qwen 3 14B (Reasoning)' },
  'qwen3-opus-4.5-8b': { label: 'Qwen 3 Opus 4.5 Distill (8B)' },
  'qwen3-opus-4.5-14b': { label: 'Qwen 3 Opus 4.5 Distill (14B)' },
  'qwen3-gpt-5.2-8b': { label: 'Qwen 3 GPT-5.2 Distill (8B)' },
  'qwen3-gpt-5.2-14b': { label: 'Qwen 3 GPT-5.2 Distill (14B)' },
  'qwen3-coder-30b': { label: 'Qwen3 Coder 30B MoE' },
  'qwen-coder-32b': { label: 'Qwen 2.5 Coder 32B' },
  'qwen-coder-14b': { label: 'Qwen 2.5 Coder 14B' },
  'qwen-coder-7b': { label: 'Qwen 2.5 Coder 7B' },
};

const MODEL_ORDER = [
  'deepseek-r1-qwen-7b', 'deepseek-r1-qwen-14b', 'deepseek-r1-qwen-32b',
  'deepseek-r1-llama-8b', 'deepseek-r1-llama-70b',
  'llama-3.1-8b', 'mistral-nemo-12b', 'qwen3-14b', 'qwen3-8b',
  'qwen3-opus-4.5-14b', 'qwen3-opus-4.5-8b', 'qwen3-gpt-5.2-14b', 'qwen3-gpt-5.2-8b',
  'qwen3-coder-30b', 'qwen-coder-32b', 'qwen-coder-14b', 'qwen-coder-7b'
];

export function CodeGenerator() {
  const { addJob } = useAppStore();
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState('');
  const [precision, setPrecision] = useState('auto');
  const [framework, setFramework] = useState(navigator.userAgent.toLowerCase().includes('mac') ? 'mlx' : 'auto');
  const [outputName, setOutputName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [isMultiFile, setIsMultiFile] = useState(false);
  const [generatedFiles, setGeneratedFiles] = useState<string[]>([]);
  const [reasoning, setReasoning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [showProjectPreview, setShowProjectPreview] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(256);

  // Fetch models on mount, use WebSocket for cleanup (same as Chat)
  // Use global models cache
  const { models } = useModels();
  const availableModels = models?.text || [];

  // Set default model when available
  useEffect(() => {
    if (availableModels.length > 0 && !model) {
      // Find default model based on backend flag
      const defaultModel = availableModels.find((m: any) => m.is_default);
      let initialModel = '';

      if (defaultModel) {
        initialModel = defaultModel.name;
      } else if (availableModels.length > 0) {
        initialModel = availableModels[0].name;
      }

      if (initialModel) {
        setModel(initialModel);
      }
    }
  }, [availableModels, model]);

  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Connect to a lightweight endpoint for cleanup on disconnect (like Chat)
    const wsBaseUrl = API_BASE_URL();
    const wsUrl = wsBaseUrl.replace(/^http/, 'ws');

    // Delay connection slightly to prevent race conditions (warnings) during quick tab switches or Strict Mode
    // If the component unmounts before this fires (e.g. quick tab switch), the socket is never created.
    const connectTimeout = setTimeout(() => {
      const ws = new WebSocket(`${wsUrl}/ws/code`);
      socketRef.current = ws;

      // Add error handler to prevent uncaught exceptions in console
      ws.onerror = () => {
        // Silently ignore connection errors
      };
    }, 500);

    // Cleanup: Close socket when navigating away - server unloads model on disconnect
    return () => {
      clearTimeout(connectTimeout);

      const ws = socketRef.current;
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close();
      }
      socketRef.current = null;
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
        precision: precision !== 'auto' ? precision : undefined,
        framework: framework !== 'auto' ? framework : undefined,
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

    let hasSeenJob = false;

    const unsubscribe = useAppStore.subscribe((state) => {
      const job = state.jobs.find(j => j.job_id === currentJobId);
      if (job) {
        hasSeenJob = true;

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
      } else if (hasSeenJob) {
        // Job was in store but now removed (e.g., cancelled)
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
    <div className="flex flex-col lg:flex-row h-full bg-primary text-primary">
      {/* Parameters Sidebar */}
      <div className="w-full lg:w-[500px] border-b lg:border-b-0 lg:border-r border-border p-4 lg:py-6 lg:pr-[27px] lg:pl-1 flex flex-col gap-6 overflow-y-auto shrink-0 h-auto lg:h-full">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
            <Code className="text-brand-400" /> Code Gen
          </h2>
          <p className="text-xs text-tertiary">Generate applications, scripts and modules</p>
        </div>

        {/* Instructions */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="label">Instructions</label>
            <RandomPrompt type="code" onPromptSelect={setPrompt} />
          </div>
          <textarea
            className="w-full bg-primary border border-border rounded-lg p-3 text-sm focus:outline-none focus:border-brand-500 resize-y min-h-[160px]"
            placeholder="Enter your coding instructions or use the Random Prompt tool..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        {/* Model Selector */}
        {/* Model Selector Section */}
        <div className="space-y-4">

          {/* 1. Precision Selector */}
          {/* 1. Framework Selector (First on list, hidden if not Mac) */}
          <div className={`space-y-1 ${!navigator.userAgent.toLowerCase().includes('mac') ? 'hidden' : ''}`}>
            <label className="label">Platform</label>
            <select
              className="select w-auto bg-primary border-border text-sm focus:border-brand-500 max-w-full"
              value={framework}
              onChange={(e) => setFramework(e.target.value)}
              disabled={isLoading}
              title="Inference Framework - Use MLX for best performance on Mac"
            >
              <option value="mlx">MLX (Native Mac)</option>
              <option value="torch">PyTorch (MPS)</option>
            </select>
          </div>

          {/* 2. Precision Selector */}
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <label className="label">Precision</label>
              <button
                onClick={() => useAppStore.getState().openHelpSection('precision')}
                className="text-tertiary hover:text-brand-500 transition-colors"
                title="Learn about precision options"
              >
                <HelpCircle size={14} />
              </button>
            </div>
            <select
              className="select w-auto bg-primary border-border text-sm focus:border-brand-500 max-w-full"
              value={precision}
              onChange={(e) => setPrecision(e.target.value)}
              disabled={isLoading}
              title="Model precision - affects speed and memory usage"
            >
              <option value="auto">
                {/* Dynamic default label based on Framework */}
                {(() => {
                  const isMac = navigator.userAgent.toLowerCase().includes('mac');
                  const isMlx = framework === 'mlx' || (framework === 'auto' && isMac);
                  return `Auto (${isMlx ? 'int4 - MLX Default' : 'bfloat16 - Default'})`;
                })()}
              </option>
              <option value="int4">int4 (4-bit, Fast)</option>
              <option value="int6">int6 (6-bit, Balanced Speed)</option>
              <option value="int8">int8 (8-bit, Balanced Quality)</option>
              <option value="float16">float16 (Half)</option>
              <option value="bfloat16">bfloat16 (Brain Float)</option>
              <option value="float32">float32 (Full)</option>
            </select>
          </div>

          {/* 3. Model Selector */}
          <div className="space-y-2">
            <label className="label flex items-center">
              Model
              <ModelHelpLink section="code" />
            </label>
            <select
              className="select w-full bg-primary border-border text-sm focus:border-brand-500"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={isLoading}
            >
              {!model && <option value="">Loading...</option>}
              {sortedModels
                // Filter out non-code models if needed, though MODEL_ORDER already handles this
                .map((name) => {
                  const info = MODEL_DISPLAY_INFO[name];
                  // Use shared utility with current precision/framework state
                  const vram = getDynamicRam(name, precision, framework);
                  // Add warning if RAM is very high (e.g. > 32GB)
                  const isHighRam = parseInt(vram.replace('~', '').replace('GB', '')) > 32;

                  return (
                    <option key={name} value={name}>
                      {info ? `${isHighRam ? '⚠️ ' : ''}${info.label} (${vram})` : name}
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
            {(model === 'qwen-coder-32b' || model.includes('70b')) && (
              <div className="flex items-center gap-2 text-amber-400 text-xs mt-1">
                <AlertTriangle size={12} />
                <span>High RAM Usage Warning</span>
              </div>
            )}
          </div>
        </div>

        {/* Output Name */}
        <div className="space-y-2">
          <label className="label">
            Output Name (Optional)
          </label>
          <input
            type="text"
            className="w-full bg-primary border border-border rounded-lg p-2 text-sm focus:outline-none focus:border-brand-500"
            placeholder="Auto-generated if empty"
            value={outputName}
            onChange={(e) => setOutputName(e.target.value)}
          />
        </div>

        <ErrorAlert error={error} onDismiss={() => setError(null)} />

        <ValidationTooltip error={!prompt.trim() ? "Please enter instructions" : null} className="w-full mt-auto pt-4">
          <button
            className="w-full bg-gradient-to-r from-brand-600 to-cyan-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-primary font-bold py-3 rounded-lg shadow-lg shadow-brand-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all"
            onClick={handleGenerate}
            disabled={isLoading || !prompt.trim()}
          >
            {isLoading ? (<><Loader2 className="animate-spin" size={18} /> Generating...</>) : (<><Code size={18} /> Generate Code</>)}
          </button>
        </ValidationTooltip>
      </div>

      {/* Main Result Area */}
      <div ref={resultRef} className="flex-1 p-6 flex items-center justify-center bg-primary/30 scroll-mt-4">
        {result ? (
          <div className="flex flex-col items-center justify-center max-w-3xl w-full gap-6 h-full">
            <div className="w-full p-8 bg-primary/80 backdrop-blur-sm rounded-2xl border border-brand-500/30 shadow-2xl flex flex-col gap-6 relative overflow-hidden">
              <div className="flex items-center justify-between border-b border-border/50 pb-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-brand-500/20 flex items-center justify-center text-brand-400">
                    {isMultiFile ? <FolderArchive size={24} /> : <Code size={24} />}
                  </div>
                  <div className="overflow-hidden">
                    <h3 className="font-medium text-primary truncate max-w-md">{resultDisplayName}{isMultiFile && '.zip'}</h3>
                    {duration && <p className="text-xs text-tertiary">Generated in {formatDuration(duration * 1000)}</p>}
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
                      ? `${API_BASE_URL()}/api/files/zip?path=${encodeURIComponent(result)}`
                      : `${API_BASE_URL()}/api/files/${result}?download=true`
                    }
                    className="btn-secondary text-sm flex items-center gap-1"
                  >
                    <Download size={14} />
                    {isMultiFile ? 'Download ZIP' : 'Download'}
                  </a>
                </div>
              </div>

              {/* Multi-file project explanation */}
              {isMultiFile && (
                <div className="p-4 bg-secondary/50 rounded-lg border border-border/50">
                  <div className="flex items-start gap-3">
                    <FolderArchive size={20} className="text-brand-400 mt-0.5 shrink-0" />
                    <div className="flex-1">
                      <p className="text-sm text-secondary font-medium mb-1">
                        Multi-File Project ({generatedFiles.length} files)
                      </p>
                      <p className="text-xs text-tertiary">
                        Click "Preview Project" above to browse the file structure.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Reasoning / Thinking Block */}
              {reasoning && (
                <div className="p-4 bg-primary/50 rounded-lg border border-border/50 max-h-[40vh] overflow-y-auto">
                  <div className="flex items-center gap-2 mb-2 text-brand-400">
                    <span className="text-xs font-bold uppercase tracking-wider opacity-70">Reasoning Process</span>
                  </div>
                  <div className="text-xs text-secondary font-mono whitespace-pre-wrap italic leading-relaxed">
                    {reasoning}
                  </div>
                </div>
              )}

              {!reasoning && (
                <div className="text-center py-12 text-tertiary">
                  <p>Code generated successfully.</p>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="text-center text-tertiary">
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
