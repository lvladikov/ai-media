
import { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../store';
import { generateArticle, useModels } from '../hooks/useApi';
import { API_BASE_URL } from '../config';
import { FileText, Loader2, Globe, AlertTriangle, HelpCircle } from 'lucide-react';
import { getDynamicRam } from '../utils/modelResources';
import { TranslateOptions } from './common/TranslateOptions';

import { ValidationTooltip } from './common/ValidationTooltip';
import { RandomPrompt } from './common/RandomPrompt';
import { NumberInput } from './common/NumberInput';
import { ErrorAlert } from './common/ErrorAlert';


import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { ModelHelpLink } from './common/ModelHelpLink';
import { formatDuration } from '../utils/formatTime';

// Display names matching CLI
// Display names matching CLI (vram removed, calculated dynamically)
// Display names matching CLI (vram removed, calculated dynamically)
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
  'qwen3-coder-30b': { label: 'Qwen3 Coder 30B (MoE, 3.3B active)' },
  'qwen-coder-32b': { label: 'Qwen 2.5 Coder 32B' },
};

const MODEL_ORDER = [
  'deepseek-r1-qwen-7b', 'deepseek-r1-qwen-14b', 'deepseek-r1-qwen-32b',
  'deepseek-r1-llama-8b', 'deepseek-r1-llama-70b',
  'llama-3.1-8b', 'mistral-nemo-12b', 'qwen3-14b', 'qwen3-8b',
  'qwen3-opus-4.5-14b', 'qwen3-opus-4.5-8b', 'qwen3-gpt-5.2-14b', 'qwen3-gpt-5.2-8b',
  'qwen3-coder-30b', 'qwen-coder-32b'
];



export function ArticleGenerator() {
  const { addJob } = useAppStore();
  const [topic, setTopic] = useState('');
  const [model, setModel] = useState('');
  const [precision, setPrecision] = useState('auto');
  const [framework, setFramework] = useState(navigator.userAgent.toLowerCase().includes('mac') ? 'mlx' : 'auto');
  const [format, setFormat] = useState('md');
  const [length, setLength] = useState("quick");
  const [online, setOnline] = useState(false);
  const [researchIterations, setResearchIterations] = useState(3);
  const [maxImages, setMaxImages] = useState(5);
  const [translate, setTranslate] = useState(false);
  const [targetLanguage, setTargetLanguage] = useState('eng_Latn');
  const [translationModel, setTranslationModel] = useState('nllb-200-3.3b');
  const [filename, setFilename] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [reasoning, setReasoning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

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
        translate: translate,
        target_language: targetLanguage,
        // @ts-ignore - API pending update
        translation_model: translationModel,
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


  // Watch for job completion to show result inline after modal closes
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
            <FileText className="text-brand-400" /> Article Gen
          </h2>
          <p className="text-xs text-tertiary">Generate articles, blogs and essays</p>
        </div>

        {/* Topic */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="label">Topic</label>
            <RandomPrompt type="article" onPromptSelect={setTopic} />
          </div>
          <textarea
            className="w-full bg-primary border border-border rounded-lg p-3 text-sm focus:outline-none focus:border-brand-500 resize-y min-h-[120px]"
            placeholder="Enter your article topic or use the Random Prompt tool..."
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
          />
        </div>

        {/* Filename */}
        <div className="space-y-2">
          <label className="label">
            Filename (Optional)
          </label>
          <input
            type="text"
            className="w-full bg-primary border border-border rounded-lg p-2 text-sm focus:outline-none focus:border-brand-500"
            placeholder="Auto-generated if empty"
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
          />
        </div>

        {/* Model Selector */}
        {/* Model Selector Section */}
        <div className="space-y-4">

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
              <ModelHelpLink section="article" />
            </label>
            <select
              className="select w-full bg-primary border-border text-sm focus:border-brand-500"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={isLoading}
            >
              {!model && <option value="">Loading...</option>}
              {sortedModels.map((name) => {
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
          </div>
        </div>

        {/* Format & Length */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="label">Format</label>
            <select
              className="select w-full bg-primary border-border text-sm focus:border-brand-500"
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
            <label className="label">Length</label>
            <select
              className="select w-full bg-primary border-border text-sm focus:border-brand-500"
              value={length}
              onChange={(e) => setLength(e.target.value)}
            >
              <option value="quick">Quick (~500 words)</option>
              <option value="standard">Standard (~1500 words)</option>
              <option value="detailed">Detailed (~3000 words)</option>
              <option value="exhaustive">Exhaustive (~10000 words)</option>
            </select>
          </div>
        </div>

        {/* Online Research */}
        <div className="space-y-3 pt-2 border-t border-border">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              className="checkbox checkbox-xs checkbox-primary"
              checked={online}
              onChange={(e) => setOnline(e.target.checked)}
            />
            <span className="text-sm font-medium text-secondary flex items-center gap-2">
              <Globe size={14} className="text-brand-400" />
              Online Research
            </span>
          </label>

          {online && (
            <div className="grid grid-cols-2 gap-4 pl-6">
              <div>
                <label className="text-xs text-secondary">Sources</label>
                <NumberInput
                  value={researchIterations}
                  onChange={setResearchIterations}
                  min={1}
                  max={10}
                  className="w-full h-8 text-sm input bg-primary border-border"
                />
              </div>
              <div>
                <label className="text-xs text-secondary">Max Images</label>
                <NumberInput
                  value={maxImages}
                  onChange={setMaxImages}
                  min={0}
                  max={20}
                  className="w-full h-8 text-sm input bg-primary border-border"
                />
              </div>
            </div>
          )}
        </div>

        {/* Translation */}
        <div className="pt-2 border-t border-border">
          <TranslateOptions
            enabled={translate}
            onEnabledChange={setTranslate}
            selectedModel={translationModel}
            onModelChange={setTranslationModel}
            targetLanguage={targetLanguage}
            onLanguageChange={setTargetLanguage}
            title="Translate Output"
            hideLanguageSelector={false}
            hideModelSelector={false}
            infoMessage="Choose NLLB for speed and broad language coverage. Use LLM models for more natural, context-aware translations - especially valuable for professional or creative content."
          />
        </div>

        <ErrorAlert error={error} onDismiss={() => setError(null)} />

        <ValidationTooltip error={!topic.trim() ? "Please enter a topic" : null} className="w-full mt-auto pt-4">
          <button
            className="w-full bg-gradient-to-r from-brand-600 to-cyan-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-primary font-bold py-3 rounded-lg shadow-lg shadow-brand-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all"
            onClick={handleGenerate}
            disabled={isLoading || !topic.trim()}
          >
            {isLoading ? (<><Loader2 className="animate-spin" size={18} /> Generating...</>) : (<><FileText size={18} /> Generate Article</>)}
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
                    <FileText size={24} />
                  </div>
                  <div>
                    <h3 className="font-medium text-primary">{result?.split(/[/\\]/).pop()}</h3>
                    {duration && <p className="text-xs text-tertiary">Generated in {formatDuration(duration * 1000)}</p>}
                  </div>
                </div>
                <div className="flex gap-2">
                  <button className="btn-secondary text-sm" onClick={() => setIsPreviewOpen(true)}>Preview</button>
                  <a href={`${API_BASE_URL()}/api/files/${result}?download=true`} className="btn-secondary text-sm">Download</a>
                </div>
              </div>

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
                  <p>Article generated successfully.</p>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="text-center text-tertiary">
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
          fileName={result?.split(/[/\\]/).pop() || 'article.md'}
        />
      )}
    </div>
  );
}
