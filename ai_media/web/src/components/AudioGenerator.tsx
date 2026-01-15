
import { RandomPrompt } from './common/RandomPrompt';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { ErrorAlert } from './common/ErrorAlert';
import { ModelHelpLink } from './common/ModelHelpLink';
import { formatDuration } from '../utils/formatTime';

// Display info matching CLI exactly
const MODEL_DISPLAY_INFO: Record<string, { label: string; vram: string }> = {
  // Music / Audio
  'musicgen-small': { label: 'MusicGen Small (Fast, Good for Music)', vram: '~4GB' },
  'musicgen-medium': { label: 'MusicGen Medium (Better Quality)', vram: '~8GB' },
  'musicgen-large': { label: 'MusicGen Large (Best Quality)', vram: '~14GB' },
  'audioldm2': { label: 'AudioLDM2 (Sound Effects & Audio)', vram: '~6GB' },
  'stable-audio': { label: 'Stable Audio (High Quality, 🔒 Gated)', vram: '~12GB' },

  // TTS
  'bark': { label: 'Bark (Expressive TTS & FX)', vram: '~12GB' },
  'speecht5': { label: 'SpeechT5 (Efficient Baseline)', vram: '~2GB' },
};

const MUSIC_MODELS = [
  'musicgen-small', 'musicgen-medium', 'musicgen-large',
  'audioldm2', 'stable-audio'
];

const TTS_MODELS = [
  'bark', 'speecht5'
];

import { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../store';
import { generateAudio, useModels } from '../hooks/useApi';
import { API_BASE_URL } from '../config';
import { FileAudio, Loader2, AlertTriangle, AlertCircle, Mic, Music } from 'lucide-react';
import { NumberInput } from './common/NumberInput';
import { Tooltip } from './common/Tooltip';
import { ValidationTooltip } from './common/ValidationTooltip';

type Tab = 'music' | 'tts';

export function AudioGenerator() {
  const { addJob } = useAppStore();
  const [activeTab, setActiveTab] = useState<Tab>('music');

  // Form State
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState('musicgen-medium');
  const [duration, setDuration] = useState(10);
  const [samplingRate, setSamplingRate] = useState('32000');
  const [format, setFormat] = useState('mp3');

  // Job State
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [genDuration, setGenDuration] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  // Global models cache
  const { models } = useModels();
  const availableModels = models?.audio || [];

  // Reset defaults on tab change
  useEffect(() => {
    if (activeTab === 'music') {
      setModel('musicgen-medium');
      setDuration(10);
      setPrompt('');
    } else {
      setModel('bark');
      setDuration(0); // Auto for TTS
      setPrompt('');
    }
    setError(null);
  }, [activeTab]);

  // Update defaults when model changes within tab
  useEffect(() => {
    if (activeTab === 'music') {
      if (model === 'stable-audio') setDuration(30);
      else setDuration(10);
    } else {
      setDuration(0);
    }
  }, [model, activeTab]);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    // Construct final prompt based on model rules
    let finalPrompt = prompt;

    setIsLoading(true);
    setResult(null);
    setGenDuration(null);
    setError(null);

    try {
      const response = await generateAudio({
        prompt: finalPrompt,
        model,
        duration,
        sampling_rate: samplingRate,
        format
      });
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

    let hasSeenJob = false;

    const unsubscribe = useAppStore.subscribe((state) => {
      const updatedJob = state.jobs.find(j => j.job_id === currentJobId);
      if (updatedJob) {
        hasSeenJob = true;

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
      } else if (hasSeenJob) {
        // Job was in store but now removed (e.g., cancelled)
        setIsLoading(false);
        setCurrentJobId(null);
      }
    });
    return () => unsubscribe();
  }, [currentJobId]);

  // Filter models based on tab
  const currentModelList = activeTab === 'music' ? MUSIC_MODELS : TTS_MODELS;
  const sortedModels = currentModelList.filter(name =>
    availableModels.some(m => m.name === name) || true // Fallback to allow unlisted models for now if cache missing
  );

  // Fallback if list empty (dev mode)
  const displayModels = sortedModels.length > 0 ? sortedModels : currentModelList;

  const resultRef = useRef<HTMLDivElement>(null);

  const handleViewResult = () => {
    setTimeout(() => {
      resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  };

  const insertToken = (token: string) => {
    setPrompt(prev => prev + (prev.endsWith(' ') ? '' : ' ') + token);
  };

  return (
    <div className="flex flex-col lg:flex-row h-full bg-primary text-primary">
      {/* Parameters Sidebar */}
      <div className="w-full lg:w-[500px] border-b lg:border-b-0 lg:border-r border-border p-4 lg:py-6 lg:pr-[27px] lg:pl-1 flex flex-col gap-6 overflow-y-auto shrink-0 h-auto lg:h-full">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
            <FileAudio className="text-brand-400" /> Audio Generator
          </h2>
          <p className="text-xs text-tertiary">Create music, sound effects, or realistic speech</p>
        </div>

        {/* Tabs */}
        <div className="bg-primary/50 p-1 rounded-lg flex border border-border">
          <button
            onClick={() => setActiveTab('music')}
            className={`flex-1 flex items-center justify-center gap-2 py-2 text-sm font-medium rounded-md transition-all ${activeTab === 'music'
              ? 'bg-brand-600 text-white shadow-md'
              : 'text-tertiary hover:text-white hover:bg-white/5'
              }`}
          >
            <Music size={16} /> Audio / Music
          </button>
          <button
            onClick={() => setActiveTab('tts')}
            className={`flex-1 flex items-center justify-center gap-2 py-2 text-sm font-medium rounded-md transition-all ${activeTab === 'tts'
              ? 'bg-brand-600 text-white shadow-md'
              : 'text-tertiary hover:text-white hover:bg-white/5'
              }`}
          >
            <Mic size={16} /> Text-to-Speech
          </button>
        </div>

        {/* Model Selector */}
        <div className="space-y-2">
          <label className="label flex items-center">
            Model
            <ModelHelpLink section="audio" />
          </label>
          <select
            className="select w-full bg-primary border-border text-sm focus:border-brand-500"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            {displayModels.map((name) => {
              const info = MODEL_DISPLAY_INFO[name];
              return (
                <option key={name} value={name}>
                  {info ? `${info.label} ${info.vram}` : name}
                </option>
              );
            })}
          </select>
          {/* Warnings / Metadata */}
          {model === 'stable-audio' && (
            <div className="flex items-center gap-2 text-amber-600 dark:text-amber-400 text-xs mt-1">
              <AlertTriangle size={12} />
              <span>Requires HF Login (Accepted Terms)</span>
            </div>
          )}
          {model === 'bark' && (
            <div className="flex items-center gap-2 text-blue-400 text-xs mt-1">
              <AlertCircle size={12} />
              <span>Supports [laugh], [sighs], [music] tags</span>
            </div>
          )}
        </div>


        {/* Prompt */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="label">
              {activeTab === 'tts' ? "Text to Speak" : "Prompt"}
            </label>
            {/* @ts-ignore - PROMPTS keys might check tts existence */}
            <RandomPrompt type={activeTab === 'tts' ? "tts" : "audio"} onPromptSelect={setPrompt} stripBarkTokens={activeTab === 'tts' && model !== 'bark'} />
          </div>
          <textarea
            className="w-full bg-primary border border-border rounded-lg p-3 text-sm focus:outline-none focus:border-brand-500 resize-y min-h-[120px]"
            placeholder={activeTab === 'tts' ? "Enter text you want the model to say..." : "Describe the music or sound effect..."}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />

          {/* Bark Quick Actions */}
          {model === 'bark' && (
            <div className="flex flex-wrap gap-2 pt-1">
              <button onClick={() => insertToken('[laugh]')} className="text-[10px] bg-white/10 hover:bg-brand-500/50 px-2 py-1 rounded transition-colors">[laugh]</button>
              <button onClick={() => insertToken('[sighs]')} className="text-[10px] bg-white/10 hover:bg-brand-500/50 px-2 py-1 rounded transition-colors">[sighs]</button>
              <button onClick={() => insertToken('[music]')} className="text-[10px] bg-white/10 hover:bg-brand-500/50 px-2 py-1 rounded transition-colors">[music]</button>
              <button onClick={() => insertToken('[gasps]')} className="text-[10px] bg-white/10 hover:bg-brand-500/50 px-2 py-1 rounded transition-colors">[gasps]</button>
              <button onClick={() => insertToken('♪ text ♪')} className="text-[10px] bg-white/10 hover:bg-brand-500/50 px-2 py-1 rounded transition-colors">♪ lyrics ♪</button>
            </div>
          )}
        </div>

        {/* Duration (Music Only) */}
        {activeTab === 'music' && (
          <div className="space-y-2">
            <label className="label flex items-center gap-1">
              Duration (seconds)
              <Tooltip content="Max duration varies by model (usually 30-60s)." align="left" />
            </label>
            <NumberInput
              value={duration}
              onChange={setDuration}
              min={1}
              max={120}
              placeholder={'10'}
            />
          </div>
        )}

        {/* Output Format */}
        <div className="space-y-2">
          <label className="label">Output Format</label>
          <select
            className="select w-full bg-primary border-border text-sm focus:border-brand-500"
            value={format}
            onChange={(e) => setFormat(e.target.value)}
            disabled={isLoading}
          >
            <option value="mp3">MP3 (Compressed, Universal)</option>
            <option value="wav">WAV (Lossless)</option>
            <option value="flac">FLAC (Lossless, Compressed)</option>
            <option value="ogg">OGG (Open, Lossy)</option>
            <option value="m4a">M4A/AAC (Apple, Lossy)</option>
            <option value="opus">OPUS (Modern, Efficient)</option>
            <option value="wma">WMA (Windows)</option>
            <option value="aiff">AIFF (Apple Lossless)</option>
          </select>
        </div>

        {/* Sampling Rate */}
        <div className="space-y-2">
          <label className="label">Sampling Rate</label>
          <select
            className="select w-full bg-primary border-border text-sm focus:border-brand-500"
            value={samplingRate}
            onChange={(e) => setSamplingRate(e.target.value)}
            disabled={isLoading}
          >
            <option value="16000">16000 Hz (Standard TTS)</option>
            <option value="24000">24000 Hz (Bark Standard)</option>
            <option value="32000">32000 Hz (Default)</option>
            <option value="44100">44100 Hz (High Quality)</option>
            <option value="48000">48000 Hz (Professional)</option>
          </select>
        </div>

        <ErrorAlert error={error} onDismiss={() => setError(null)} />

        <ValidationTooltip error={!prompt.trim() ? "Please enter a prompt" : null} className="w-full mt-auto pt-4">
          <button
            className="w-full bg-gradient-to-r from-brand-600 to-purple-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-primary font-bold py-3 rounded-lg shadow-lg shadow-brand-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all"
            onClick={handleGenerate}
            disabled={isLoading || !prompt.trim()}
          >
            {isLoading ? (<><Loader2 className="animate-spin" size={18} /> Generating...</>) : (<>{activeTab === 'tts' ? <Mic size={18} /> : <FileAudio size={18} />} Generate</>)}
          </button>
        </ValidationTooltip>
      </div>

      {/* Main Preview */}
      <div ref={resultRef} className="flex-1 p-6 flex items-center justify-center bg-primary/30 min-h-[500px] lg:min-h-0 scroll-mt-4">
        {result ? (
          <div className="flex flex-col items-center justify-center max-w-2xl w-full gap-6">
            <div className="w-full p-8 bg-primary/80 backdrop-blur-sm rounded-2xl border border-brand-500/30 shadow-2xl flex flex-col items-center gap-6">
              <div className="w-24 h-24 rounded-full bg-brand-500/10 flex items-center justify-center text-brand-400 animate-pulse">
                {activeTab === 'tts' ? <Mic size={48} /> : <FileAudio size={48} />}
              </div>
              <audio src={`${API_BASE_URL()}/api/files/${result}`} controls className="w-full" />
              <div className="flex flex-col items-center">
                <p className="text-sm text-secondary">Generated Audio Result</p>
                {genDuration && <p className="text-xs text-tertiary mt-1">Generated in {formatDuration(genDuration * 1000)}</p>}
              </div>
            </div>

            <div className="flex gap-2">
              <a href={`${API_BASE_URL()}/api/files/${result}?download=true`} className="btn-secondary text-sm">Download</a>
            </div>
          </div>
        ) : (
          <div className="text-center text-tertiary">
            {activeTab === 'tts' ? (
              <Mic size={48} className="mx-auto mb-4 opacity-20" />
            ) : (
              <FileAudio size={48} className="mx-auto mb-4 opacity-20" />
            )}
            <h3 className="text-lg font-medium mb-2">{activeTab === 'tts' ? 'Speech Synthesis' : 'Audio Composition'}</h3>
            <p className="max-w-sm mx-auto">
              {activeTab === 'tts'
                ? "Enter text and select a voice style to generate realistic speech."
                : "Describe the soundscape or music you want to create."}
            </p>
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
