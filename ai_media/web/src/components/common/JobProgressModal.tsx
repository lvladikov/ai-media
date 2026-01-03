import { useEffect, useState } from 'react';
import { useAppStore } from '../../store';
import { Loader2, X, CheckCircle, AlertOctagon, StopCircle, XCircle, Trash2 } from 'lucide-react';
import { ResourceStats } from './ResourceStats';
import { cancelJob } from '../../hooks/useApi';
import { formatDuration } from '../../utils/formatTime';

interface JobProgressModalProps {
  jobId: string | null;
  onClose?: () => void;
  onViewResult?: () => void;
}


const getFriendlyModelName = (modelName: string | undefined | null) => {
  if (!modelName) return 'Model';

  // Map common HF IDs to friendly names
  if (modelName.includes('Meta-Llama-3.1-8B')) return 'Llama 3.1 8B (Fast & Stable)';
  if (modelName.includes('Mistral-Nemo-Instruct')) return 'Mistral Nemo 12B';
  if (modelName.includes('Qwen2.5-14B')) return 'Qwen 2.5 14B Instruct';
  if (modelName.includes('DeepSeek-R1-Distill-Llama-8B')) return 'DeepSeek R1 Llama 8B';
  if (modelName.includes('DeepSeek-R1-Distill-Qwen-7B')) return 'DeepSeek R1 7B';
  if (modelName.includes('DeepSeek-R1-Distill-Qwen-14B')) return 'DeepSeek R1 14B';
  if (modelName.includes('DeepSeek-R1-Distill-Qwen-32B')) return 'DeepSeek R1 32B';

  // Fallback: Use clean name or original
  if (modelName.includes('/')) return modelName.split('/')[1];
  return modelName;
};

export function JobProgressModal({ jobId, onClose, onViewResult }: JobProgressModalProps) {
  const { jobs, removeJob } = useAppStore();
  const [dots, setDots] = useState('');
  const [isCancelling, setIsCancelling] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  // Animated dots for "Loading..." text
  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');

      const job = jobs.find(j => j.job_id === jobId);
      if (job && job.phase === 'generating' && job.generation_started_at) {
        const startTime = new Date(job.generation_started_at).getTime();
        const now = new Date().getTime();
        setElapsed(Math.max(0, Math.round((now - startTime) / 1000)));
      } else {
        setElapsed(0);
      }
    }, 1000); // 1s interval for timer
    return () => clearInterval(interval);
  }, [jobs, jobId]);

  if (!jobId) return null;

  const job = jobs.find(j => j.job_id === jobId);

  // If job not found in store yet (race condition), just show loading
  if (!job) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
        <div className="bg-secondary border border-border p-6 rounded-xl shadow-2xl max-w-md w-full flex flex-col items-center gap-4">
          <Loader2 className="animate-spin text-brand-400" size={32} />
          <p className="text-secondary">Initializing job{dots}</p>
        </div>
      </div>
    );
  }

  const isComplete = job.status === 'complete';
  const isFailed = job.status === 'failed';
  const isCancelled = job.status === 'cancelled';
  const isOngoing = !isComplete && !isFailed && !isCancelled;

  const handleCancel = async () => {
    if (isCancelling) return;
    setIsCancelling(true);
    try {
      await cancelJob(jobId);
      // Immediately remove from store and close modal for responsive feel
      removeJob(jobId!);
      if (onClose) onClose();
    } catch (err) {
      console.error('Failed to cancel job:', err);
      setIsCancelling(false);
    }
  };

  // Format progress percentage
  const percent = Math.round(job.progress || 0);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-secondary border border-border rounded-xl shadow-2xl max-w-3xl w-full overflow-hidden animate-in fade-in zoom-in-95 duration-200">

        {/* Header */}
        <div className="bg-primary/50 p-4 border-b border-border flex justify-between items-center">
          <h3 className="font-semibold text-lg flex items-center gap-2">
            {isComplete ? <CheckCircle className="text-green-400" size={20} /> :
              isFailed ? <AlertOctagon className="text-red-400" size={20} /> :
                isCancelled ? <XCircle className="text-yellow-400" size={20} /> :
                  <Loader2 className="animate-spin text-brand-400" size={20} />}

            <span className="capitalize text-primary">
              {isComplete ? 'Generation Complete' :
                isFailed ? 'Generation Failed' :
                  isCancelled ? 'Generation Cancelled' :
                    job.type + ' Generation'}{elapsed > 0 && job.phase === 'generating' ? ` (${formatDuration(elapsed * 1000)})` : ''}
            </span>
          </h3>
          {/* Only allow closing if complete/failed/cancelled, or if user explicitly wants to background it */}
          {(isComplete || isFailed || isCancelled) && onClose && (
            <button onClick={onClose} className="hover:text-primary text-secondary transition-colors">
              <X size={20} />
            </button>
          )}
        </div>

        {/* Body */}
        <div className="p-4 md:p-6 space-y-4">

          {/* Status Header - Compact */}
          <div className="text-center">
            <p className="text-lg font-medium text-primary">
              {job.phase === 'loading' ? `Loading ${getFriendlyModelName(job.model)}...` :
                job.phase === 'generating' ? (job.message || 'Generating...') :
                  job.phase === 'complete' ? 'Complete!' :
                    job.phase === 'failed' ? 'Failed' :
                      'Processing...'}
            </p>
            <p className="text-xs text-secondary uppercase tracking-wider mt-1">
              {job.phase}
            </p>
          </div>

          {/* Server Logs Panel - Like Chat */}
          {isOngoing && job.logs && job.logs.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-slate-500 uppercase tracking-wider">Server Logs</span>
                <button
                  onClick={() => {
                    // Clear logs locally - they'll repopulate
                    if (job) job.logs = [];
                  }}
                  className="text-xs flex items-center gap-1 text-slate-500 hover:text-slate-300 transition-colors"
                  title="Clear logs"
                >
                  <Trash2 size={12} /> Clear
                </button>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-lg font-mono text-xs text-slate-400 max-h-32 overflow-y-auto border border-slate-700/50 shadow-inner scrollbar-themed">
                {job.logs.map((log: string, i: number) => (
                  <div key={i} className="whitespace-pre-wrap">{log}</div>
                ))}
              </div>
            </div>
          )}

          {/* Progress Bar */}
          {!isComplete && !isFailed && (
            <div className="space-y-2">
              <div className="h-2 w-full bg-tertiary rounded-full overflow-hidden relative">
                {(job.type === 'code' || job.type === 'article' || (job.type === 'upscale' && job.model !== 'ai') || (job.type === 'transform' && job.model === 'remove-bg')) ? (
                  <div className="h-full bg-brand-500 w-1/3 absolute rounded-full animate-progress-bounce" />
                ) : (
                  <div
                    className="h-full bg-brand-500 transition-all duration-300 ease-out relative"
                    style={{ width: `${percent}%` }}
                  >
                    <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
                  </div>
                )}
              </div>
              <div className="flex justify-between text-xs text-secondary">
                <span>{(job.type === 'code' || job.type === 'article' || (job.type === 'upscale' && job.model !== 'ai') || (job.type === 'transform' && job.model === 'remove-bg')) ? (job.phase === 'queued' ? 'Queued' : 'Processing...') : `${percent}%`}</span>
                {percent < 100 && <span>Please wait...</span>}
              </div>
            </div>
          )}

          {/* Error Message */}
          {isFailed && job.error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-200 p-4 rounded-lg text-sm font-mono break-words">
              {job.error}
            </div>
          )}

          {/* Success Actions */}
          {isComplete && (
            <button
              onClick={() => {
                if (onViewResult) onViewResult();
                if (onClose) onClose();
              }}
              className="btn-primary w-full"
            >
              View Result
            </button>
          )}

          {/* Failed Actions */}
          {isFailed && (
            <button onClick={onClose} className="btn-secondary w-full">
              Close
            </button>
          )}

          {/* Cancelled Actions */}
          {isCancelled && (
            <button onClick={onClose} className="btn-secondary w-full">
              Close
            </button>
          )}

          {/* Cancel Button for Ongoing Jobs */}
          {isOngoing && (
            <button
              onClick={handleCancel}
              disabled={isCancelling}
              className="btn-secondary w-full flex items-center justify-center gap-2 text-red-400 hover:text-red-300 border-red-500/30 hover:border-red-500/50"
            >
              <StopCircle size={18} />
              {isCancelling ? 'Cancelling...' : 'Cancel Job'}
            </button>
          )}

        </div>

        {/* Footer / Resource Monitor */}
        {!isComplete && !isFailed && (
          <div className="bg-primary/30 p-4 border-t border-border">
            <ResourceStats />
          </div>
        )}
      </div>
    </div>
  );
}
