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
  title?: string;
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

export function JobProgressModal({ jobId, onClose, onViewResult, title }: JobProgressModalProps) {
  const { jobs, removeJob } = useAppStore();
  const [dots, setDots] = useState('');
  const [isCancelling, setIsCancelling] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  // Animated dots for "Loading..." text
  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');

      const job = jobs.find(j => j.job_id === jobId);
      if (job && job.status === 'generating' && job.generation_started_at) {
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
    } catch (err) {
      // Job may not exist on server yet (optimistic UI) - that's ok
      console.warn('Cancel request failed (job may not exist on server):', err);
    }
    // Always remove from store and close modal for responsive feel
    removeJob(jobId!);
    if (onClose) onClose();
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
              {title ? title : (
                isComplete ? 'Generation Complete' :
                  isFailed ? 'Generation Failed' :
                    isCancelled ? 'Generation Cancelled' :
                      (job.type === 'analysis' ? 'Analysis' : job.type) + ' Generation'
              )}
              {(() => {
                if (isComplete && job.generation_started_at && job.updated_at) {
                  const start = new Date(job.generation_started_at).getTime();
                  const end = new Date(job.updated_at).getTime();
                  const dur = Math.max(0, Math.round((end - start) / 1000));
                  return dur > 0 ? ` (${formatDuration(dur * 1000)})` : '';
                }
                return elapsed > 0 && job.status === 'generating' ? ` (${formatDuration(elapsed * 1000)})` : '';
              })()}
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
              {job.phase === 'loading' && percent === 0 ? `Loading ${getFriendlyModelName(job.model)}...` :
                job.phase === 'generating' || (job.phase === 'loading' && percent > 0) ? (
                  percent > 0 ? (job.type === 'video' ? `Rendering Video...` : `Generating...`) :
                    (job.message || 'Generating...')
                ) :
                  job.phase === 'complete' ? 'Complete!' :
                    job.phase === 'failed' ? 'Failed' :
                      'Processing...'}
            </p>
            {!['complete', 'failed'].includes(job.phase) && (
              <p className="text-xs text-secondary uppercase tracking-wider mt-1">
                {(job.phase === 'loading' && percent > 0) ? 'RENDERING' :
                  (job.phase === 'loading' ? `LOADING${dots}` : job.phase)}
              </p>
            )}
          </div>

          {/* Server Logs Panel - Like Chat */}
          {isOngoing && job.logs && job.logs.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-tertiary uppercase tracking-wider">Server Logs</span>
                <button
                  onClick={() => {
                    // Clear logs locally - they'll repopulate
                    if (job) job.logs = [];
                  }}
                  className="text-xs flex items-center gap-1 text-tertiary hover:text-primary transition-colors"
                  title="Clear logs"
                >
                  <Trash2 size={12} /> Clear
                </button>
              </div>
              <div className="bg-primary/50 p-3 rounded-lg font-mono text-xs text-secondary max-h-32 overflow-y-auto border border-border shadow-inner scrollbar-themed">
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
                {(job.type === 'translate' || job.type === 'code' || job.type === 'article' || job.type === 'analysis' || (job.type === 'upscale' && job.model !== 'ai') || (job.type === 'transform' && job.model === 'remove-bg') || (job.type === 'convert' && (job.params?.translate || (job.target_format !== 'mp3' && job.target_format !== 'wav' && job.target_format !== 'mp4' && job.target_format !== 'mov')))) ? (
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
                <span>{(job.type === 'translate' || job.type === 'code' || job.type === 'article' || job.type === 'analysis' || (job.type === 'upscale' && job.model !== 'ai') || (job.type === 'transform' && job.model === 'remove-bg') || (job.type === 'convert' && (job.params?.translate || (job.target_format !== 'mp3' && job.target_format !== 'wav' && job.target_format !== 'mp4' && job.target_format !== 'mov')))) ? (job.phase === 'queued' ? 'Queued' : 'Processing...') : `${percent}%`}</span>
                {percent < 100 && (
                  <span>
                    {/* Extract remaining time from message like "Generating: 50%, Remaining Time: 00:15" */}
                    {job.message?.match(/Remaining Time:\s*(\d+:\d+)/)?.[1]
                      ? `Remaining: ${job.message.match(/Remaining Time:\s*(\d+:\d+)/)?.[1]}`
                      : 'Please wait...'}
                  </span>
                )}
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
              className="btn-secondary w-full flex items-center justify-center gap-2 text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-slate-800 border-red-200 dark:border-red-500/30"
            >
              <StopCircle size={18} />
              {isCancelling ? 'Cancelling...' : 'Cancel Job'}
            </button>
          )}

        </div>

        {/* Footer / Resource Monitor */}
        {!isComplete && !isFailed && (
          <div className="bg-primary/30 p-4 border-t border-border">
            <ResourceStats variant="modal" />
          </div>
        )}
      </div>
    </div>
  );
}
