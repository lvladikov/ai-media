import { useEffect, useState } from 'react';
import { useAppStore } from '../../store';
import { Loader2, X, CheckCircle, AlertOctagon, StopCircle, XCircle } from 'lucide-react';
import { ResourceStats } from './ResourceStats';
import { cancelJob } from '../../hooks/useApi';

interface JobProgressModalProps {
  jobId: string | null;
  onClose?: () => void;
}

export function JobProgressModal({ jobId, onClose }: JobProgressModalProps) {
  const { jobs } = useAppStore();
  const [dots, setDots] = useState('');
  const [isCancelling, setIsCancelling] = useState(false);

  // Animated dots for "Loading..." text
  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 500);
    return () => clearInterval(interval);
  }, []);

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
      // The job will be updated via WebSocket, onClose will be called when status turns to cancelled
    } catch (err) {
      console.error('Failed to cancel job:', err);
      setIsCancelling(false);
    }
  };

  // Format progress percentage
  const percent = Math.round(job.progress || 0);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-secondary border border-border rounded-xl shadow-2xl max-w-lg w-full overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        
        {/* Header */}
        <div className="bg-primary/50 p-4 border-b border-border flex justify-between items-center">
          <h3 className="font-semibold text-lg flex items-center gap-2">
            {isComplete ? <CheckCircle className="text-green-400" size={20}/> : 
             isFailed ? <AlertOctagon className="text-red-400" size={20}/> :
             isCancelled ? <XCircle className="text-yellow-400" size={20}/> :
             <Loader2 className="animate-spin text-brand-400" size={20}/>}
            
            <span className="capitalize text-primary">
              {isComplete ? 'Generation Complete' : 
               isFailed ? 'Generation Failed' : 
               isCancelled ? 'Generation Cancelled' :
               job.type + ' Generation'}
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
        <div className="p-6 space-y-6">
          
          {/* Status Message */}
          <div className="text-center space-y-2">
             <p className="text-lg font-medium text-primary">
                {job.message || 'Processing...'}
             </p>
             <p className="text-sm text-secondary uppercase tracking-wider text-xs">
                {job.phase}
             </p>
          </div>

          {/* Progress Bar */}
          {!isComplete && !isFailed && (
            <div className="space-y-2">
              <div className="h-2 w-full bg-tertiary rounded-full overflow-hidden">
                <div 
                  className="h-full bg-brand-500 transition-all duration-300 ease-out relative"
                  style={{ width: `${percent}%` }}
                >
                    <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
                </div>
              </div>
              <div className="flex justify-between text-xs text-secondary">
                 <span>{percent}%</span>
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
             <button onClick={onClose} className="btn-primary w-full">
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
