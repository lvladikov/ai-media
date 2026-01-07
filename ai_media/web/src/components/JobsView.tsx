import { useState } from 'react';
import { useAppStore } from '../store';
import type { Job } from '../store';
import { History, Clock, CheckCircle, XCircle, Loader2, Eye, StopCircle } from 'lucide-react';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { AnalysisPreviewModal } from './common/AnalysisPreviewModal';
import { ComparisonPreviewModal } from './common/ComparisonPreviewModal';
import { cancelJob } from '../hooks/useApi';

function JobCard({ job, onOpen }: { job: Job; onOpen: (job: Job) => void }) {
  const statusColors = {
    pending: 'text-yellow-400',
    loading: 'text-blue-400',
    generating: 'text-purple-400',
    complete: 'text-green-700 dark:text-green-400',
    failed: 'text-red-400',
    cancelled: 'text-secondary',
  };

  const statusIcons = {
    pending: <Clock size={16} />,
    loading: <Loader2 size={16} className="animate-spin" />,
    generating: <Loader2 size={16} className="animate-spin" />,
    complete: <CheckCircle size={16} />,
    failed: <XCircle size={16} />,
    cancelled: <XCircle size={16} />,
  };

  // Format params for display
  const formatParams = (params: Record<string, string | number | boolean> | undefined) => {
    if (!params || Object.keys(params).length === 0) return null;
    return Object.entries(params)
      .filter(([key]) => key !== 'input_path') // Skip file paths
      .map(([key, value]) => {
        const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        return `${label}: ${value}`;
      })
      .join(' • ');
  };

  const paramsStr = formatParams(job.params);

  return (
    <div className="card">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className={statusColors[job.status]}>{statusIcons[job.status]}</span>
            <span className="font-medium capitalize">{job.type}</span>
            {job.model && (
              <span className="text-xs px-2 py-0.5 rounded bg-tertiary/50 text-secondary">
                {job.model}
              </span>
            )}
            <span className={`text-xs px-2 py-0.5 rounded ${statusColors[job.status]} bg-tertiary/50`}>
              {job.status}
            </span>
          </div>

          {/* Prompt */}
          {job.prompt && (
            <p className="text-sm text-secondary mt-2 line-clamp-2" title={job.prompt}>
              "{job.prompt}"
            </p>
          )}

          {/* Params */}
          {paramsStr && (
            <p className="text-xs text-tertiary mt-1">
              {paramsStr}
            </p>
          )}

          <p className="text-sm text-secondary mt-1">{job.message}</p>
          <p className="text-xs text-tertiary mt-1">
            {new Date(job.created_at).toLocaleString()}
          </p>
        </div>

        <div className="flex gap-2 flex-shrink-0">
          {(job.status === 'complete' || job.status === 'loading' || job.status === 'generating' || job.status === 'pending') && (
            <button
              onClick={() => onOpen(job)}
              className="btn-secondary text-xs flex items-center gap-1"
            >
              <Eye size={14} />
              {job.status === 'complete' ? 'Open' : 'View Progress'}
            </button>
          )}
          {(job.status === 'loading' || job.status === 'generating' || job.status === 'pending') && (
            <button
              onClick={async (e) => {
                e.stopPropagation();
                try {
                  await cancelJob(job.job_id);
                } catch (err) {
                  console.error('Failed to cancel job:', err);
                }
              }}
              className="btn-secondary text-xs flex items-center gap-1 text-red-400 hover:text-red-300 border-red-500/30 hover:border-red-500/50"
            >
              <StopCircle size={14} />
              Cancel
            </button>
          )}
        </div>
      </div>

      {(job.status === 'loading' || job.status === 'generating') && (
        <div className="mt-3">
          <div className="h-2 bg-tertiary rounded-full overflow-hidden">
            <div
              className="h-full bg-primary-500 progress-animated transition-all duration-300"
              style={{ width: `${job.progress}%` }}
            />
          </div>
          <p className="text-xs text-tertiary mt-1">{job.progress}% - {job.phase}</p>
        </div>
      )}

      {job.error && (
        <p className="text-sm text-red-400 mt-2">{job.error}</p>
      )}
    </div>
  );
}

export function JobsView() {
  const { jobs } = useAppStore();
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [viewType, setViewType] = useState<'progress' | 'preview' | null>(null);

  const handleOpen = (job: Job) => {
    setSelectedJob(job);
    if (job.status === 'complete') {
      setViewType('preview');
    } else {
      setViewType('progress');
    }
  };

  const handleClose = () => {
    setSelectedJob(null);
    setViewType(null);
  };

  return (
    <div className="max-w-4xl pt-6">
      <h1 className="text-2xl font-bold mb-6 flex items-center gap-2">
        <History className="text-primary-400" />
        Job History
      </h1>

      {jobs.length === 0 ? (
        <div className="card text-center text-secondary py-12">
          <History size={48} className="mx-auto mb-4 opacity-50" />
          <p>No jobs yet. Start generating to see your job history here.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {[...jobs].reverse().map((job) => (
            <JobCard key={job.job_id} job={job} onOpen={handleOpen} />
          ))}
        </div>
      )}

      {selectedJob && viewType === 'progress' && (
        <JobProgressModal
          jobId={selectedJob.job_id}
          onClose={handleClose}
        />
      )}

      {selectedJob && viewType === 'preview' && selectedJob.result_path && (
        <>
          {/* Analysis jobs use AnalysisPreviewModal */}
          {selectedJob.type === 'analysis' && (
            <AnalysisPreviewModal
              isOpen={true}
              onClose={handleClose}
              originalPath={String(selectedJob.params?.input || '')}
              resultPath={selectedJob.result_path}
              resultText={(selectedJob as any).result || null}
              fileName={selectedJob.result_path.split('/').pop() || 'description.txt'}
              originalIsVideo={String(selectedJob.params?.input || '').match(/\.(mp4|webm|mov|mkv)$/i) !== null}
            />
          )}

          {/* Upscale/Transform jobs use ComparisonPreviewModal */}
          {(selectedJob.type === 'upscale' || selectedJob.type === 'transform') && (
            <ComparisonPreviewModal
              isOpen={true}
              onClose={handleClose}
              originalPath={String(selectedJob.params?.input || '')}
              resultPath={selectedJob.result_path}
              fileName={selectedJob.result_path.split('/').pop() || 'result'}
              resultLabel={selectedJob.type === 'upscale' ? 'Upscaled' : 'Transformed'}
              factor={Number(selectedJob.params?.factor) || 1}
            />
          )}

          {/* Article/Code and other jobs use generic PreviewModal (has Code/Preview toggle for md/html) */}
          {selectedJob.type !== 'analysis' && selectedJob.type !== 'upscale' && selectedJob.type !== 'transform' && (
            <PreviewModal
              isOpen={true}
              onClose={handleClose}
              filePath={selectedJob.result_path}
              fileName={selectedJob.result_path.split('/').pop() || 'file'}
            />
          )}
        </>
      )}
    </div>
  );
}
