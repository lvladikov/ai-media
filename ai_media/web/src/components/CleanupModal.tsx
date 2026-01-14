import { useState, useEffect } from 'react';
import { X, Trash2, FolderOpen, HardDrive, AlertTriangle, Loader2, CheckCircle, XCircle } from 'lucide-react';
import { useAppStore } from '../store';
import { API_BASE_URL as API_BASE } from '../config';

interface HubModel {
  name: string;
  size: number;
  size_formatted: string;
  path: string;
}

interface HubModelsResponse {
  models: HubModel[];
  total_size: number;
  total_size_formatted: string;
  hub_path: string;
}

interface FolderStats {
  file_count: number;
  size: number;
  size_formatted: string;
  path: string;
}

interface OutputStats {
  testing_output: FolderStats;
  media_output: FolderStats;
}

interface CleanupModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type CleanupAction = 'clear-data-output' | 'clear-media-output' | 'clear-all-outputs' | 'clear-hub-model';

// Helper to format bytes into human-readable size
function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes < 1024 ** 4) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  return `${(bytes / 1024 ** 4).toFixed(1)} TB`;
}

export function CleanupModal({ isOpen, onClose }: CleanupModalProps) {
  const { jobs } = useAppStore();
  const [hubModels, setHubModels] = useState<HubModelsResponse | null>(null);
  const [outputStats, setOutputStats] = useState<OutputStats | null>(null);
  const [loadingModels, setLoadingModels] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [confirmAction, setConfirmAction] = useState<{ action: CleanupAction; folderName?: string } | null>(null);

  // Fetch data when modal opens
  useEffect(() => {
    if (isOpen) {
      fetchHubModels();
      fetchOutputStats();
    }
  }, [isOpen]);

  // Track active job
  const activeJob = activeJobId ? jobs.find(j => j.job_id === activeJobId) : null;
  const isJobRunning = activeJob?.status === 'pending' || activeJob?.status === 'generating';

  const fetchHubModels = async () => {
    setLoadingModels(true);
    try {
      const res = await fetch(`${API_BASE()}/api/cleanup/hub-models`);
      if (res.ok) {
        const data = await res.json();
        setHubModels(data);
      }
    } catch (e) {
      console.error('Failed to fetch hub models:', e);
    } finally {
      setLoadingModels(false);
    }
  };

  const fetchOutputStats = async () => {
    try {
      const res = await fetch(`${API_BASE()}/api/cleanup/output-stats`);
      if (res.ok) {
        const data = await res.json();
        setOutputStats(data);
      }
    } catch (e) {
      console.error('Failed to fetch output stats:', e);
    }
  };

  const startCleanupJob = async (action: CleanupAction, folderName?: string) => {
    try {
      const res = await fetch(`${API_BASE()}/api/cleanup/job`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, folder_name: folderName })
      });
      if (res.ok) {
        const data = await res.json();
        setActiveJobId(data.job_id);
        setConfirmAction(null);
        // Refresh stats after cleanup
        setTimeout(() => {
          fetchOutputStats();
          if (action === 'clear-hub-model') {
            fetchHubModels();
          }
        }, 1000);
      }
    } catch (e) {
      console.error('Failed to start cleanup job:', e);
    }
  };

  const handleAction = (action: CleanupAction, folderName?: string) => {
    setActiveJobId(null);  // Clear previous job logs
    setConfirmAction({ action, folderName });
  };

  const confirmAndExecute = () => {
    if (confirmAction) {
      startCleanupJob(confirmAction.action, confirmAction.folderName);
    }
  };

  if (!isOpen) return null;

  const actionLabels: Record<CleanupAction, string> = {
    'clear-data-output': 'Clear Test Outputs',
    'clear-media-output': 'Clear Media Output',
    'clear-all-outputs': 'Clear All Outputs',
    'clear-hub-model': 'Delete Hub Model'
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-secondary rounded-xl shadow-2xl max-w-4xl w-full max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <h2 className="text-xl font-bold flex items-center gap-2 text-primary">
            <Trash2 className="text-red-500" size={24} />
            Cleanup Options
          </h2>
          <button onClick={onClose} className="p-2 hover:bg-tertiary rounded-lg transition-colors">
            <X size={20} className="text-secondary" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Confirmation Dialog */}
          {confirmAction && (
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <AlertTriangle className="text-yellow-500 shrink-0" size={24} />
                <div className="flex-1">
                  <p className="font-medium text-primary">
                    {confirmAction.action === 'clear-hub-model' 
                      ? `Delete "${confirmAction.folderName}"?`
                      : `${actionLabels[confirmAction.action]}?`}
                  </p>
                  <p className="text-sm text-secondary mt-1">
                    {confirmAction.action === 'clear-hub-model'
                      ? 'This model will be permanently deleted. You will need to redownload it next time.'
                      : confirmAction.action === 'clear-all-outputs'
                      ? 'This action cannot be undone. All files in both folders will be deleted.'
                      : 'This action cannot be undone. All files in the folder will be deleted.'}
                  </p>
                  <div className="flex gap-2 mt-3">
                    <button 
                      onClick={confirmAndExecute}
                      className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm font-medium"
                    >
                      Delete
                    </button>
                    <button 
                      onClick={() => setConfirmAction(null)}
                      className="px-4 py-2 bg-tertiary hover:bg-primary text-secondary rounded-lg text-sm font-medium"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Job Status */}
          {activeJob && (
            <div className={`rounded-lg p-4 border ${
              activeJob.status === 'complete' ? 'bg-green-500/10 border-green-500/30' :
              activeJob.status === 'failed' ? 'bg-red-500/10 border-red-500/30' :
              'bg-brand-500/10 border-brand-500/30'
            }`}>
              <div className="flex items-center gap-2 mb-2">
                {isJobRunning ? (
                  <Loader2 className="animate-spin text-brand-500" size={20} />
                ) : activeJob.status === 'complete' ? (
                  <CheckCircle className="text-green-500" size={20} />
                ) : (
                  <XCircle className="text-red-500" size={20} />
                )}
                <span className="font-medium text-primary">
                  {activeJob.status === 'complete' ? 'Completed' :
                   activeJob.status === 'failed' ? 'Failed' : 'Running...'}
                </span>
              </div>
              
              {/* Logs */}
              {activeJob.logs && activeJob.logs.length > 0 ? (
                <div className="bg-primary/50 rounded p-2 mt-2 max-h-32 overflow-y-auto font-mono text-xs">
                  {activeJob.logs.map((log: string, i: number) => (
                    <div key={i} className="text-secondary">{log}</div>
                  ))}
                </div>
              ) : (
                <>
                  {activeJob.message && (
                    <p className="text-sm text-secondary mt-2">{activeJob.message}</p>
                  )}
                  {activeJob.error && (
                    <p className="text-sm text-red-500 mt-2">{activeJob.error}</p>
                  )}
                </>
              )}
            </div>
          )}

          {/* Quick Actions */}
          <div className="space-y-2">
            <h3 className="text-sm font-semibold text-secondary uppercase tracking-wide">Quick Actions</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <button
                onClick={() => handleAction('clear-data-output')}
                disabled={isJobRunning}
                className="flex flex-col items-start p-4 bg-tertiary hover:bg-primary rounded-lg transition-colors disabled:opacity-50"
              >
                <div className="flex items-center gap-2 mb-1">
                  <FolderOpen size={18} className="text-brand-500" />
                  <span className="text-sm font-medium text-primary">Clear Test Outputs</span>
                </div>
                <span className="text-xs text-secondary">
                  {outputStats ? `${outputStats.testing_output.file_count} files • ${outputStats.testing_output.size_formatted}` : 'Loading...'}
                </span>
              </button>
              <button
                onClick={() => handleAction('clear-media-output')}
                disabled={isJobRunning}
                className="flex flex-col items-start p-4 bg-tertiary hover:bg-primary rounded-lg transition-colors disabled:opacity-50"
              >
                <div className="flex items-center gap-2 mb-1">
                  <FolderOpen size={18} className="text-orange-500" />
                  <span className="text-sm font-medium text-primary">Clear Media Output</span>
                </div>
                <span className="text-xs text-secondary">
                  {outputStats ? `${outputStats.media_output.file_count} files • ${outputStats.media_output.size_formatted}` : 'Loading...'}
                </span>
              </button>
              <button
                onClick={() => handleAction('clear-all-outputs')}
                disabled={isJobRunning}
                className="flex flex-col items-start p-4 bg-tertiary hover:bg-primary rounded-lg transition-colors disabled:opacity-50"
              >
                <div className="flex items-center gap-2 mb-1">
                  <Trash2 size={18} className="text-red-500" />
                  <span className="text-sm font-medium text-primary">Clear All Outputs</span>
                </div>
                <span className="text-xs text-secondary">
                  {outputStats ? `${outputStats.testing_output.file_count + outputStats.media_output.file_count} files • ${formatSize(outputStats.testing_output.size + outputStats.media_output.size)}` : 'Loading...'}
                </span>
              </button>
            </div>
          </div>

          {/* Hub Models */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-secondary uppercase tracking-wide flex items-center gap-2">
                <HardDrive size={16} />
                Cached Hub Models
              </h3>
              {hubModels && (
                <span className="text-xs text-secondary">
                  {hubModels.models.length} models • {hubModels.total_size_formatted}
                </span>
              )}
            </div>

            {loadingModels ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="animate-spin text-brand-500" size={24} />
                <span className="ml-2 text-secondary">Scanning hub folder...</span>
              </div>
            ) : hubModels && hubModels.models.length > 0 ? (
              <div className="bg-tertiary rounded-lg max-h-64 overflow-y-auto border border-border">
                {hubModels.models.map((model, index) => (
                  <div 
                    key={model.name}
                    className={`flex items-center justify-between p-3 border-b border-white/10 last:border-b-0 hover:bg-primary/50 ${index % 2 === 1 ? 'bg-primary/20' : ''}`}
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-primary truncate" title={model.name}>
                        {model.name}
                      </p>
                      <p className="text-xs text-secondary">{model.size_formatted}</p>
                    </div>
                    <button
                      onClick={() => handleAction('clear-hub-model', model.name)}
                      disabled={isJobRunning}
                      className="p-2 text-red-500 hover:bg-red-500/10 rounded-lg transition-colors disabled:opacity-50"
                      title="Delete this model"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-secondary">
                {hubModels ? 'No cached models found' : 'Failed to load hub models'}
              </div>
            )}

            {hubModels && (
              <p className="text-xs text-secondary">
                Hub path: {hubModels.hub_path}
              </p>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border">
          <button
            onClick={onClose}
            className="w-full py-2 bg-tertiary hover:bg-primary text-secondary rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
