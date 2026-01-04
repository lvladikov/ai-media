import { useState } from 'react';
import { X, Download, Image, Columns, ArrowLeftRight } from 'lucide-react';

interface ImageMetadata {
  width: number;
  height: number;
  fileSize?: number; // in bytes
}

interface ComparisonPreviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  originalPath: string;  // Data URL or server path for original
  resultPath: string;    // Server path for result
  fileName: string;
  resultLabel?: string;  // Label for the result tab (e.g., "Transformed", "Converted")
  originalFormat?: string; // Format of original (e.g., "JPG", "PNG")
  resultFormat?: string;   // Format of result (e.g., "GIF", "TIFF")
  originalFileSize?: number; // Original file size in bytes (optional)
  resultFileSize?: number;   // Result file size in bytes (optional)
  factor?: number;       // Scaling factor (e.g. 2.0, 0.5)
}

type ViewTab = 'result' | 'original' | 'sideBySide';

// Format file size for display
const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
};

export function ComparisonPreviewModal({ 
  isOpen, 
  onClose, 
  originalPath, 
  resultPath, 
  fileName,
  resultLabel = 'Result',
  originalFormat,
  resultFormat,
  originalFileSize,
  resultFileSize,
  factor = 1.0
}: ComparisonPreviewModalProps) {
  const [activeTab, setActiveTab] = useState<ViewTab>('result');
  const [originalMeta, setOriginalMeta] = useState<ImageMetadata | null>(null);
  const [resultMeta, setResultMeta] = useState<ImageMetadata | null>(null);
  const [originalError, setOriginalError] = useState(false);
  const [resultError, setResultError] = useState(false);
  
  if (!isOpen) return null;

  // Check for formats that can't be previewed in browsers
  const nonPreviewableFormats = ['tiff', 'tif', 'psd', 'raw'];
  const resultCannotPreview = resultFormat && nonPreviewableFormats.includes(resultFormat.toLowerCase());
  const originalCannotPreview = originalFormat && nonPreviewableFormats.includes(originalFormat.toLowerCase());

  // Build URLs - originalPath might be data URL, blob URL, or server path
  const isFullUrl = originalPath && (originalPath.startsWith('data:') || originalPath.startsWith('blob:') || originalPath.startsWith('http'));
  const originalUrl = isFullUrl ? originalPath : `http://localhost:8000/api/files/${originalPath}`;
  const resultUrl = `http://localhost:8000/api/files/${resultPath}`;

  const handleDownload = () => {
    const a = document.createElement('a');
    a.href = `${resultUrl}?download=true`;
    a.download = fileName;
    a.click();
  };

  // Build labels with optional format suffix
  const resultTabLabel = resultFormat ? `${resultLabel} (${resultFormat.toUpperCase()})` : resultLabel;
  const originalTabLabel = originalFormat ? `Original (${originalFormat.toUpperCase()})` : 'Original';

  const tabs: { id: ViewTab; label: string; icon: React.ReactNode }[] = [
    { id: 'result', label: resultTabLabel, icon: <Image size={14} /> },
    { id: 'original', label: originalTabLabel, icon: <Image size={14} /> },
    { id: 'sideBySide', label: 'Side by Side', icon: <Columns size={14} /> },
  ];

  // Calculate relative sizes for Side by Side
  // If factor > 1 (Upscale): Result is 100% height, Original is (1/factor)% height
  // If factor < 1 (Downscale): Original is 100% height, Result is (factor)% height
  const getRelativeHeight = (isOriginal: boolean) => {
      // Base styles
      const base = "max-w-full object-contain rounded-lg border shadow-xl";
      
      if (factor >= 1) {
          // Upscaling
          if (!isOriginal) return { className: `${base} border-primary-500/30 shadow-primary-900/10`, style: { maxHeight: 'calc(100% - 2rem)', height: '100%' } };
          return { className: `${base} border-slate-700`, style: { maxHeight: 'calc(100% - 2rem)', height: `${(1/factor) * 100}%` } };
      } else {
          // Downscaling
          if (isOriginal) return { className: `${base} border-slate-700`, style: { maxHeight: 'calc(100% - 2rem)', height: '100%' } };
          return { className: `${base} border-primary-500/30 shadow-primary-900/10`, style: { maxHeight: 'calc(100% - 2rem)', height: `${factor * 100}%` } };
      }
  };

  const originalStyle = getRelativeHeight(true);
  const resultStyle = getRelativeHeight(false);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4" onClick={onClose}>
      <div 
        className="relative bg-slate-800 rounded-xl p-4 w-[90vw] h-[90vh] flex flex-col" 
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: '90vw', maxHeight: '90vh' }}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-0 flex-shrink-0 px-4 pt-4">
          <div className="flex items-center gap-3 pb-4">
            <h3 className="text-lg font-semibold truncate max-w-md flex items-center gap-2 text-white">
              <ArrowLeftRight size={18} className="text-primary-400" />
              {fileName}
            </h3>
          </div>
          
          <div className="flex items-end gap-4 h-full relative top-[1px]">
            {/* Tab buttons */}
            <div className="flex items-end">
              {tabs.map((tab, idx) => (
                <button 
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-4 py-2 rounded-t-lg text-sm font-medium flex items-center gap-2 transition-all border-t border-x ${idx > 0 ? 'ml-[-1px]' : ''} ${
                    activeTab === tab.id 
                      ? 'bg-[#1e1e1e] text-primary-400 border-slate-600 border-b-[#1e1e1e] z-10' 
                      : 'bg-slate-800 text-slate-400 border-transparent hover:bg-slate-750 hover:text-slate-300 border-b-slate-600'
                  }`}
                >
                  {tab.icon} {tab.label}
                </button>
              ))}
            </div>
          
            <div className="flex items-center gap-2 pb-3 mb-1 pl-4 border-l border-slate-700/50">
              <button className="btn-secondary p-1.5 h-8 w-8 flex items-center justify-center" onClick={handleDownload} title={`Download ${resultLabel}`}>
                <Download size={16} />
              </button>
              <button className="btn-secondary p-1.5 h-8 w-8 flex items-center justify-center hover:bg-red-500/20 hover:text-red-200 hover:border-red-500/50" onClick={onClose} title="Close">
                <X size={16} />
              </button>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="relative flex-1 min-h-0 overflow-auto scrollbar-themed bg-[#1e1e1e] border border-slate-600 rounded-b-lg rounded-tr-lg flex items-center justify-center p-4">
          {activeTab === 'result' && (
            (resultCannotPreview || resultError) ? (
              <div className="flex flex-col items-center gap-4 text-center p-8">
                <div className="w-24 h-24 bg-slate-700/50 rounded-xl flex items-center justify-center border border-slate-600">
                  <Image size={48} className="text-slate-500" />
                </div>
                <div className="space-y-2">
                  <h4 className="text-lg font-medium text-slate-300">
                    {resultCannotPreview ? `${resultFormat?.toUpperCase()} files cannot be previewed` : 'Failed to load image'}
                  </h4>
                  <p className="text-sm text-slate-500 max-w-sm">
                    {resultCannotPreview 
                      ? 'This format is not supported for browser preview. Download the file to view it in an image editor.'
                      : 'There was an error loading the result image. It may be corrupted or missing.'}
                  </p>
                </div>
                <button 
                  onClick={handleDownload}
                  className="btn-primary flex items-center gap-2"
                >
                  <Download size={16} />
                  Download {resultFormat?.toUpperCase()}
                </button>
              </div>
            ) : (
              <div className="relative">
                <img
                  src={resultUrl}
                  alt={resultLabel}
                  className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
                  onLoad={(e) => setResultMeta({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })}
                  onError={() => setResultError(true)}
                />
                {resultMeta && (
                  <div className="absolute bottom-2 left-2 text-[11px] text-primary-300/80 bg-black/60 px-2 py-1 rounded backdrop-blur-sm">
                    {resultMeta.width}×{resultMeta.height}{resultFileSize && ` • ${formatFileSize(resultFileSize)}`}
                  </div>
                )}
              </div>
            )
          )}
          
          {activeTab === 'original' && (
            (originalCannotPreview || originalError) ? (
              <div className="flex flex-col items-center gap-4 text-center p-8">
                <div className="w-24 h-24 bg-slate-700/50 rounded-xl flex items-center justify-center border border-slate-600">
                  <Image size={48} className="text-slate-500" />
                </div>
                <div className="space-y-2">
                  <h4 className="text-lg font-medium text-slate-300">
                    {originalCannotPreview ? `${originalFormat?.toUpperCase() || 'This'} file cannot be previewed` : 'Failed to load original image'}
                  </h4>
                  <p className="text-sm text-slate-500 max-w-sm">
                    {originalCannotPreview 
                      ? 'This format is not supported for browser preview.'
                      : 'There was an error loading the original image.'}
                  </p>
                </div>
              </div>
            ) : (
              <div className="relative">
                <img
                  src={originalUrl}
                  alt="Original"
                  className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
                  onLoad={(e) => setOriginalMeta({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })}
                  onError={() => setOriginalError(true)}
                />
                {originalMeta && (
                  <div className="absolute bottom-2 left-2 text-[11px] text-slate-400 bg-black/60 px-2 py-1 rounded backdrop-blur-sm">
                    {originalMeta.width}×{originalMeta.height}{originalFileSize && ` • ${formatFileSize(originalFileSize)}`}
                  </div>
                )}
              </div>
            )
          )}
          
          {activeTab === 'sideBySide' && (
            <div className="flex flex-col md:flex-row gap-4 w-full h-full items-center justify-center">
              <div className="flex-1 flex flex-col items-center gap-2 h-full justify-center min-h-0 w-full">
                <div className="flex flex-col items-center gap-0.5">
                  <span className="text-xs font-medium text-slate-400 bg-slate-800 px-2 py-1 rounded-t border border-b-0 border-slate-700">Original</span>
                  {(originalMeta || originalFileSize) && (
                    <span className="text-[10px] text-slate-500 bg-slate-800/80 px-2 py-0.5 rounded-b border border-t-0 border-slate-700">
                      {originalMeta && `${originalMeta.width}×${originalMeta.height}`}
                      {originalMeta && originalFileSize && ' • '}
                      {originalFileSize && formatFileSize(originalFileSize)}
                    </span>
                  )}
                </div>
                {originalCannotPreview || originalError ? (
                  <div className="flex flex-col items-center justify-center gap-3 p-6 bg-slate-800/50 rounded-lg border border-slate-700 h-full w-full max-w-[400px]">
                    <Image size={32} className="text-slate-500" />
                    <span className="text-xs text-slate-500">
                      {originalCannotPreview ? `${originalFormat?.toUpperCase()} preview unavailable` : 'Failed to load original'}
                    </span>
                  </div>
                ) : (
                  <img
                    src={originalUrl}
                    alt="Original"
                    className={originalStyle.className}
                    style={originalStyle.style}
                    onLoad={(e) => setOriginalMeta({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })}
                    onError={() => setOriginalError(true)}
                  />
                )}
              </div>
              <div className="flex-1 flex flex-col items-center gap-2 h-full justify-center min-h-0 w-full">
                <div className="flex flex-col items-center gap-0.5">
                  <span className="text-xs font-medium text-primary-400 bg-primary-500/10 px-2 py-1 rounded-t border border-b-0 border-primary-500/30">{resultLabel}</span>
                  {!resultCannotPreview && (resultMeta || resultFileSize) && (
                    <span className="text-[10px] text-primary-300/70 bg-primary-500/5 px-2 py-0.5 rounded-b border border-t-0 border-primary-500/20">
                      {resultMeta && `${resultMeta.width}×${resultMeta.height}`}
                      {resultMeta && resultFileSize && ' • '}
                      {resultFileSize && formatFileSize(resultFileSize)}
                    </span>
                  )}
                </div>
                {resultCannotPreview || resultError ? (
                  <div className="flex flex-col items-center justify-center gap-3 p-6 bg-slate-800/50 rounded-lg border border-slate-700 h-full w-full max-w-[400px]">
                    <Image size={32} className="text-slate-500" />
                    <span className="text-xs text-slate-500">
                      {resultCannotPreview ? `${resultFormat?.toUpperCase()} preview unavailable` : 'Failed to load result'}
                    </span>
                    <button onClick={handleDownload} className="btn-primary text-xs py-1 px-3 flex items-center gap-1.5 mt-2">
                       <Download size={12} />
                       Download
                    </button>
                  </div>
                ) : (
                  <img
                    src={resultUrl}
                    alt={resultLabel}
                    className={resultStyle.className}
                    style={resultStyle.style}
                    onLoad={(e) => setResultMeta({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })}
                  />
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
