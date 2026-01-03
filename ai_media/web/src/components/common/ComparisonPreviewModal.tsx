import { useState } from 'react';
import { X, Download, Image, Columns, ArrowLeftRight } from 'lucide-react';

interface ComparisonPreviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  originalPath: string;  // Data URL or server path for original
  resultPath: string;    // Server path for result
  fileName: string;
  resultLabel?: string;  // Label for the result tab (e.g., "Transformed", "Converted")
  factor?: number;       // Scaling factor (e.g. 2.0, 0.5)
}

type ViewTab = 'result' | 'original' | 'sideBySide';

export function ComparisonPreviewModal({ 
  isOpen, 
  onClose, 
  originalPath, 
  resultPath, 
  fileName,
  resultLabel = 'Result',
  factor = 1.0
}: ComparisonPreviewModalProps) {
  const [activeTab, setActiveTab] = useState<ViewTab>('result');
  
  if (!isOpen) return null;

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

  const tabs: { id: ViewTab; label: string; icon: React.ReactNode }[] = [
    { id: 'result', label: resultLabel, icon: <Image size={14} /> },
    { id: 'original', label: 'Original', icon: <Image size={14} /> },
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
            <img
              src={resultUrl}
              alt={resultLabel}
              className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
            />
          )}
          
          {activeTab === 'original' && (
            <img
              src={originalUrl}
              alt="Original"
              className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
            />
          )}
          
          {activeTab === 'sideBySide' && (
            <div className="flex flex-col md:flex-row gap-4 w-full h-full items-center justify-center">
              <div className="flex-1 flex flex-col items-center gap-2 h-full justify-center min-h-0 w-full">
                <span className="text-xs font-medium text-slate-400 bg-slate-800 px-2 py-1 rounded border border-slate-700">Original</span>
                <img
                  src={originalUrl}
                  alt="Original"
                  className={originalStyle.className}
                  style={originalStyle.style}
                />
              </div>
              <div className="flex-1 flex flex-col items-center gap-2 h-full justify-center min-h-0 w-full">
                <span className="text-xs font-medium text-primary-400 bg-primary-500/10 px-2 py-1 rounded border border-primary-500/30">{resultLabel}</span>
                <img
                  src={resultUrl}
                  alt={resultLabel}
                  className={resultStyle.className}
                  style={resultStyle.style}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
