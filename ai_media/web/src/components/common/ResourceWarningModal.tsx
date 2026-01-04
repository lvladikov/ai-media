import { AlertTriangle, AlertOctagon } from 'lucide-react';

interface ResourceWarningModalProps {
  isOpen: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  warning: string;
  type?: 'warning' | 'critical';
  details?: {
    input_resolution: string;
    target_resolution: string;
    megapixels: number;
    estimated_ram_gb: number;
    available_ram_gb: number;
  };
}

export function ResourceWarningModal({
  isOpen,
  onConfirm,
  onCancel,
  warning,
  type = 'warning',
  details
}: ResourceWarningModalProps) {
  if (!isOpen) return null;

  const isCritical = type === 'critical';
  const Icon = isCritical ? AlertOctagon : AlertTriangle;
  const colorClass = isCritical ? 'text-red-500' : 'text-orange-500';
  const bgClass = isCritical ? 'bg-red-500/10 border-red-500/30' : 'bg-orange-500/10 border-orange-500/30';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4">
      <div className="bg-primary border border-border rounded-xl max-w-md w-full shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        
        {/* Header */}
        <div className={`p-6 border-b border-border flex items-start gap-4 ${bgClass}`}>
          <div className={`p-3 rounded-full bg-primary/50 ${colorClass}`}>
            <Icon size={32} />
          </div>
          <div>
            <h3 className="text-xl font-bold text-primary mb-1">
              {isCritical ? 'Critical Resource Warning' : 'High Resolution Warning'}
            </h3>
            <p className="text-secondary text-sm leading-relaxed">
              {warning}
            </p>
          </div>
        </div>

        {/* Details */}
        {details && (
          <div className="p-6 bg-primary/50 space-y-3 text-sm">
            <h4 className="font-semibold text-secondary uppercase text-xs tracking-wider mb-3">Estimation Details</h4>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="text-tertiary block text-xs">Target Resolution</span>
                <span className="text-primary font-mono">{details.target_resolution}</span>
              </div>
              <div>
                 <span className="text-tertiary block text-xs">Megapixels</span>
                 <span className="text-primary font-mono">{details.megapixels} MP</span>
              </div>
              <div>
                 <span className="text-tertiary block text-xs">Est. RAM Required</span>
                 <span className={`font-mono ${isCritical ? 'text-red-400' : 'text-orange-400'}`}>
                   ~{details.estimated_ram_gb} GB
                 </span>
              </div>
               <div>
                 <span className="text-tertiary block text-xs">Available RAM</span>
                 <span className="text-secondary font-mono">{details.available_ram_gb} GB</span>
              </div>
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="p-4 bg-primary flex justify-end gap-3 border-t border-border">
          <button 
            onClick={onCancel}
            className="px-4 py-2 rounded-lg font-medium text-secondary hover:text-primary hover:bg-secondary transition-colors"
          >
            Cancel
          </button>
          <button 
            onClick={onConfirm}
            className={`px-6 py-2 rounded-lg font-bold text-primary shadow-lg transition-all ${
              isCritical 
                ? 'bg-red-600 hover:bg-red-500 shadow-red-900/20' 
                : 'bg-orange-600 hover:bg-orange-500 shadow-orange-900/20'
            }`}
          >
            {isCritical ? 'Proceed Anyway (Risk Freeze)' : 'Proceed'}
          </button>
        </div>
      </div>
    </div>
  );
}
