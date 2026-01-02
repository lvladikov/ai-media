import { useAppStore } from '../../store';
import { Cpu, HardDrive, Gpu, Activity, Server, Zap } from 'lucide-react';

interface ResourceStatsProps {
  className?: string;
  variant?: 'compact' | 'full';
}

export function ResourceStats({ className = '', variant = 'full' }: ResourceStatsProps) {
  const { resources, systemInfo, activeTab } = useAppStore();

  if (!resources) {
    return (
      <div className={`flex items-center gap-2 text-slate-500 text-xs ${className}`}>
        <Activity size={12} className="animate-pulse" />
        <span>Loading resources...</span>
      </div>
    );
  }

  const formatGb = (gb: number) => gb.toFixed(1);
  const formatPercent = (p: number) => `${p.toFixed(0)}%`;
  
  // Hiding Rules
  const isMacMPS = systemInfo?.device === 'mps';
  // On Mac/MPS, we hide VRAM/GPU metrics for simplicity as requested
  const showVram = !isMacMPS; 
  const showGpu = !isMacMPS;

  // Dynamic Process Label
  let processLabel = 'AI Model';
  switch(activeTab) {
    case 'chat': processLabel = 'Chat Model'; break;
    case 'image': processLabel = 'Image Model'; break;
    case 'video': processLabel = 'Video Model'; break;
    case 'audio': processLabel = 'Audio Model'; break;
    case 'article': processLabel = 'Article Gen'; break;
    case 'code': processLabel = 'Code Gen'; break;
    default: processLabel = 'AI Process';
  }

  const getUsageColor = (value: number) => {
    if (value > 80) return 'text-red-400';
    if (value > 50) return 'text-yellow-400';
    return 'text-green-400';
  };

  return (
    <div className={`flex flex-col gap-1 text-xs text-slate-400 ${className}`}>
      
      {/* Row 1: Global Resources */}
      <div className="flex items-center">
        <div className="flex items-center gap-1.5 w-40 shrink-0">
           <Server size={12} className="text-slate-500" />
           <span className="font-semibold text-slate-500 uppercase tracking-wider text-[10px]">Global:</span>
        </div>

        {/* Global CPU */}
        <div className="flex items-center gap-1.5 w-32 shrink-0">
          <Cpu size={14} className="text-primary-400/70" />
          <span className={variant === 'compact' ? 'hidden sm:inline' : ''}>CPU:</span>
          <span className="text-slate-200 font-medium">{formatPercent(resources.global.cpu_percent)}</span>
        </div>

        {/* Global RAM */}
        <div className="flex items-center gap-1.5 w-48 shrink-0">
          <HardDrive size={14} className="text-green-400/70" />
          <span className={variant === 'compact' ? 'hidden sm:inline' : ''}>RAM:</span>
          <span className="text-slate-200 font-medium">
            {formatGb(resources.global.ram_used_gb)}{variant === 'full' && ` / ${formatGb(resources.global.ram_total_gb)}`} GB
          </span>
        </div>

        {/* Global Swap */}
        {resources.global.swap_used_gb !== undefined && resources.global.swap_used_gb > 0 && (
          <div className="flex items-center gap-1.5 w-44 shrink-0">
            <Activity size={12} className="text-orange-400/70" />
            <span className={variant === 'compact' ? 'hidden sm:inline' : ''}>Swap:</span>
            <span className="text-slate-200 font-medium">{formatGb(resources.global.swap_used_gb)} GB</span>
          </div>
        )}

        {/* Global VRAM (Hidden on Mac) */}
        {showVram && (
          <div className="flex items-center gap-1.5 w-48 shrink-0">
            <Gpu size={14} className="text-yellow-400/70" />
            <span className={variant === 'compact' ? 'hidden sm:inline' : ''}>VRAM:</span>
            <span className="text-slate-200 font-medium">
              {formatGb(resources.global.vram_used_gb)}{variant === 'full' && ` / ${formatGb(resources.global.vram_total_gb)}`} GB
            </span>
          </div>
        )}

         {/* Global GPU (Hidden on Mac) */}
         {showGpu && resources.global.gpu_percent > 0 && (
          <div className="flex items-center gap-1.5 w-32 shrink-0">
            <Activity size={14} className="text-purple-400/70" />
            <span className={variant === 'compact' ? 'hidden sm:inline' : ''}>GPU:</span>
            <span className="text-slate-200 font-medium">{formatPercent(resources.global.gpu_percent)}</span>
          </div>
        )}
      </div>

      {/* Row 2: Process Resources */}
      <div className="flex items-center">
                <div className="flex items-center text-blue-400 w-40 shrink-0 gap-1.5">
                    <Zap size={12} />
                    <span className="font-semibold uppercase tracking-wider text-[10px] truncate" title={`PID: ${resources.process.pid}`}>
                        {processLabel}:
                    </span>
                </div>
                
                <div className={`flex items-center gap-1.5 ${getUsageColor(resources.process.cpu_percent)} w-32 shrink-0`}>
          <Cpu size={14} className="text-primary-400" />
          <span className={variant === 'compact' ? 'hidden sm:inline' : ''}>CPU:</span>
          <span className="text-white font-bold">{formatPercent(resources.process.cpu_percent)}</span>
        </div>

        {/* Process RAM */}
        <div className="flex items-center gap-1.5 w-48 shrink-0">
          <HardDrive size={14} className="text-green-400" />
          <span className={variant === 'compact' ? 'hidden sm:inline' : ''}>RAM:</span>
          <span className="text-white font-bold">
            {formatGb(resources.process.ram_used_gb)} GB
          </span>
        </div>

        {/* Process VRAM (CUDA only) */}
        {resources.process.vram_used_gb !== undefined && resources.process.vram_used_gb > 0 && (
          <div className="flex items-center gap-1.5 w-48 shrink-0">
             <Gpu size={14} className="text-yellow-400" />
             <span className={variant === 'compact' ? 'hidden sm:inline' : ''}>VRAM:</span>
             <span className="text-white font-bold">
               {formatGb(resources.process.vram_used_gb)} GB
             </span>
          </div>
        )}
      </div>
      
    </div>
  );
}
