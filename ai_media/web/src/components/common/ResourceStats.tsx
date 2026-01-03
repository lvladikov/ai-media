import { useAppStore } from '../../store';
import { Cpu, HardDrive, Gpu, Activity, Server, Zap } from 'lucide-react';

interface ResourceStatsProps {
  className?: string;
  variant?: 'compact' | 'full' | 'modal';
}

export function ResourceStats({ className = '', variant = 'full' }: ResourceStatsProps) {
  const { resources, systemInfo, activeTab } = useAppStore();
  const isModal = variant === 'modal';

  if (!resources) {
    return (
      <div className={`flex items-center gap-2 text-slate-500 text-xs ${className}`}>
        <Activity size={12} className="animate-pulse" />
        <span>Loading resources...</span>
      </div>
    );
  }


  // Hiding Rules
  const isMacMPS = systemInfo?.device === 'mps';
  // const showVram = !isMacMPS;
  // const showGpu = !isMacMPS;
  const showVram = true;
  const showGpu = true;

  // Dynamic Process Label
  let processLabel = 'Process';
  let processFullLabel = 'AI Process';

  switch (activeTab) {
    case 'chat': 
      processLabel = 'Chat'; 
      processFullLabel = 'Chat Model'; 
      break;
    case 'image': 
      processLabel = 'Image'; 
      processFullLabel = 'Image Model'; 
      break;
    case 'video': 
      processLabel = 'Video'; 
      processFullLabel = 'Video Model'; 
      break;
    case 'audio': 
      processLabel = 'Audio'; 
      processFullLabel = 'Audio Model'; 
      break;
    case 'article': 
      processLabel = 'Article'; 
      processFullLabel = 'Article Generation'; 
      break;
    case 'code': 
      processLabel = 'Code'; 
      processFullLabel = 'Code Generation'; 
      break;
  }

  const getUsageColor = (value: number) => {
    if (value > 80) return 'text-red-400';
    if (value > 50) return 'text-yellow-400';
    return 'text-green-400';
  };

  // Custom grid for modal to fit in smaller width (max-w-3xl with padding)
  // Default: ~745px fixed + flex
  // Modal:   ~510px fixed + flex (Compact for short labels)
  const gridClass = isModal 
    ? "grid grid-cols-[70px_80px_140px_80px_140px_1fr] items-center" 
    : "grid grid-cols-[80px_110px_200px_110px_200px_1fr] items-center";

  const textSize = isModal ? "text-[10px]" : "text-xs";
  const iconSize = isModal ? 12 : 14;

  return (
    <div className={`flex flex-col gap-1.5 ${textSize} text-slate-400 ${className}`}>

      {/* --- GRID HEADER / GLOBAL --- */}
      <div className={gridClass}>

        {/* Label Column */}
        <div className="flex items-center gap-2 pr-2">
          <Server size={12} className="text-slate-500 shrink-0" />
          <span 
            className="font-semibold text-slate-500 uppercase tracking-wider text-[10px] whitespace-nowrap cursor-help"
            title="Global System Resources"
          >
            Global
          </span>
        </div>

        {/* CPU */}
        <div className="flex items-center gap-2">
          <Cpu size={iconSize} className="text-primary-400/70 shrink-0" />
          <span className={variant === 'compact' ? 'hidden' : ''}>CPU</span>
          <span className="ml-auto font-bold text-white">{resources.global.cpu_percent.toFixed(0)}%</span>
        </div>

        {/* RAM */}
        <div className="flex items-center gap-2 px-2 md:px-4 border-l border-slate-800/30 text-green-400">
          <HardDrive size={iconSize} className="shrink-0" />
          <span className={variant === 'compact' ? 'hidden' : ''}>RAM</span>
          <div className={`flex items-center ml-auto font-bold text-white gap-1 w-[140px] justify-between`}>
            <span className="text-right flex-1">{resources.global.ram_used_gb.toFixed(1)}</span>
            <span className="opacity-50 px-1">/</span>
            <div className="flex items-center gap-1 justify-end flex-1">
              <span>{resources.global.ram_total_gb.toFixed(1)}</span>
              <span className="opacity-70 text-[10px] font-normal">GB</span>
            </div>
          </div>
        </div>

        {/* GPU */}
        {showGpu ? (
          <div 
            className="flex items-center gap-2 px-2 md:px-4 border-l border-slate-800/30 text-purple-400"
            title={isMacMPS ? "GPU usage monitoring is not available on Apple Silicon (MPS)." : undefined}
          >
            <Activity size={iconSize} className="shrink-0" />
            <span className={variant === 'compact' ? 'hidden' : ''}>GPU</span>
            <span className="ml-auto font-bold text-white">{(resources.global.gpu_percent || 0).toFixed(0)}%</span>
          </div>
        ) : <div />}

        {/* VRAM */}
        {showVram ? (
          <div 
            className="flex items-center gap-2 px-2 md:px-4 border-l border-slate-800/30 text-yellow-400"
            title={isMacMPS ? "Max VRAM is limited by macOS (~75% of System RAM) to prevent system instability." : undefined}
          >
            <Gpu size={iconSize} className="shrink-0" />
            <span className={variant === 'compact' ? 'hidden' : ''}>VRAM</span>
            <div className={`flex items-center ml-auto font-bold text-white gap-1 w-[140px] justify-between`}>
              <span className="text-right flex-1">{resources.global.vram_used_gb.toFixed(1)}</span>
              <span className="opacity-50 px-1">/</span>
              <div className="flex items-center gap-1 justify-end flex-1">
                <span>{resources.global.vram_total_gb.toFixed(1)}</span>
                <span className="opacity-70 text-[10px] font-normal">GB</span>
              </div>
            </div>
          </div>
        ) : <div />}

        {/* Swap */}
        {resources.global.swap_used_gb !== undefined && resources.global.swap_used_gb > 0 ? (
          <div className="flex items-center gap-2 px-2 md:px-4 border-l border-slate-800/30 text-orange-400">
            <Activity size={12} className="shrink-0" />
            <span className={variant === 'compact' ? 'hidden' : ''}>Swap</span>
            <span className="text-white font-bold whitespace-nowrap">
              {resources.global.swap_used_gb.toFixed(1)} <span className="text-[10px] font-normal">GB</span>
            </span>
          </div>
        ) : <div />}
      </div>

      {/* --- FEATURE ROW --- */}
      <div className={`${gridClass} border-t border-slate-800/40 pt-1.5 opacity-90`}>

        {/* Label Column */}
        <div className="flex items-center text-blue-400 pr-2 gap-2">
          <Zap size={12} className="shrink-0" />
          <span 
            className="font-semibold uppercase tracking-wider text-[10px] whitespace-nowrap cursor-help" 
            title={`${processFullLabel} (${resources.process.pid})`}
          >
            {processLabel}
          </span>
        </div>

        {/* CPU */}
        <div className="flex items-center gap-2">
          <Cpu size={iconSize} className={`${getUsageColor(resources.process.cpu_percent)} shrink-0`} />
          <span className={variant === 'compact' ? 'hidden' : ''}>CPU</span>
          <span className="ml-auto font-bold text-white">{resources.process.cpu_percent.toFixed(0)}%</span>
        </div>

        {/* RAM */}
        <div className="flex items-center gap-2 px-2 md:px-4 border-l border-slate-800/30 text-green-400">
          <HardDrive size={iconSize} className="shrink-0" />
          <span className={variant === 'compact' ? 'hidden' : ''}>RAM</span>
          <div className={`flex items-center ml-auto font-bold text-white gap-1 w-[140px] justify-end`}>
            <span className="inline-block text-right">{resources.process.ram_used_gb.toFixed(1)}</span>
            <span className="opacity-70 text-[10px] font-normal">GB</span>
          </div>
        </div>

        {/* GPU */}
        {showGpu ? (
          <div 
            className="flex items-center gap-2 px-2 md:px-4 border-l border-slate-800/30 text-purple-400"
            title={isMacMPS ? "GPU usage monitoring is not available on Apple Silicon (MPS). Values will verify between 0%." : undefined}
          >
            <Activity size={iconSize} className="shrink-0" />
            <span className={variant === 'compact' ? 'hidden' : ''}>GPU</span>
            <span className="ml-auto font-bold text-white">{(resources.process.gpu_percent || 0).toFixed(0)}%</span>
          </div>
        ) : <div />}

        {/* VRAM */}
        {showVram ? (
          <div 
            className="flex items-center gap-2 px-2 md:px-4 border-l border-slate-800/30 text-yellow-400"
            title={isMacMPS ? "Max VRAM is limited by macOS (~75% of System RAM) to prevent system instability." : undefined}
          >
            <Gpu size={iconSize} className="shrink-0" />
            <span className={variant === 'compact' ? 'hidden' : ''}>VRAM</span>
            <div className={`flex items-center ml-auto font-bold text-white gap-1 w-[140px] justify-end`}>
              <span className="inline-block text-right">{(resources.process.vram_used_gb || 0).toFixed(1)}</span>
              <span className="opacity-70 text-[10px] font-normal">GB</span>
            </div>
          </div>
        ) : <div />}

        <div />
      </div>

    </div>
  );
}
