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


  // Hiding Rules
  const isMacMPS = systemInfo?.device === 'mps';
  const showVram = !isMacMPS;
  const showGpu = !isMacMPS;

  // Dynamic Process Label
  let processLabel = 'AI Model';
  switch (activeTab) {
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
    <div className={`flex flex-col gap-1.5 text-xs text-slate-400 ${className}`}>

      {/* --- GRID HEADER / GLOBAL --- */}
      <div className="grid grid-cols-[125px_110px_200px_110px_200px_1fr] items-center">

        {/* Label Column */}
        <div className="flex items-center gap-2 pr-2">
          <Server size={12} className="text-slate-500 shrink-0" />
          <span className="font-semibold text-slate-500 uppercase tracking-wider text-[10px] whitespace-nowrap">Global System</span>
        </div>

        {/* CPU */}
        <div className="flex items-center gap-2">
          <Cpu size={14} className="text-primary-400/70 shrink-0" />
          <span className={variant === 'compact' ? 'hidden' : ''}>CPU</span>
          <span className="ml-auto font-bold text-white">{resources.global.cpu_percent.toFixed(0)}%</span>
        </div>

        {/* RAM */}
        <div className="flex items-center gap-2 px-4 border-l border-slate-800/30 text-green-400">
          <HardDrive size={14} className="shrink-0" />
          <span className={variant === 'compact' ? 'hidden' : ''}>RAM</span>
          <div className="flex items-center ml-auto font-bold text-white gap-1 w-[140px] justify-between">
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
          <div className="flex items-center gap-2 px-4 border-l border-slate-800/30 text-purple-400">
            <Activity size={14} className="shrink-0" />
            <span className={variant === 'compact' ? 'hidden' : ''}>GPU</span>
            <span className="ml-auto font-bold text-white">{(resources.global.gpu_percent || 0).toFixed(0)}%</span>
          </div>
        ) : <div />}

        {/* VRAM */}
        {showVram ? (
          <div className="flex items-center gap-2 px-4 border-l border-slate-800/30 text-yellow-400">
            <Gpu size={14} className="shrink-0" />
            <span className={variant === 'compact' ? 'hidden' : ''}>VRAM</span>
            <div className="flex items-center ml-auto font-bold text-white gap-1 w-[140px] justify-between">
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
          <div className="flex items-center gap-2 px-4 border-l border-slate-800/30 text-orange-400">
            <Activity size={12} className="shrink-0" />
            <span className={variant === 'compact' ? 'hidden' : ''}>Swap</span>
            <span className="text-white font-bold ml-auto whitespace-nowrap">
              {resources.global.swap_used_gb.toFixed(1)} <span className="text-[10px] font-normal">GB</span>
            </span>
          </div>
        ) : <div />}
      </div>

      {/* --- FEATURE ROW --- */}
      <div className="grid grid-cols-[125px_110px_200px_110px_200px_1fr] items-center border-t border-slate-800/40 pt-1.5 opacity-90">

        {/* Label Column */}
        <div className="flex items-center text-blue-400 pr-2 gap-2">
          <Zap size={12} className="shrink-0" />
          <span className="font-semibold uppercase tracking-wider text-[10px] whitespace-nowrap" title={`PID: ${resources.process.pid}`}>
            {processLabel}
          </span>
        </div>

        {/* CPU */}
        <div className="flex items-center gap-2">
          <Cpu size={14} className={`${getUsageColor(resources.process.cpu_percent)} shrink-0`} />
          <span className={variant === 'compact' ? 'hidden' : ''}>CPU</span>
          <span className="ml-auto font-bold text-white">{resources.process.cpu_percent.toFixed(0)}%</span>
        </div>

        {/* RAM */}
        <div className="flex items-center gap-2 px-4 border-l border-slate-800/30 text-green-400">
          <HardDrive size={14} className="shrink-0" />
          <span className={variant === 'compact' ? 'hidden' : ''}>RAM</span>
          <div className="flex items-center ml-auto font-bold text-white gap-1 w-[140px] justify-end">
            <span className="inline-block text-right">{resources.process.ram_used_gb.toFixed(1)}</span>
            <span className="opacity-70 text-[10px] font-normal">GB</span>
          </div>
        </div>

        {/* GPU */}
        {showGpu ? (
          <div className="flex items-center gap-2 px-4 border-l border-slate-800/30 text-purple-400">
            <Activity size={14} className="shrink-0" />
            <span className={variant === 'compact' ? 'hidden' : ''}>GPU</span>
            <span className="ml-auto font-bold text-white">{(resources.process.gpu_percent || 0).toFixed(0)}%</span>
          </div>
        ) : <div />}

        {/* VRAM */}
        {showVram ? (
          <div className="flex items-center gap-2 px-4 border-l border-slate-800/30 text-yellow-400">
            <Gpu size={14} className="shrink-0" />
            <span className={variant === 'compact' ? 'hidden' : ''}>VRAM</span>
            <div className="flex items-center ml-auto font-bold text-white gap-1 w-[140px] justify-end">
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
