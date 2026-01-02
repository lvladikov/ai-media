export function TitleBar() {
  // Only render in Electron
  if (!(window as any).electronAPI?.isElectron) return null;

  return (
    <div 
      className="h-10 w-full bg-slate-950/50 border-b border-white/5 flex items-center justify-center text-xs text-slate-500 select-none shrink-0"
      style={{ WebkitAppRegion: 'drag' } as any}
    >
      <span className="font-medium opacity-50">AI-Media Studio</span>
    </div>
  );
}
