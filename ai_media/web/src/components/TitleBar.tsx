export function TitleBar() {
  // Only render in Electron
  if (!(window as any).electronAPI?.isElectron) return null;

  return (
    <div 
      className="h-10 w-full bg-secondary border-b border-border flex items-center justify-center text-xs text-slate-500 dark:text-slate-400 select-none shrink-0"
      style={{ WebkitAppRegion: 'drag' } as any}
    >
      <span className="font-medium">AI-Media</span>
    </div>
  );
}
