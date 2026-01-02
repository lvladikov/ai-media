import { useState, type ReactNode } from 'react';

interface ValidationTooltipProps {
  error?: string | null | boolean; // If string, shows it. If true, shows generic. If falsy, no tooltip.
  children: ReactNode;
  className?: string; 
}

export function ValidationTooltip({ error, children, className = "" }: ValidationTooltipProps) {
  const [show, setShow] = useState(false);

  // If no error message (or false/null), just render children wrapper
  // We still render wrapper to preserve layout consistency if needed, 
  // but if you prefer to only wrap when error exists, that's fine too.
  // For simplicity and z-index handling, always wrapping is safer for "disabled" button layout.
  
  if (!error) {
     return <div className={className}>{children}</div>;
  }

  const errorMessage = typeof error === 'string' ? error : "Please fill in required fields";

  return (
    <div 
      className={`relative ${className}`}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 bg-slate-800 border border-slate-600 text-slate-200 text-xs rounded-md shadow-xl whitespace-nowrap z-50 pointer-events-none animate-in fade-in zoom-in-95 duration-200">
          {errorMessage}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800" />
        </div>
      )}
    </div>
  );
}
