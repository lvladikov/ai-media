import { type ReactNode, useState } from 'react';
import { Info } from 'lucide-react';

interface TooltipProps {
  content: string;
  children?: ReactNode;
  align?: 'center' | 'left' | 'right';
}

export function Tooltip({ content, children, align = 'center' }: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);

  const alignClasses = {
    center: 'left-1/2 -translate-x-1/2',
    left: 'left-0',
    right: 'right-0',
  };

  const arrowClasses = {
    center: 'left-1/2 -translate-x-1/2',
    left: 'left-3',
    right: 'right-3',
  };

  return (
    <div className="relative inline-flex items-center ml-2">
      <div
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        className="cursor-help text-slate-400 hover:text-primary-400 transition-colors"
      >
        {children || <Info size={14} />}
      </div>
      
      {isVisible && (
        <div className={`absolute bottom-full mb-2 w-48 p-2 bg-slate-900 border border-slate-700 rounded shadow-xl text-xs text-slate-200 z-50 pointer-events-none ${alignClasses[align]}`}>
          {content}
          <div className={`absolute top-full border-4 border-transparent border-t-slate-900 ${arrowClasses[align]}`} />
        </div>
      )}
    </div>
  );
}
