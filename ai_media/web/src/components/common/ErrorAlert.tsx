import { AlertTriangle, X } from 'lucide-react';

interface ErrorAlertProps {
  error: string | null;
  onDismiss: () => void;
  title?: string;
}

export function ErrorAlert({ error, onDismiss, title = "Generation Failed" }: ErrorAlertProps) {
  if (!error) return null;

  return (
    <div className="mt-6 p-4 bg-red-50 dark:bg-red-500/20 border border-red-200 dark:border-red-500/50 rounded-lg text-red-800 dark:text-red-200 flex items-start gap-2 relative group animate-in fade-in slide-in-from-top-2 duration-200">
       <AlertTriangle className="shrink-0 mt-0.5" size={18} />
       <div className="flex-1 pr-6">
         <p className="font-semibold">{title}</p>
         <p className="text-sm opacity-90 break-words">{error}</p>
       </div>
       <button 
         onClick={onDismiss}
         className="absolute top-4 right-4 text-red-600 dark:text-red-300 hover:text-red-900 dark:hover:text-red-100 transition-colors p-1 rounded-full hover:bg-red-200 dark:hover:bg-red-500/20"
         aria-label="Dismiss error"
       >
         <X size={16} />
       </button>
    </div>
  );
}
