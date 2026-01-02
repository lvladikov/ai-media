import { useState, useEffect } from 'react';
import { Save, FileText, Download, Loader2 } from 'lucide-react';

interface SaveModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (options: SaveOptions) => void;
  lastMessage: string | null;
  fullChat: string;
  suggestedFilename?: string;
}

export interface SaveOptions {
  scope: 'last' | 'full';
  format: string;
  filename: string;
  content: string;
}

const FORMATS = [
  { id: 'md', label: 'Markdown', ext: '.md' },
  { id: 'txt', label: 'Plain Text', ext: '.txt' },
  { id: 'html', label: 'HTML', ext: '.html' },
  { id: 'pdf', label: 'PDF Document', ext: '.pdf' },
  { id: 'docx', label: 'MS Word (DOCX)', ext: '.docx' },
  { id: 'rtf', label: 'Rich Text (RTF)', ext: '.rtf' },
  { id: 'json', label: 'JSON Data', ext: '.json' },
];

export function SaveModal({ isOpen, onClose, onSave, lastMessage, fullChat, suggestedFilename }: SaveModalProps) {
  const [scope, setScope] = useState<'last' | 'full'>('last');
  const [format, setFormat] = useState('md');
  const [filename, setFilename] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  // Initialize defaults when modal opens
  useEffect(() => {
    if (isOpen) {
      // Set initial scope based on message availability
      setScope(lastMessage ? 'last' : 'full');
      
      // Set initial filename based on suggestion and current format
      const baseName = suggestedFilename || `chat_export_${Date.now()}`;
      const ext = FORMATS.find(f => f.id === format)?.ext || '.md';
      const cleanBase = baseName.includes('.') ? baseName.split('.').slice(0, -1).join('.') : baseName;
      setFilename(cleanBase + ext);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  // Update filename extension only when format changes
  useEffect(() => {
    if (isOpen) {
      const ext = FORMATS.find(f => f.id === format)?.ext || '.md';
      setFilename(prev => {
        const base = prev.includes('.') ? prev.split('.').slice(0, -1).join('.') : prev;
        return base + ext;
      });
    }
  }, [format, isOpen]);

  if (!isOpen) return null;

  const handleSave = async () => {
    setIsSaving(true);
    const content = scope === 'last' ? (lastMessage || '') : fullChat;
    
    // Ensure filename matches extension
    const ext = FORMATS.find(f => f.id === format)?.ext || '.md';
    let finalFilename = filename;
    if (!finalFilename.toLowerCase().endsWith(ext)) {
      const base = finalFilename.includes('.') ? finalFilename.split('.').slice(0, -1).join('.') : finalFilename;
      finalFilename = base + ext;
    }

    try {
      await onSave({
        scope,
        format,
        filename: finalFilename,
        content
      });
      onClose();
    } catch (err) {
      console.error('Save failed:', err);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[60] flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-secondary border border-border rounded-xl shadow-2xl w-full max-w-md p-6 animate-in fade-in zoom-in-95 duration-200" onClick={e => e.stopPropagation()}>
        <h2 className="text-xl font-bold text-primary mb-6 flex items-center gap-2">
          <Save className="text-brand-400" />
          Save Chat As...
        </h2>
        
        <div className="space-y-6">
          {/* Scope Selection */}
          <div>
            <label className="block text-xs font-semibold text-tertiary uppercase tracking-wider mb-2">What to Save</label>
            <div className="grid grid-cols-2 gap-2">
              <button 
                className={`px-4 py-2 rounded-lg border text-sm font-medium transition-all ${scope === 'last' ? 'bg-brand-500/10 border-brand-500 text-brand-400' : 'bg-primary border-border text-tertiary hover:border-slate-600'}`}
                onClick={() => setScope('last')}
                disabled={!lastMessage}
              >
                Last Message
              </button>
              <button 
                className={`px-4 py-2 rounded-lg border text-sm font-medium transition-all ${scope === 'full' ? 'bg-brand-500/10 border-brand-500 text-brand-400' : 'bg-primary border-border text-tertiary hover:border-slate-600'}`}
                onClick={() => setScope('full')}
              >
                Full Chat History
              </button>
            </div>
          </div>

          {/* Format Selection */}
          <div>
            <label className="block text-xs font-semibold text-tertiary uppercase tracking-wider mb-2">Format</label>
            <div className="grid grid-cols-3 gap-2">
              {FORMATS.map(f => (
                <button 
                  key={f.id}
                  className={`px-3 py-2 rounded-lg border text-xs font-medium transition-all ${format === f.id ? 'bg-brand-500/10 border-brand-500 text-brand-400' : 'bg-primary border-border text-tertiary hover:border-slate-600'}`}
                  onClick={() => setFormat(f.id)}
                >
                  {f.label}
                </button>
              ))}
            </div>
          </div>

          {/* Filename Input */}
          <div>
            <label className="block text-xs font-semibold text-tertiary uppercase tracking-wider mb-2">Filename</label>
            <div className="relative">
              <input 
                className="input w-full pr-10" 
                value={filename}
                onChange={e => setFilename(e.target.value)}
                placeholder="filename.ext"
              />
              <FileText className="absolute right-3 top-2.5 text-slate-500" size={18} />
            </div>
          </div>
          
          <div className="flex justify-end gap-2 mt-8">
            <button className="btn-secondary" onClick={onClose}>Cancel</button>
            <button 
              className="btn-primary flex items-center gap-2" 
              onClick={handleSave}
              disabled={isSaving || !filename.trim()}
            >
              {isSaving ? <Loader2 size={18} className="animate-spin" /> : <Download size={18} />}
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
