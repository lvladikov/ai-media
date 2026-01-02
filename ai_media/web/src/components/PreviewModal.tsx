import { useState, useEffect, useRef } from 'react';
import { X, Download, Loader2, Copy, Check, Image } from 'lucide-react';
import Prism from 'prismjs';
import 'prismjs/themes/prism-tomorrow.css';
import domToImage from 'dom-to-image';

// Load Core Dependencies
import 'prismjs/components/prism-clike';
import 'prismjs/components/prism-markup';
import 'prismjs/components/prism-markup-templating';

// Load languages
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-markdown';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-css';
import 'prismjs/components/prism-jsx';
import 'prismjs/components/prism-tsx';
import 'prismjs/components/prism-sql';
import 'prismjs/components/prism-go';
import 'prismjs/components/prism-rust';
import 'prismjs/components/prism-java';
import 'prismjs/components/prism-c';
import 'prismjs/components/prism-cpp';
import 'prismjs/components/prism-csharp';
import 'prismjs/components/prism-yaml';
import 'prismjs/components/prism-xml-doc'; // Using xml-doc or just xml/markup usually
import 'prismjs/components/prism-php';
import 'prismjs/components/prism-ruby';
import 'prismjs/components/prism-swift';
import 'prismjs/components/prism-kotlin';
import 'prismjs/components/prism-docker';
import 'prismjs/components/prism-toml';
import 'prismjs/components/prism-ini';

interface PreviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  filePath: string;
  fileName: string;
}

type FileType = 'image' | 'video' | 'audio' | 'pdf' | 'text' | 'unsupported';

const getFileType = (fileName: string): FileType => {
  const ext = fileName.split('.').pop()?.toLowerCase() || '';
  if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'tiff'].includes(ext)) return 'image';
  if (['mp4', 'webm', 'mov', 'mkv'].includes(ext)) return 'video';
  if (['mp3', 'wav', 'flac', 'm4a', 'aac'].includes(ext)) return 'audio';
  if (ext === 'pdf') return 'pdf';
  if (['md', 'txt', 'json', 'html', 'css', 'js', 'py', 'ts', 'tsx', 'sql', 'go', 'rs', 'rust', 'sh', 'bash', 'java', 'c', 'h', 'cpp', 'hpp', 'cc', 'cs', 'yaml', 'yml', 'xml', 'php', 'rb', 'swift', 'kt', 'kts', 'toml', 'ini', 'dockerfile'].includes(ext)) return 'text';
  return 'unsupported';
};

const getLanguage = (fileName: string) => {
  const ext = fileName.split('.').pop()?.toLowerCase() || '';
  const map: Record<string, string> = {
    'py': 'python',
    'md': 'markdown',
    'json': 'json',
    'js': 'javascript',
    'ts': 'typescript',
    'tsx': 'tsx',
    'jsx': 'jsx',
    'css': 'css',
    'html': 'html',
    'bash': 'bash',
    'sh': 'bash',
    'sql': 'sql',
    'go': 'go',
    'rust': 'rust',
    'rs': 'rust',
    'java': 'java',
    'c': 'c',
    'h': 'c',
    'cpp': 'cpp',
    'hpp': 'cpp',
    'cc': 'cpp',
    'cs': 'csharp',
    'yaml': 'yaml',
    'yml': 'yaml',
    'xml': 'xml',
    'php': 'php',
    'rb': 'ruby',
    'swift': 'swift',
    'kt': 'kotlin',
    'kts': 'kotlin',
    'toml': 'toml',
    'ini': 'ini',
    'dockerfile': 'docker',
  };
  return map[ext] || 'none';
};

export function PreviewModal({ isOpen, onClose, filePath, fileName }: PreviewModalProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const fileUrl = `http://localhost:8000/api/files/${filePath}`;
  const fileType = getFileType(fileName);

  // Handle loading state for static/unsupported types
  useEffect(() => {
    if (fileType === 'unsupported' || fileType === 'pdf') {
        setLoading(false);
    }
  }, [fileType]);

  if (!isOpen) return null;

  const handleDownload = () => {
    const a = document.createElement('a');
    a.href = `${fileUrl}?download=true`;
    a.download = fileName;
    a.click();
  };

  const renderPreview = () => {
    switch (fileType) {
      case 'image':
        return (
          <img
            src={fileUrl}
            alt={fileName}
            className="max-w-full max-h-full object-contain rounded-lg mx-auto"
            onLoad={() => setLoading(false)}
            onError={() => { setLoading(false); setError('Failed to load image'); }}
          />
        );
      case 'video':
        return (
          <video
            src={fileUrl}
            controls
            className="max-w-full max-h-full rounded-lg mx-auto"
            onLoadedData={() => setLoading(false)}
            onError={() => { setLoading(false); setError('Failed to load video'); }}
          />
        );
      case 'audio':
        return (
          <div className="flex flex-col items-center justify-center gap-4 p-8 h-full">
            <div className="w-32 h-32 rounded-full bg-gradient-to-br from-primary-500 to-purple-600 flex items-center justify-center">
              <span className="text-4xl">🎵</span>
            </div>
            <p className="text-lg font-medium">{fileName}</p>
            <audio
              src={fileUrl}
              controls
              className="w-full max-w-md"
              onLoadedData={() => setLoading(false)}
              onError={() => { setLoading(false); setError('Failed to load audio'); }}
            />
          </div>
        );
      case 'pdf':
        return (
          <iframe
            src={fileUrl}
            className="w-full h-full rounded-lg"
            onLoad={() => setLoading(false)}
            onError={() => { setLoading(false); setError('Failed to load PDF'); }}
          />
        );
      case 'text':
        return <TextPreview fileName={fileName} url={fileUrl} onLoad={() => setLoading(false)} onError={() => { setLoading(false); setError('Failed to load file'); }} />;
      default:
        return (
          <div className="text-center p-8 flex flex-col items-center justify-center h-full">
            <p className="text-lg mb-4">Cannot preview this file type</p>
            <p className="text-slate-400 mb-6">{fileName}</p>
            <button className="btn-primary flex items-center gap-2 mx-auto" onClick={handleDownload}>
              <Download size={18} />
              Download File
            </button>
          </div>
        );
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4" onClick={onClose}>
      <div 
        className="relative bg-slate-800 rounded-xl p-4 w-[90vw] h-[90vh] flex flex-col" 
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: '90vw', maxHeight: '90vh' }}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4 flex-shrink-0">
          <h3 className="text-lg font-semibold truncate max-w-md">{fileName}</h3>
          <div className="flex items-center gap-2">
            <button className="btn-secondary p-2" onClick={handleDownload} title="Download">
              <Download size={18} />
            </button>
            <button className="btn-secondary p-2" onClick={onClose} title="Close">
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="relative flex-1 min-h-0 overflow-auto scrollbar-themed">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center">
              <Loader2 className="animate-spin" size={32} />
            </div>
          )}
          {error ? (
            <div className="text-center text-red-400 p-8">{error}</div>
          ) : (
            renderPreview()
          )}
        </div>
      </div>
    </div>
  );
}

function TextPreview({ fileName, url, onLoad, onError }: { fileName: string; url: string; onLoad: () => void; onError: () => void }) {
  const [content, setContent] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [copiedImage, setCopiedImage] = useState(false);
  const codeRef = useRef<HTMLElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const language = getLanguage(fileName);

  useEffect(() => {
    fetch(url)
      .then((res) => res.text())
      .then((text) => { 
        setContent(text); 
        onLoad();
      })
      .catch(() => onError());
  }, [url]);

  useEffect(() => {
    if (content && codeRef.current) {
      Prism.highlightElement(codeRef.current);
    }
  }, [content]);

  const handleCopy = () => {
    if (!content) return;
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCopyAsImage = async () => {
    if (!containerRef.current) return;
    
    try {
      const el = containerRef.current;
      const scale = 2;
      
      // Get the full scroll dimensions (not just visible area)
      const fullWidth = el.scrollWidth;
      const fullHeight = el.scrollHeight;
      
      // Temporarily adjust styles to capture full content
      const originalOverflow = el.style.overflow;
      const originalHeight = el.style.height;
      const originalMaxHeight = el.style.maxHeight;
      
      el.style.overflow = 'visible';
      el.style.height = 'auto';
      el.style.maxHeight = 'none';
      
      const blob = await domToImage.toBlob(el, { 
        width: fullWidth * scale,
        height: fullHeight * scale,
        style: {
          transform: `scale(${scale})`,
          transformOrigin: "top left",
          overflow: "visible",
          height: "auto",
          maxHeight: "none",
        },
      });
      
      // Restore original styles
      el.style.overflow = originalOverflow;
      el.style.height = originalHeight;
      el.style.maxHeight = originalMaxHeight;
      
      if (blob) {
        await navigator.clipboard.write([
          new ClipboardItem({ 'image/png': blob })
        ]);
        setCopiedImage(true);
        setTimeout(() => setCopiedImage(false), 2000);
      }
    } catch (err) {
      console.error('Failed to copy as image:', err);
    }
  };

  if (!content) return null;

  return (
    <div className="relative group h-full flex flex-col">
      {/* Action buttons */}
      <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity z-10">
        <button 
          onClick={handleCopyAsImage}
          className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-slate-200"
          title="Copy as image"
        >
          {copiedImage ? <Check size={16} className="text-green-400" /> : <Image size={16} />}
        </button>
        <button 
          onClick={handleCopy}
          className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-slate-200"
          title="Copy to clipboard"
        >
          {copied ? <Check size={16} className="text-green-400" /> : <Copy size={16} />}
        </button>
      </div>
      <div 
        ref={containerRef}
        className="flex-1 rounded-lg border border-slate-700 bg-[#2d2d2d] overflow-auto scrollbar-themed"
      >
         <pre 
           className={`language-${language} border-none m-0 rounded-none bg-transparent !p-4 !m-0 text-sm`}
           style={{ minWidth: 'max-content' }}
         >
          <code ref={codeRef} className={`language-${language}`}>
            {content}
          </code>
        </pre>
      </div>
    </div>
  );
}
