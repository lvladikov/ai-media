import { useState, useEffect, useRef } from 'react';
import { X, Download, Loader2, Copy, Check, Image, Code, FileText, WrapText } from 'lucide-react';
import { API_BASE_URL } from '../config';
import Prism from 'prismjs';
import 'prismjs/themes/prism-tomorrow.css';
import domToImage from 'dom-to-image';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';

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
import 'prismjs/components/prism-xml-doc';
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

type FileType = 'image' | 'video' | 'audio' | 'pdf' | 'markdown' | 'html' | 'text' | 'unsupported';

const getFileType = (fileName: string): FileType => {
  const ext = fileName.split('.').pop()?.toLowerCase() || '';
  if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'tiff'].includes(ext)) return 'image';
  if (['mp4', 'webm', 'mov', 'mkv'].includes(ext)) return 'video';
  if (['mp3', 'wav', 'flac', 'm4a', 'aac'].includes(ext)) return 'audio';
  if (ext === 'pdf') return 'pdf';
  if (['md', 'markdown'].includes(ext)) return 'markdown';
  if (['html', 'htm', 'xhtml'].includes(ext)) return 'html';
  if (['txt', 'json', 'css', 'js', 'py', 'ts', 'tsx', 'sql', 'go', 'rs', 'rust', 'sh', 'bash', 'java', 'c', 'h', 'cpp', 'hpp', 'cc', 'cs', 'yaml', 'yml', 'xml', 'php', 'rb', 'swift', 'kt', 'kts', 'toml', 'ini', 'dockerfile'].includes(ext)) return 'text';
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
    'xhtml': 'html',
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
  const [viewMode, setViewMode] = useState<'render' | 'code'>('render');

  const fileUrl = `${API_BASE_URL()}/api/files/${filePath}`;
  const fileType = getFileType(fileName);

  // Handle loading state for static/unsupported types
  useEffect(() => {
    if (fileType === 'unsupported' || fileType === 'pdf') {
      setLoading(false);
    }
  }, [fileType]);

  // Reset view mode when file changes
  useEffect(() => {
    setViewMode('render');
  }, [filePath]);

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
          <div className="w-full h-full flex items-center justify-center p-4">
            <img
              src={fileUrl}
              alt={fileName}
              className="max-w-full max-h-full object-contain rounded-lg shadow-lg"
              onLoad={() => setLoading(false)}
              onError={() => { setLoading(false); setError('Failed to load image'); }}
            />
          </div>
        );
      case 'video':
        return (
          <video
            src={fileUrl}
            controls
            className="w-full h-full object-contain rounded-lg"
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
            className="w-full h-full rounded-lg bg-white"
            onLoad={() => setLoading(false)}
            onError={() => { setLoading(false); setError('Failed to load PDF'); }}
          />
        );
      case 'markdown':
        if (viewMode === 'code') {
          return <TextPreview fileName={fileName} url={fileUrl} onLoad={() => setLoading(false)} onError={() => { setLoading(false); setError('Failed to load file'); }} />;
        }
        return <MarkdownPreview url={fileUrl} onLoad={() => setLoading(false)} onError={() => { setLoading(false); setError('Failed to load markdown'); }} />;

      case 'html':
        if (viewMode === 'code') {
          return <TextPreview fileName={fileName} url={fileUrl} onLoad={() => setLoading(false)} onError={() => { setLoading(false); setError('Failed to load file'); }} />;
        }
        return (
          <iframe
            src={fileUrl}
            className="w-full h-full rounded-lg bg-white"
            onLoad={() => setLoading(false)}
            onError={() => { setLoading(false); setError('Failed to load HTML'); }}
            sandbox="allow-same-origin allow-scripts"
          />
        );

      case 'text':
        return <TextPreview fileName={fileName} url={fileUrl} onLoad={() => setLoading(false)} onError={() => { setLoading(false); setError('Failed to load file'); }} />;
      default:
        return (
          <div className="text-center p-8 flex flex-col items-center justify-center h-full">
            <p className="text-lg mb-4">Cannot preview this file type in browser</p>
            <p className="text-secondary mb-6">{fileName}</p>
            <button className="btn-primary flex items-center gap-2 mx-auto" onClick={handleDownload}>
              <Download size={18} />
              Download File
            </button>
          </div>
        );
    }
  };

  const showViewToggle = fileType === 'markdown' || fileType === 'html';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4" onClick={onClose}>
      <div
        className="relative bg-secondary rounded-xl p-4 w-[90vw] h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: '90vw', maxHeight: '90vh' }}
      >
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-0 flex-shrink-0 px-4 pt-4">
          <div className="flex items-center gap-3 pb-2 sm:pb-4 min-w-0">
            <h3 className="text-lg font-semibold truncate flex items-center gap-2">
              {fileType === 'image' && <Image size={18} className="text-purple-400 shrink-0" />}
              {fileType === 'video' && <span className="text-blue-400 shrink-0">🎥</span>}
              {fileType === 'audio' && <span className="text-pink-400 shrink-0">🎵</span>}
              {fileType === 'pdf' && <FileText className="text-red-400 shrink-0" size={18} />}
              {fileType === 'markdown' && <FileText className="text-blue-400 shrink-0" size={18} />}
              {fileType === 'html' && <Code className="text-orange-400 shrink-0" size={18} />}
              {fileType === 'text' && <FileText className="text-secondary shrink-0" size={18} />}
              <span className="truncate">{fileName}</span>
            </h3>
          </div>

          <div className="flex items-end justify-between sm:justify-end gap-4 h-full relative sm:top-[1px]">
            {showViewToggle && (
              <div className="flex items-end">
                <button
                  onClick={() => setViewMode('render')}
                  className={`px-3 md:px-4 py-2 rounded-t-lg text-sm font-medium flex items-center gap-2 transition-all border-t border-x ${viewMode === 'render'
                    ? 'bg-white dark:bg-zinc-950 text-primary border-border border-b-white dark:border-b-zinc-950 z-10'
                    : 'bg-secondary text-secondary border-transparent hover:bg-slate-200 dark:hover:bg-slate-700 hover:text-secondary border-b-border'
                    }`}
                >
                  <FileText size={14} /> <span className="hidden md:inline">Preview</span>
                </button>
                <button
                  onClick={() => setViewMode('code')}
                  className={`px-3 md:px-4 py-2 rounded-t-lg text-sm font-medium flex items-center gap-2 transition-all border-t border-x ml-[-1px] ${viewMode === 'code'
                    ? 'bg-white dark:bg-zinc-950 text-primary border-border border-b-white dark:border-b-zinc-950 z-10'
                    : 'bg-secondary text-secondary border-transparent hover:bg-slate-200 dark:hover:bg-slate-700 hover:text-secondary border-b-border'
                    }`}
                >
                  <Code size={14} /> <span className="hidden md:inline">Code</span>
                </button>
              </div>
            )}

            <div className="flex items-center gap-2 pb-0 sm:pb-3 mb-1 pl-4 sm:border-l border-border/50">
              <button className="btn-secondary p-1.5 h-8 w-8 flex items-center justify-center" onClick={handleDownload} title="Download">
                <Download size={16} />
              </button>
              <button className="btn-secondary p-1.5 h-8 w-8 flex items-center justify-center hover:bg-red-500/20 hover:text-red-600 dark:hover:text-red-200 hover:border-red-500/50" onClick={onClose} title="Close">
                <X size={16} />
              </button>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className={`relative flex-1 min-h-0 overflow-hidden flex flex-col bg-white dark:bg-zinc-950 border border-border ${showViewToggle ? 'rounded-b-lg rounded-tr-lg' : 'rounded-lg mt-4'}`}>
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center z-10 bg-zinc-100 dark:bg-zinc-950">
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

function MarkdownPreview({ url, onLoad, onError }: { url: string; onLoad: () => void; onError: () => void }) {
  const [content, setContent] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

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
    if (content && containerRef.current) {
      Prism.highlightAllUnder(containerRef.current);
    }
  }, [content]);

  if (!content) return null;

  return (
    <div ref={containerRef} className="flex-1 h-full overflow-y-auto scrollbar-themed p-8 prose dark:prose-invert prose-slate max-w-none break-all prose-headings:text-primary prose-p:text-secondary prose-a:text-primary-400 hover:prose-a:text-primary-300 prose-strong:text-primary prose-code:text-primary-300 prose-code:bg-secondary/50 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-pre:bg-primary prose-pre:border prose-pre:border-border [&_pre>code]:bg-transparent [&_pre>code]:p-0 [&_pre>code]:text-inherit">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          code({ node, inline, className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '')
            return !inline && match ? (
              <div className="relative group">
                {/* We could add a copy button here but for now simple rendering is enough */}
                <code className={className} {...props}>
                  {children}
                </code>
              </div>
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            )
          }
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function TextPreview({ fileName, url, onLoad, onError }: { fileName: string; url: string; onLoad: () => void; onError: () => void }) {
  const [content, setContent] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [copiedImage, setCopiedImage] = useState(false);
  const [wordWrap, setWordWrap] = useState(true); // Default ON for vision descriptions
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

      const fullWidth = el.scrollWidth;
      const fullHeight = el.scrollHeight;

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
    <div className="relative group h-full flex flex-col bg-white dark:bg-[#2d2d2d] rounded-b-lg">
      {/* Action buttons */}
      <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity z-10">
        <button
          onClick={() => setWordWrap(!wordWrap)}
          className={`p-2 rounded-lg transition-colors ${wordWrap ? 'bg-brand-500/20 text-brand-400' : 'bg-tertiary hover:bg-tertiary text-primary'}`}
          title={wordWrap ? 'Disable word wrap' : 'Enable word wrap'}
        >
          <WrapText size={16} />
        </button>
        <button
          onClick={handleCopyAsImage}
          className="p-2 bg-tertiary hover:bg-tertiary rounded-lg text-primary"
          title="Copy as image"
        >
          {copiedImage ? <Check size={16} className="text-green-400" /> : <Image size={16} />}
        </button>
        <button
          onClick={handleCopy}
          className="p-2 bg-tertiary hover:bg-tertiary rounded-lg text-primary"
          title="Copy to clipboard"
        >
          {copied ? <Check size={16} className="text-green-400" /> : <Copy size={16} />}
        </button>
      </div>
      <div
        ref={containerRef}
        className="flex-1 overflow-x-auto overflow-y-auto scrollbar-themed w-full h-full relative"
      >
        <pre
          className={`language-${language} border-none m-0 rounded-none bg-transparent !p-4 text-sm`}
          style={{
            fontFamily: 'JetBrains Mono, monospace',
            whiteSpace: wordWrap ? 'pre-wrap' : 'pre',
            wordBreak: wordWrap ? 'break-all' : 'normal',
            overflowWrap: wordWrap ? 'anywhere' : 'normal',
            width: wordWrap ? '100%' : undefined,
            minWidth: wordWrap ? undefined : '100%',
          }}
        >
          <code
            ref={codeRef}
            className={`language-${language} block`}
            style={{
              whiteSpace: wordWrap ? 'pre-wrap' : 'pre',
              wordBreak: wordWrap ? 'break-all' : 'normal',
              overflowWrap: wordWrap ? 'anywhere' : 'normal',
            }}
          >
            {content}
          </code>
        </pre>
      </div>
    </div>
  );
}
