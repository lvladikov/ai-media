import React, { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../store';
import { useModels } from '../hooks/useApi';
import { API_BASE_URL as API_BASE } from '../config';
import { MessageSquare, Send, Loader2, LogOut, FileText, Save, Globe, ChevronRight, ChevronLeft, ChevronDown, Copy, Check, Trash2, Image, AlertCircle, RefreshCw, HelpCircle } from 'lucide-react';
import domToImage from 'dom-to-image';
import { NumberInput } from './common/NumberInput';
import { MarkdownWithAnsi, MarkdownWithAnsiNoHtml } from './common/AnsiRenderer';
import { usePromptTriggers } from '../hooks/usePromptTriggers';
import { SaveModal } from './SaveModal';
import type { SaveOptions } from './SaveModal';
import { formatDuration } from '../utils/formatTime';
import { PROMPTS } from '../data/prompts';
import { getDynamicRam } from '../utils/modelResources';

// Clipboard CSS mapping for rich-text copy (inline styles for external apps)
const TAILWIND_TO_CSS: Record<string, string> = {
  'text-tertiary': 'color: #475569;',
  'text-red-400': 'color: #f87171;',
  'text-green-400': 'color: #4ade80;',
  'text-yellow-400': 'color: #facc15;',
  'text-blue-400': 'color: #60a5fa;',
  'text-purple-400': 'color: #c084fc;',
  'text-cyan-400': 'color: #22d3ee;',
  'text-primary': 'color: #e2e8f0;',
  'text-secondary': 'color: #94a3b8;',
  'text-red-300': 'color: #fca5a5;',
  'text-green-300': 'color: #86efac;',
  'text-yellow-300': 'color: #fde047;',
  'text-blue-300': 'color: #93c5fd;',
  'text-purple-300': 'color: #d8b4fe;',
  'text-cyan-300': 'color: #67e8f9;',
  'font-bold': 'font-weight: bold;',
  'italic': 'font-style: italic;',
  'underline': 'text-decoration: underline;',
  'bg-secondary': 'background-color: #1e293b;',
  'bg-red-500/20': 'background-color: rgba(239, 68, 68, 0.2);',
  'bg-green-500/20': 'background-color: rgba(34, 197, 94, 0.2);',
  'bg-yellow-500/20': 'background-color: rgba(234, 179, 8, 0.2);',
  'bg-blue-500/20': 'background-color: rgba(59, 130, 246, 0.2);',
  'bg-purple-500/20': 'background-color: rgba(168, 85, 247, 0.2);',
  'bg-cyan-500/20': 'background-color: rgba(6, 182, 212, 0.2);',
  'bg-slate-500/20': 'background-color: rgba(100, 116, 139, 0.2);',
  'bg-slate-400/20': 'background-color: rgba(148, 163, 184, 0.2);',
  'bg-red-300/20': 'background-color: rgba(252, 165, 165, 0.2);',
  'bg-green-300/20': 'background-color: rgba(134, 239, 172, 0.2);',
  'bg-yellow-300/20': 'background-color: rgba(253, 224, 71, 0.2);',
  'bg-blue-300/20': 'background-color: rgba(147, 197, 253, 0.2);',
  'bg-purple-300/20': 'background-color: rgba(216, 180, 254, 0.2);',
  'bg-cyan-300/20': 'background-color: rgba(103, 232, 249, 0.2);',
  'bg-slate-100/20': 'background-color: rgba(241, 245, 249, 0.2);',
  'px-1': 'padding-left: 0.25rem; padding-right: 0.25rem;',
  'rounded': 'border-radius: 0.25rem;',
};






// Local RAM constant removed - using shared utility

// Display names matching CLI (vram field removed, now calculated dynamically)
const MODEL_DISPLAY_INFO: Record<string, { label: string }> = {
  'deepseek-r1-qwen-7b': { label: 'DeepSeek R1 Qwen 7B (Reasoning)' },
  'deepseek-r1-qwen-14b': { label: 'DeepSeek R1 Qwen 14B (Reasoning)' },
  'deepseek-r1-qwen-32b': { label: 'DeepSeek R1 Qwen 32B (Reasoning)' },
  'deepseek-r1-llama-8b': { label: 'DeepSeek R1 Llama 8B (Reasoning)' },
  'deepseek-r1-llama-70b': { label: 'DeepSeek R1 Llama 70B (Reasoning)' },
  'llama-3.1-8b': { label: 'Llama 3.1 8B (Fast & Stable)' },
  'mistral-nemo-12b': { label: 'Mistral Nemo 12B' },
  'qwen3-8b': { label: 'Qwen 3 8B (Reasoning)' },
  'qwen3-14b': { label: 'Qwen 3 14B (Reasoning)' },
  'qwen3-opus-4.5-8b': { label: 'Qwen 3 Opus 4.5 Distill (8B)' },
  'qwen3-opus-4.5-14b': { label: 'Qwen 3 Opus 4.5 Distill (14B)' },
  'qwen3-gpt-5.2-8b': { label: 'Qwen 3 GPT-5.2 Distill (8B)' },
  'qwen3-gpt-5.2-14b': { label: 'Qwen 3 GPT-5.2 Distill (14B)' },
  'qwen3-coder-30b': { label: 'Qwen3 Coder 30B (MoE, 3.3B active)' },
  'qwen-coder-32b': { label: 'Qwen 2.5 Coder 32B' },
  'qwen-coder-14b': { label: 'Qwen 2.5 Coder 14B' },
  'qwen-coder-7b': { label: 'Qwen 2.5 Coder 7B' },
  'qwen-vl': { label: 'Qwen3-VL 8B (Vision)' },
  'qwen3-vl-4b': { label: 'Qwen3-VL 4B (Vision)' },
  'qwen3-vl-2b': { label: 'Qwen3-VL 2B (Vision)' },
};

const MODEL_ORDER = [
  'deepseek-r1-qwen-7b', 'deepseek-r1-qwen-14b', 'deepseek-r1-qwen-32b',
  'deepseek-r1-llama-8b', 'deepseek-r1-llama-70b',
  'llama-3.1-8b', 'mistral-nemo-12b', 'qwen3-14b', 'qwen3-8b',
  'qwen3-opus-4.5-14b', 'qwen3-opus-4.5-8b', 'qwen3-gpt-5.2-14b', 'qwen3-gpt-5.2-8b',
  'qwen3-coder-30b', 'qwen-coder-32b', 'qwen-coder-14b', 'qwen-coder-7b',
  'qwen-vl', 'qwen3-vl-4b', 'qwen3-vl-2b'
];

import { TranslateOptions } from './common/TranslateOptions';

const ReasoningAccordion = ({ reasoning }: { reasoning: string }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="border-l-2 border-border/50 pl-3 my-1 bg-secondary/10 py-1 pr-2 rounded-r">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 font-semibold text-xs text-slate-600 dark:text-slate-400 mb-0.5 hover:text-slate-900 dark:hover:text-slate-200 transition-colors w-full text-left"
      >
        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span>💭 Reasoning</span>
      </button>

      {isOpen && (
        <div className="prose dark:prose-invert max-w-none text-secondary/90 leading-tight [&>p]:!text-xs [&>p]:italic [&>p]:my-0.5 [&>pre]:not-italic [&>pre]:my-1 mt-2 animate-in fade-in slide-in-from-top-1 duration-200">
          <MarkdownWithAnsiNoHtml>{reasoning}</MarkdownWithAnsiNoHtml>
        </div>
      )}
    </div>
  );
};

// Translation accordion - shows translated input or original English
const TranslationAccordion = ({ label, content, icon = "🌐" }: { label: string; content: string; icon?: string }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="border-l-2 border-cyan-600/40 pl-3 my-1 bg-cyan-900/10 py-1 pr-2 rounded-r">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 font-semibold text-[11px] text-cyan-600 dark:text-cyan-400 mb-0.5 hover:text-cyan-800 dark:hover:text-cyan-200 transition-colors w-full text-left"
      >
        {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span>{icon} {label}</span>
      </button>

      {isOpen && (
        <div className="prose dark:prose-invert max-w-none text-secondary/80 leading-tight text-xs italic mt-1 animate-in fade-in slide-in-from-top-1 duration-200">
          {content}
        </div>
      )}
    </div>
  );
};

const FileContextAccordion = ({ filename, content }: { filename: string; content: string }) => {
  const [isOpen, setIsOpen] = useState(false);
  const lines = content.split('\n');
  const previewLines = lines.slice(0, 5).join('\n');
  const hasMore = lines.length > 5;

  return (
    <div className="border-l-2 border-blue-600/50 pl-3 my-1 bg-secondary/10 py-1 pr-2 rounded-r">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 font-semibold text-xs text-slate-600 dark:text-slate-400 mb-0.5 hover:text-slate-900 dark:hover:text-slate-200 transition-colors w-full text-left"
      >
        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span>📄 File: {filename}</span>
        {hasMore && !isOpen && <span className="text-tertiary ml-1">({lines.length} lines)</span>}
      </button>

      <div className={`prose dark:prose-invert max-w-none text-secondary/90 leading-tight text-xs font-mono mt-2 ${isOpen ? '' : 'max-h-[120px] overflow-hidden'}`}>
        <pre className="!bg-transparent !p-0 !m-0 whitespace-pre-wrap">
          {isOpen ? content : previewLines}
          {!isOpen && hasMore && <span className="text-tertiary italic">...</span>}
        </pre>
      </div>
    </div>
  );
};

const ThinkingMessage = React.memo(({ content, originalContent, reasoning, thinkingTime }: { content: string, originalContent?: string, reasoning?: string, thinkingTime?: string }) => {
  const [copied, setCopied] = useState(false);
  const [copiedImage, setCopiedImage] = useState(false);
  const answerRef = useRef<HTMLDivElement>(null);

  const copyRichText = async () => {
    if (!answerRef.current) return;

    try {
      // Create a temporary container to clone and process the HTML
      const tempDiv = document.createElement('div');
      tempDiv.innerHTML = answerRef.current.innerHTML;

      // 1. Inline styles for elements with our specific Tailwind classes
      const allElements = tempDiv.querySelectorAll('*');
      allElements.forEach(el => {
        const classes = el.className.split(' ');
        let inlineStyle = '';
        classes.forEach(cls => {
          if (TAILWIND_TO_CSS[cls]) {
            inlineStyle += TAILWIND_TO_CSS[cls] + ' ';
          }
        });
        if (inlineStyle) {
          const currentStyle = el.getAttribute('style') || '';
          el.setAttribute('style', currentStyle + inlineStyle);
        }
      });

      // 2. Special handling for tables to ensure they look okay in Word
      const tables = tempDiv.querySelectorAll('table');
      tables.forEach(t => {
        t.setAttribute('style', 'border-collapse: collapse; width: 100%; border: 1px solid #444; margin-bottom: 1em;');
      });
      const ths = tempDiv.querySelectorAll('th, td');
      ths.forEach(th => {
        th.setAttribute('style', (th.getAttribute('style') || '') + ' border: 1px solid #444; padding: 8px; text-align: left;');
      });

      const html = tempDiv.innerHTML;
      const text = answerRef.current.innerText;

      // Detect OS - Windows needs execCommand, Mac works with ClipboardItem
      const isWindows = navigator.platform.toLowerCase().includes('win') ||
        navigator.userAgent.toLowerCase().includes('windows');

      if (!isWindows && navigator.clipboard && typeof ClipboardItem !== 'undefined') {
        // Mac/Linux: Use modern clipboard API
        const htmlBlob = new Blob([html], { type: 'text/html' });
        const textBlob = new Blob([text], { type: 'text/plain' });

        await navigator.clipboard.write([
          new ClipboardItem({
            'text/html': htmlBlob,
            'text/plain': textBlob,
          })
        ]);

        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } else {
        // Windows: Use execCommand with selection (preserves HTML formatting)
        const selection = window.getSelection();
        const range = document.createRange();

        // Create temporary element with the styled HTML
        const clipboardDiv = document.createElement('div');
        clipboardDiv.innerHTML = html;
        clipboardDiv.style.position = 'fixed';
        clipboardDiv.style.left = '-9999px';
        clipboardDiv.style.whiteSpace = 'pre-wrap';
        document.body.appendChild(clipboardDiv);

        range.selectNodeContents(clipboardDiv);
        selection?.removeAllRanges();
        selection?.addRange(range);

        const success = document.execCommand('copy');

        document.body.removeChild(clipboardDiv);
        selection?.removeAllRanges();

        if (success) {
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        } else {
          throw new Error('execCommand copy failed');
        }
      }
    } catch (err) {
      console.error('Failed to copy rich text:', err);
      // Final fallback: plain text
      navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };
  const copyAsImage = async () => {
    if (!answerRef.current) return;

    try {
      // Capture the DOM as a PNG blob
      const scale = 2; // Render at 2x resolution
      const blob = await domToImage.toBlob(answerRef.current, {
        width: answerRef.current.clientWidth * scale,
        height: answerRef.current.clientHeight * scale,
        style: {
          transform: `scale(${scale})`,
          transformOrigin: "top left",
        },
      });

      if (blob) {
        await navigator.clipboard.write([
          new ClipboardItem({ 'image/png': blob })
        ]);
        setCopiedImage(true);
        setTimeout(() => setCopiedImage(false), 2000);
      }
    } catch (err) {
      console.error('Failed to copy image:', err);
    }
  };

  const renderContent = (markdown: string) => (
    <div ref={answerRef} className="prose dark:prose-invert prose-sm max-w-none">
      <MarkdownWithAnsi>{markdown}</MarkdownWithAnsi>
    </div>
  );

  // Case 1: Structured reasoning provided
  if (reasoning) {
    return (
      <div className="font-sans font-normal whitespace-pre-wrap">
        <ReasoningAccordion reasoning={reasoning} />
        {thinkingTime && (
          <div className="mt-2 mb-2 text-xs text-slate-500 dark:text-slate-400 italic">
            Thought for {thinkingTime}
          </div>
        )}
        {originalContent && (
          <TranslationAccordion
            label="Original (English response before translation)"
            content={originalContent}
          />
        )}
        <div className="mt-4">
          <div className="flex items-center justify-between mb-1">
            <div className="font-bold text-primary italic">Answer:</div>
            <div className="flex gap-1">
              <button
                onClick={copyAsImage}
                className="p-1 hover:bg-tertiary/50 rounded transition-colors text-secondary hover:text-primary"
                title="Copy answer as image"
              >
                {copiedImage ? <Check size={14} className="text-green-400" /> : <Image size={14} />}
              </button>
              <button
                onClick={copyRichText}
                className="p-1 hover:bg-tertiary/50 rounded transition-colors text-secondary hover:text-primary"
                title="Copy answer with formatting"
              >
                {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
              </button>
            </div>
          </div>
          {renderContent(content)}
        </div>
      </div>
    );
  }

  // Case 2: Embedded reasoning in content
  if (content.includes('</think>')) {
    const parts = content.split('</think>');
    const r = parts[0].replace('<think>', '').trim();
    const a = parts.slice(1).join('</think>').trim();

    return (
      <div className="font-sans font-normal whitespace-pre-wrap">
        <ReasoningAccordion reasoning={r} />
        {thinkingTime && (
          <div className="mt-2 mb-2 text-xs text-slate-500 dark:text-slate-400 italic">
            Thought for {thinkingTime}
          </div>
        )}
        {a && (
          <div className="mt-4">
            <div className="flex items-center justify-between mb-1">
              <div className="font-bold text-primary italic">Answer:</div>
              <div className="flex gap-1">
                <button
                  onClick={copyAsImage}
                  className="p-1 hover:bg-tertiary/50 rounded transition-colors text-secondary hover:text-primary"
                  title="Copy answer as image"
                >
                  {copiedImage ? <Check size={14} className="text-green-400" /> : <Image size={14} />}
                </button>
                <button
                  onClick={copyRichText}
                  className="p-1 hover:bg-tertiary/50 rounded transition-colors text-secondary hover:text-primary"
                  title="Copy answer with formatting"
                >
                  {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
                </button>
              </div>
            </div>
            {renderContent(a)}
          </div>
        )}
      </div>
    );
  }

  // Case 3: Standard content (Direct Answer)

  return (
    <div className="mt-2 group relative">
      <div className="absolute -top-3 right-0 opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
        <button
          onClick={copyAsImage}
          className="p-1 bg-secondary/80 hover:bg-tertiary rounded transition-colors text-secondary hover:text-primary border border-border/50"
          title="Copy as image"
        >
          {copiedImage ? <Check size={14} className="text-green-400" /> : <Image size={14} />}
        </button>
        <button
          onClick={copyRichText}
          className="p-1 bg-secondary/80 hover:bg-tertiary rounded transition-colors text-secondary hover:text-primary border border-border/50"
          title="Copy formatted text"
        >
          {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
        </button>
      </div>
      {/* Show timing for all responses */}
      {thinkingTime && (
        <div className="mb-2 text-xs text-slate-500 dark:text-slate-400 italic">
          {content.includes('🌍') ? `Searched for ${thinkingTime}` : `Responded in ${thinkingTime}`}
        </div>
      )}
      {renderContent(content)}
      {/* Show "Original" accordion when output was translated */}
      {originalContent && (
        <TranslationAccordion
          label="Original (English response before translation)"
          content={originalContent}
        />
      )}
    </div>
  );
});


const UserMessage = ({ content, translatedInput }: { content: string; translatedInput?: string }) => {
  // Check if message contains file context (added via Read File button)
  const fileContextMatch = content.match(/\[File Context: ([^\]]+)\]/);
  if (fileContextMatch) {
    const filename = fileContextMatch[1];
    // Extract file content (everything after the header line, stripping markdown code fences)
    let fileContent = content.replace(/```markdown\n?\[File Context:[^\]]+\]\n?```\n?/, '').replace(/```\n?/g, '').trim();

    return (
      <div className="prose dark:prose-invert prose-sm max-w-none">
        <FileContextAccordion filename={filename} content={fileContent} />
      </div>
    );
  }

  // Check if message is a slash command
  if (content.trim().startsWith('/')) {
    const parts = content.trim().split(' ');
    const command = parts[0];
    const args = parts.slice(1).join(' ');

    return (
      <div className="prose dark:prose-invert prose-sm max-w-none">
        <span className="text-yellow-600 dark:text-yellow-400 font-mono italic font-bold">{command}</span>
        {' '}
        <span className="text-secondary">
          {args}
        </span>
      </div>
    );
  }

  return (
    <div className="prose dark:prose-invert prose-sm max-w-none">
      <MarkdownWithAnsi>{content}</MarkdownWithAnsi>
      {/* Show "Translated" accordion when input was auto-translated to English */}
      {translatedInput && (
        <TranslationAccordion
          label="Translated (English sent to model)"
          content={translatedInput}
        />
      )}
    </div>
  );
};

const SearchModal = ({ isOpen, onClose, onSearch }: { isOpen: boolean; onClose: () => void; onSearch: (q: string, l: number) => void }) => {
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(3);

  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSubmit = () => {
    if (query.trim()) {
      onSearch(query, limit);
      onClose();
      setQuery('');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[60] flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-secondary border border-border rounded-xl shadow-2xl w-full max-w-md p-6 animate-in fade-in zoom-in-95 duration-200" onClick={e => e.stopPropagation()}>
        <h2 className="text-xl font-bold text-primary mb-4 flex items-center gap-2">
          <Globe className="text-brand-400" />
          Deep Research
        </h2>

        <div className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-tertiary uppercase tracking-wider mb-1">Search Query</label>
            <input
              ref={inputRef}
              className="input w-full"
              placeholder="What are we searching for?"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSubmit()}
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-tertiary uppercase tracking-wider mb-1">Number of Sources</label>
            <NumberInput
              min={1}
              max={10}
              className="input w-full"
              value={limit}
              onChange={setLimit}
              onKeyDown={e => e.key === 'Enter' && handleSubmit()}
            />
          </div>

          <div className="flex justify-end gap-2 mt-6">
            <button className="btn-secondary" onClick={onClose}>Cancel</button>
            <button
              className="btn-primary"
              onClick={handleSubmit}
              disabled={!query.trim()}
            >
              Search
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

const FilePreviewModal = ({ isOpen, onClose, file, onConfirm }: { isOpen: boolean; onClose: () => void; file: { name: string; content: string } | null; onConfirm: () => void }) => {
  if (!isOpen || !file) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[60] flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-secondary border border-border rounded-xl shadow-2xl w-[90vw] max-w-[90vw] p-6 animate-in fade-in zoom-in-95 duration-200 flex flex-col max-h-[90vh]" onClick={e => e.stopPropagation()}>
        <h2 className="text-xl font-bold text-primary mb-4 flex items-center gap-2 shrink-0">
          <FileText className="text-brand-400" />
          Read File
        </h2>

        <div className="mb-4 shrink-0">
          <label className="block text-xs font-semibold text-tertiary uppercase tracking-wider mb-1">Filename</label>
          <div className="px-3 py-2 bg-primary rounded border border-border text-primary text-sm font-mono truncate">
            {file.name}
          </div>
        </div>

        <div className="flex-1 overflow-hidden flex flex-col min-h-0">
          <label className="block text-xs font-semibold text-tertiary uppercase tracking-wider mb-1">Content Preview</label>
          <div className="flex-1 bg-primary rounded border border-border overflow-y-auto p-3 font-mono text-xs text-secondary whitespace-pre">
            {file.content}
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-6 shrink-0">
          <button className="btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={onConfirm}>Add to Context</button>
        </div>
      </div>
    </div>
  );
};

export function ChatInterface() {
  const { triggers: randomPromptTriggers } = usePromptTriggers();
  const { chatSessionId, chatMessages, setChatSessionId, addChatMessage, updateLastUserMessage, clearChat } = useAppStore();
  const [model, setModel] = useState(''); // Initial value empty, set after fetch
  const [precision, setPrecision] = useState('auto'); // Precision override: auto, int4, int8, float16, bfloat16, float32
  const [framework, setFramework] = useState(navigator.userAgent.toLowerCase().includes('mac') ? 'mlx' : 'auto'); // Platform override: auto, torch, mlx
  const [defaultModelId, setDefaultModelId] = useState(''); // Store default model ID from backend
  const [input, setInput] = useState('');
  const [isConnecting, setIsConnecting] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [thinkingStartTime, setThinkingStartTime] = useState<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState<string>('');
  const [isModelReady, setIsModelReady] = useState(false);
  const [loadingLogs, setLoadingLogs] = useState<string[]>([]);
  const [showSearchModal, setShowSearchModal] = useState(false);
  const [showFileModal, setShowFileModal] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [fileData, setFileData] = useState<{ name: string; content: string } | null>(null);
  const [queuedMessages, setQueuedMessages] = useState<string[]>([]);

  // Translation State
  const [autoTranslateInput, setAutoTranslateInput] = useState(false);
  const [translateOutput, setTranslateOutput] = useState(false);
  const [targetLanguage, setTargetLanguage] = useState('spa_Latn');
  const [translationModel, setTranslationModel] = useState('nllb-200-3.3b');
  const [inputTranslationModel, setInputTranslationModel] = useState('nllb-200-3.3b');
  const [inputSourceLanguage, setInputSourceLanguage] = useState('auto'); // Auto-detect by default

  // Use global models cache
  const { models } = useModels();
  const availableModels = models?.text || [];

  const socketRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const thinkingStartTimeRef = useRef<number | null>(null);

  // History navigation state
  const [historyIndex, setHistoryIndex] = useState<number | null>(null);
  const [historyGhost, setHistoryGhost] = useState(false);

  // Sidebar collapse state
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [showCollapseHint, setShowCollapseHint] = useState(true); // Pulsating hint animation

  // Set default model when models are loaded from global cache
  useEffect(() => {
    if (availableModels.length > 0 && !model) {
      // Find default model based on backend flag
      const defaultModel = availableModels.find((m: any) => m.is_default);
      const initialModel = defaultModel?.name || availableModels[0]?.name || '';

      if (initialModel) {
        setModel(initialModel);
        setDefaultModelId(initialModel);
      }
    }
  }, [availableModels, model]);

  // Session cleanup on mount
  useEffect(() => {
    // Session is lost on refresh since backend generates new ID on connect
    setChatSessionId('');

    // Cleanup: Disconnect when component unmounts (navigating away)
    return () => {
      if (socketRef.current?.readyState === WebSocket.OPEN || socketRef.current?.readyState === WebSocket.CONNECTING) {
        // console.log('Navigating away: Auto-disconnecting chat...');
        socketRef.current.close();
      }
      // Reset critical state
      setChatSessionId('');
      clearChat();
      setModel(''); // Will wait for re-fetch on mount
      setStatusMessage(null);
      setIsProcessing(false);
      setQueuedMessages([]);
      setLoadError(null);
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, loadingLogs, queuedMessages, isProcessing]); // Scroll when any of these change

  // Elapsed time timer for "Thinking..." indicator
  useEffect(() => {
    if (!thinkingStartTime) {
      setElapsedTime('');
      return;
    }

    // Updated to use shared utility
    // No local formatElapsed needed

    // Update immediately
    setElapsedTime(formatDuration(Date.now() - thinkingStartTime));

    // Then update every second
    const interval = setInterval(() => {
      setElapsedTime(formatDuration(Date.now() - thinkingStartTime));
    }, 1000);

    return () => clearInterval(interval);
  }, [thinkingStartTime]);

  const performSend = (contentToSend: string, isOverride?: boolean) => {
    // Add user message immediately
    const userMsg = { role: 'user' as const, content: contentToSend };
    addChatMessage(userMsg);

    // Set immediate processing state with appropriate message
    setIsProcessing(true);
    const isSearch = contentToSend.startsWith('/search') || contentToSend.startsWith('/online-search');
    const isRead = contentToSend.startsWith('/read ') || contentToSend.includes('[File Context:');
    setStatusMessage(isSearch ? "Searching..." : isRead ? "Reading..." : "Thinking...");
    const now = Date.now();
    setThinkingStartTime(now);  // Start elapsed time timer
    thinkingStartTimeRef.current = now;

    // Send to server
    socketRef.current?.send(JSON.stringify({
      type: 'message',
      content: contentToSend,
      model: model,
      session_id: chatSessionId,
      translate_input: autoTranslateInput,
      translate_output: translateOutput,
      target_language: targetLanguage,
      translation_model: translationModel,
      input_translation_model: inputTranslationModel,
      input_source_language: inputSourceLanguage,  // Source language for input translation
      precision: precision !== 'auto' ? precision : undefined,
      framework: framework !== 'auto' ? framework : undefined,
    }));

    // Only clear input if we sent from the text area
    if (!isOverride) {
      setInput('');
    }
  };

  const sendMessage = (overrideContent?: string) => {
    let contentToSend = overrideContent || input;
    if (!contentToSend.trim() || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN || !isModelReady) return;

    // Client-side expansion of Random Prompt command (rndPr)
    // This allows the user bubble to show the ACTUAL prompt instead of the command.
    const normalized = contentToSend.trim().toLowerCase();
    
    // Check against dynamic triggers from backend
    if (randomPromptTriggers.includes(normalized)) {
       // Combine text-relevant prompts (Article + Code)
       // We can iterate PROMPTS keys if we want, or stick to 'code'/'article' default for text models
       // Note: PROMPTS is imported from data/prompts
       const pool: string[] = [];
       if (PROMPTS.article) pool.push(...PROMPTS.article);
       if (PROMPTS.code) pool.push(...PROMPTS.code);
       
       if (pool.length > 0) {
          const randomPrompt = pool[Math.floor(Math.random() * pool.length)];
          contentToSend = randomPrompt;
       }
    }

    // If already processing, add to queue instead of sending
    if (isProcessing) {
      setQueuedMessages(prev => [...prev, contentToSend]);
      if (!overrideContent) setInput('');
      return;
    }

    performSend(contentToSend, !!overrideContent);
  };

  // Process message queue automatically when idle
  useEffect(() => {
    if (!isProcessing && queuedMessages.length > 0 && isModelReady && socketRef.current?.readyState === WebSocket.OPEN) {
      const nextMessage = queuedMessages[0];
      setQueuedMessages(prev => prev.slice(1));
      performSend(nextMessage);
    }
  }, [isProcessing, queuedMessages, isModelReady, socketRef.current?.readyState]);

  const handleRetry = () => {
    setLoadError(null);
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      setIsConnecting(true);
      socketRef.current.send(JSON.stringify({ type: 'load', model }));
    } else {
      connect();
    }
  };

  const connect = () => {
    if (socketRef.current?.readyState === WebSocket.OPEN) return;
    setIsConnecting(true);
    setIsModelReady(false);
    setLoadError(null);

    // Connect using dynamically loaded API_BASE
    const wsBaseUrl = API_BASE();
    if (!wsBaseUrl) {
      console.error("API_BASE is empty despite init check!");
      return;
    }

    // Ensure correct protocol (http->ws, https->wss)
    const precisionParam = precision !== 'auto' ? `&precision=${precision}` : '';
    const frameworkParam = framework !== 'auto' ? `&framework=${framework}` : '';
    socketRef.current = new WebSocket(`${wsBaseUrl.replace(/^http/, 'ws')}/ws/chat?model=${model}${precisionParam}${frameworkParam}`);
    socketRef.current.onopen = () => {
      // Trigger model loading immediately with precision and framework
      socketRef.current?.send(JSON.stringify({ 
        type: 'load', 
        model, 
        precision: precision !== 'auto' ? precision : undefined,
        framework: framework !== 'auto' ? framework : undefined
      }));
      // Don't set isConnecting(false) here - wait for model to be ready
    };

    socketRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'session') {
        setChatSessionId(data.session_id);
      } else if (data.type === 'response') {
        // Calculate thinking time directly (elapsedTime might be stale in closure)
        const formatElapsed = (ms: number): string => {
          const seconds = Math.floor(ms / 1000);
          if (seconds < 60) return `${seconds}s`;
          const minutes = Math.floor(seconds / 60);
          const remainingSeconds = seconds % 60;
          if (minutes < 60) return `${minutes}m ${remainingSeconds}s`;
          const hours = Math.floor(minutes / 60);
          const remainingMinutes = minutes % 60;
          return `${hours}h ${remainingMinutes}m`;
        };

        // Get current thinkingStartTime from ref directly (closure safe)
        const startTime = thinkingStartTimeRef.current;
        const finalThinkingTime = startTime ? formatElapsed(Date.now() - startTime) : undefined;

        addChatMessage({
          role: 'assistant',
          content: data.content,
          originalContent: data.original_content,
          reasoning: data.reasoning, // Store reasoning from server
          thinkingTime: finalThinkingTime
        });

        // Update the last user message with translated input if translation occurred
        if (data.translated_input) {
          updateLastUserMessage({ translatedInput: data.translated_input });
        }

        setIsProcessing(false);
        thinkingStartTimeRef.current = null;
      } else if (data.type === 'command_response') {
        // Handle command responses (like search, file read) - stop processing since no AI response follows
        addChatMessage({
          role: 'system',
          content: data.content
        });
        // Clear processing state since command is complete and no AI response is coming
        setIsProcessing(false);
        setStatusMessage(null);
        thinkingStartTimeRef.current = null;
        setThinkingStartTime(null);
      } else if (data.type === 'status') {
        setIsProcessing(true);
        setStatusMessage(data.message || (data.status === 'loading' ? 'Loading model...' : 'Thinking...'));

        // Mark connection as complete when model is ready
        if (data.status === 'ready') {
          setIsConnecting(false);
          setIsModelReady(true);
          setLoadError(null);
          setLoadingLogs([]); // Clear logs when ready
        } else if (data.status === 'error') {
          setIsConnecting(false);
          setIsModelReady(false);
          setLoadError(data.message || 'Failed to load model.');
          setIsProcessing(false);
        }
      } else if (data.type === 'log') {
        const msg = data.message;
        setLoadingLogs(prev => {
          // Handle TQDM-style concurrent progress bars
          const cleanMsg = msg.replace(/\r/g, '').trim();
          const isProgress = msg.includes('%') || msg.includes('it/s');

          // Match prefix: everything before the first colon or pipe
          const barMatch = cleanMsg.match(/^([^:|]+)[:|]/);

          if (isProgress) {
            if (barMatch) {
              const prefix = barMatch[1].trim();
              // Find existing line that contains this prefix near the start
              const existingIndex = prev.findIndex(line => line.trim().startsWith(prefix));

              if (existingIndex !== -1) {
                const updated = [...prev];
                updated[existingIndex] = cleanMsg; // Store cleaned version
                return updated;
              }
              return [...prev, cleanMsg];
            } else {
              // Fallback for generic bars without prefix
              const lastLog = prev[prev.length - 1];
              if (lastLog && (lastLog.includes('%') || lastLog.includes('it/s'))) {
                return [...prev.slice(0, -1), cleanMsg];
              }
            }
          }
          return [...prev, cleanMsg];
        });
      } else if (data.type === 'status_clear') {
        setIsProcessing(false);
        setStatusMessage('');
      } else if (data.type === 'error') {
        addChatMessage({ role: 'assistant', content: `Error: ${data.message}` });
        setIsProcessing(false);
        setStatusMessage('');
      }
    };

    socketRef.current.onerror = () => {
      setIsConnecting(false);
      setIsModelReady(false);
    }
    socketRef.current.onclose = () => {
      setChatSessionId('');
      setIsModelReady(false);
      setLoadingLogs([]);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // History Navigation
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      const userMessages = chatMessages.filter(m => m.role === 'user');
      if (userMessages.length === 0) return;

      if (!input && !historyIndex && e.key === 'ArrowUp') {
        // Start history traversal from end
        const newIndex = userMessages.length - 1;
        setHistoryIndex(newIndex);
        setInput(userMessages[newIndex].content);
        setHistoryGhost(true);
        e.preventDefault();
        return;
      }

      if (historyIndex !== null) {
        let newIndex = historyIndex + (e.key === 'ArrowUp' ? -1 : 1);

        // Wrap around
        if (newIndex < 0) newIndex = userMessages.length - 1;
        if (newIndex >= userMessages.length) newIndex = 0;

        setHistoryIndex(newIndex);
        setInput(userMessages[newIndex].content);
        setHistoryGhost(true);
        e.preventDefault();
        return;
      }
    }

    // Confirm Ghost State
    if (historyGhost && (e.key === 'Tab' || e.key === ' ')) {
      setHistoryGhost(false);
      setHistoryIndex(null);
      if (e.key === 'Tab') {
        e.preventDefault(); // Don't move focus
      }
      return;
    }

    // Clear ghost on any other modification key
    if (historyGhost && e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
      setHistoryGhost(false);
      setHistoryIndex(null);
    }

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      setHistoryGhost(false);
      setHistoryIndex(null);
      sendMessage();
    }
  };



  const handleClear = () => {
    socketRef.current?.close();
    setChatSessionId('');
    setIsModelReady(false);
    setLoadingLogs([]);
    clearChat();
    setModel(defaultModelId);
    setStatusMessage(null);
    setIsProcessing(false);
    setQueuedMessages([]);
    setLoadError(null);
  };

  const cancelQueuedMessage = (index: number) => {
    setQueuedMessages(prev => prev.filter((_, i) => i !== index));
  };

  // --- File Reader Logic ---
  const handleReadFileClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.size > 1024 * 1024 * 5) {
      alert("File too large (>5MB). Please copy text manually.");
      e.target.value = '';
      return;
    }

    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result;
      if (typeof text === 'string') {
        if (text.slice(0, 1000).includes('\0')) {
          alert("File appears to be binary. Cannot read as text.");
          return;
        }
        setFileData({ name: file.name, content: text });
        setShowFileModal(true);
      }
    };
    reader.onerror = () => alert("Failed to read file.");
    reader.readAsText(file);
    e.target.value = '';
  };

  const confirmFileRead = () => {
    if (!fileData) return;

    const formatted = `\n\`\`\`markdown\n[File Context: ${fileData.name}]\n\`\`\`\n${fileData.content}\n\`\`\`\n`;

    // 1. Send file content to model
    sendMessage(formatted);

    setShowFileModal(false);
    setFileData(null);
  };

  // --- Save / Export Logic ---
  const handleSaveDownload = () => {
    if (chatMessages.length === 0) return;
    setShowSaveModal(true);
  };

  const handleExport = async (options: SaveOptions) => {
    try {
      const response = await fetch(`${API_BASE()}/api/text/export`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: options.content,
          format: options.format,
          filename: options.filename,
        }),
      });

      if (!response.ok) throw new Error('Export failed');

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = options.filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export error:', err);
      alert('Failed to export file. Please try again.');
    }
  };

  const getFullChatMarkdown = () => {
    let content = `# Chat Session History\n\n`;
    chatMessages.forEach(msg => {
      const role = msg.role === 'user' ? 'User' : 'Assistant';
      content += `## ${role}\n${msg.content}\n\n`;
    });
    return content;
  };

  const getLastMessageInfo = () => {
    const lastAssistantMsg = [...chatMessages].reverse().find(m => m.role === 'assistant');
    if (!lastAssistantMsg) return { content: null, suggestedFilename: null };

    const content = lastAssistantMsg.content;
    let suggestedFilename = `chat_export_${Date.now()}`;

    // Detect code blocks
    const codeMatch = content.match(/```(.*?)\n([\s\S]*?)```/);
    if (codeMatch) {
      const lang = codeMatch[1].trim().toLowerCase();
      const langMap: Record<string, string> = {
        'python': 'py', 'py': 'py',
        'javascript': 'js', 'js': 'js',
        'typescript': 'ts', 'ts': 'ts',
        'tsx': 'tsx', 'jsx': 'jsx',
        'html': 'html', 'css': 'css',
        'cpp': 'cpp', 'c++': 'cpp',
        'c': 'c', 'java': 'java',
        'rust': 'rs', 'rs': 'rs',
        'go': 'go', 'bash': 'sh', 'sh': 'sh',
        'sql': 'sql', 'json': 'json', 'yaml': 'yml', 'yml': 'yml'
      };

      const ext = langMap[lang] || 'txt';

      // Try to find a filename comment like # filename: script.py or // filename: script.js
      const filenameMatch = content.match(/(?:#|\/\/)\s*(?:filename:\s*)?([^\s]+\.\w+)/i);
      if (filenameMatch) {
        suggestedFilename = filenameMatch[1];
      } else {
        // Fallback to slugified user prompt leading to this message
        const lastUserMsg = chatMessages.find((m, idx) => {
          const assistantIdx = chatMessages.indexOf(lastAssistantMsg);
          return m.role === 'user' && idx < assistantIdx;
        });

        if (lastUserMsg) {
          const cleanText = lastUserMsg.content.replace(/[^a-zA-Z0-9\s]/g, '').trim();
          const slug = cleanText.split(/\s+/).slice(0, 5).join('_').toLowerCase();
          if (slug) suggestedFilename = `${slug}.${ext}`;
        }
      }
    } else {
      // No code block, slugify user prompt
      const lastUserMsg = [...chatMessages].reverse().find(m => m.role === 'user');
      if (lastUserMsg) {
        const cleanText = lastUserMsg.content.replace(/[^a-zA-Z0-9\s]/g, '').trim();
        const slug = cleanText.split(/\s+/).slice(0, 5).join('_').toLowerCase();
        if (slug) suggestedFilename = `${slug}`;
      }
    }

    return { content, suggestedFilename };
  };



  // Sort models
  const sortedModels = MODEL_ORDER.filter(name =>
    name === 'default' || availableModels.some((m) => m.name === name)
  );


  return (
    <div className="flex flex-col lg:flex-row h-full bg-primary text-primary">
      {/* Parameters Sidebar - hidden when collapsed */}
      {!isSidebarCollapsed && (
        <div className="w-full lg:w-[500px] border-b lg:border-b-0 lg:border-r border-border p-4 lg:py-6 lg:pr-[27px] lg:pl-1 flex flex-col gap-6 overflow-y-auto shrink-0 h-auto lg:h-full">
          <div className="flex items-start justify-between">
            <div>
              <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
                <MessageSquare className="text-brand-400" /> Chat
              </h2>
              <p className="text-xs text-tertiary">Chat with AI models locally</p>
            </div>
            {/* Collapse button - only show when model is ready */}
            {isModelReady && (
              <button
                onClick={(e) => {
                  if (e.ctrlKey || e.metaKey) {
                    // Easter egg: Ctrl+Click stops the animation
                    setShowCollapseHint(false);
                  } else {
                    setIsSidebarCollapsed(true);
                  }
                }}
                className={`p-2 rounded-lg transition-all text-secondary hover:text-primary -mr-2 ${showCollapseHint
                  ? 'animate-pulse bg-tertiary/30 hover:bg-tertiary/50'
                  : 'hover:bg-tertiary/50'
                  }`}
                title={showCollapseHint ? "Collapse sidebar (Ctrl+Click to stop animation)" : "Collapse sidebar"}
              >
                <ChevronLeft size={20} />
              </button>
            )}
          </div>

          {/* Model Selector with Connect/Disconnect */}
          <div className="space-y-4"> {/* Increased spacing for visual separation */}
            
            {/* 1. Framework Selector (First on list, hidden if not Mac) */}
            <div className={`space-y-1 ${!navigator.userAgent.toLowerCase().includes('mac') ? 'hidden' : ''}`}>
              <label className="text-sm font-medium text-secondary block">Platform</label>
              <select
                className="select w-auto bg-primary border-border text-sm focus:border-brand-500 max-w-full"
                value={framework}
                onChange={(e) => setFramework(e.target.value)}
                disabled={!!chatSessionId}
                title="Inference Framework - Use MLX for best performance on Mac"
              >
                <option value="mlx">MLX (Native Mac)</option>
                <option value="torch">PyTorch (MPS)</option>
              </select>
            </div>

            {/* 2. Precision Selector */}
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-secondary">Precision</label>
                <button
                  onClick={() => useAppStore.getState().openHelpSection('precision')}
                  className="text-tertiary hover:text-brand-500 transition-colors"
                  title="Learn about precision options"
                >
                  <HelpCircle size={14} />
                </button>
              </div>
              <select
                className="select w-auto bg-primary border-border text-sm focus:border-brand-500 max-w-full"
                value={precision}
                onChange={(e) => setPrecision(e.target.value)}
                disabled={!!chatSessionId}
                title="Model precision - affects speed and memory usage"
              >
                <option value="auto">
                  {/* Dynamic default label based on Framework */}
                  {(() => {
                    const isMac = navigator.userAgent.toLowerCase().includes('mac');
                    // On Mac, effective framework is MLX unless explicit 'torch'. On others, it's 'torch'.
                    // Our state 'framework' defaults to 'mlx' on Mac in previous steps.
                    const isMlx = framework === 'mlx' || (framework === 'auto' && isMac);
                    
                    return `Auto (${isMlx ? 'int4 - MLX Default' : 'bfloat16 - Default'})`;
                  })()}
                </option>
                <option value="int4">int4 (4-bit, Fast)</option>
                <option value="int6">int6 (6-bit, Balanced Speed)</option>
                <option value="int8">int8 (8-bit, Balanced Quality)</option>
                <option value="float16">float16 (Half)</option>
                <option value="bfloat16">bfloat16 (Brain Float)</option>
                <option value="float32">float32 (Full)</option>
              </select>
            </div>

            {/* 3. Model Selector */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-secondary">Model</label>
                <button
                  onClick={() => useAppStore.getState().openHelpSection('chat')}
                  className="text-tertiary hover:text-brand-500 transition-colors"
                  title="Need help choosing your model?"
                >
                  <HelpCircle size={14} />
                </button>
              </div>
              <select
                className="select w-full bg-primary border-border text-sm focus:border-brand-500"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={!!chatSessionId}
              >
                {sortedModels.map((name) => {
                  const info = MODEL_DISPLAY_INFO[name];
                  // Use shared utility with current precision/framework state
                  const vram = getDynamicRam(name, precision, framework);
                  // Add warning if RAM is very high (e.g. > 32GB)
                  const isHighRam = parseInt(vram.replace('~', '').replace('GB', '')) > 32;
                  return (
                    <option key={name} value={name}>
                      {info ? `${isHighRam ? '⚠️ ' : ''}${info.label} (${vram})` : name}
                    </option>
                  );
                })}
              </select>
            </div>

            {(!chatSessionId || isConnecting) ? (
              <button
                className="w-full btn-primary py-2.5 flex items-center justify-center gap-2"
                onClick={connect}
                disabled={isConnecting}
              >
                {isConnecting ? (<><Loader2 className="animate-spin" size={16} /> Connecting...</>) : (<><MessageSquare size={16} /> Connect</>)}
              </button>
            ) : (
              <button
                className="w-full btn-secondary py-2.5 flex items-center justify-center gap-2"
                onClick={handleClear}
                title="Disconnect"
              >
                <LogOut size={16} /> Disconnect
              </button>
            )}
          </div>

          {/* Translation Parameters */}
          <div className="border border-border rounded-lg bg-secondary/30 p-3 space-y-3 shrink-0">
            <div className="flex items-center gap-2 mb-1">
              <Globe size={14} className="text-brand-400" />
              <span className="text-xs font-bold uppercase tracking-wider text-secondary">Translation</span>
            </div>

            <TranslateOptions
              title="Auto-Translate Input to English"
              enabled={autoTranslateInput}
              onEnabledChange={setAutoTranslateInput}
              selectedModel={inputTranslationModel}
              onModelChange={setInputTranslationModel}
              targetLanguage="" // Not used when source language shown
              onLanguageChange={() => { }}
              hideLanguageSelector={true}
              showToggle={true}
              showSourceLanguage={true}
              sourceLanguage={inputSourceLanguage}
              onSourceLanguageChange={setInputSourceLanguage}
              infoMessage="NLLB-200: Fast with auto-detect, best for quick translations. LLM models: Better nuance, tone and context awareness - ideal for professional or creative content. LLMs require explicit source language."
            />

            <div className="border-t border-border/50 pt-3">
              <TranslateOptions
                title="Translate Output"
                enabled={translateOutput}
                onEnabledChange={setTranslateOutput}
                selectedModel={translationModel}
                onModelChange={setTranslationModel}
                targetLanguage={targetLanguage}
                onLanguageChange={setTargetLanguage}
                showToggle={true}
                infoMessage="Choose NLLB for speed and broad language coverage. Use LLM models for more natural, context-aware translations - especially valuable for professional or creative content."
              />
            </div>
          </div>

          {/* Chat Input Area */}
          <div className="space-y-2 mt-auto">
            {/* Action Toolbar - directly above input */}
            <div className="flex flex-wrap gap-2 mb-2">
              <button
                className="btn-secondary px-3 py-1 text-xs flex items-center gap-1"
                onClick={handleReadFileClick}
                disabled={!isModelReady}
                title={!isModelReady ? "Connect to chat model first" : "Read a file (all text/code types)"}
              >
                <FileText size={14} /> <span>Read File</span>
              </button>
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                onChange={handleFileChange}
                accept=".txt,.md,.json,.py,.js,.ts,.tsx,.jsx,.html,.css,.scss,.xml,.yaml,.yml,.sh,.bat,.c,.cpp,.h,.java,.rs,.go,.php,.rb,.pl,.lua,.sql,.log,.ini,.conf,.env,.dockerfile,makefile,text/*"
              />
              <button
                className="btn-secondary px-3 py-1 text-xs flex items-center gap-1"
                onClick={() => setShowSearchModal(true)}
                disabled={!isModelReady}
                title={!isModelReady ? "Connect to chat model first" : "Deep Research (/search <query>)"}
              >
                <Globe size={14} /> <span>Search</span>
              </button>
              <button
                className="btn-secondary px-3 py-1 text-xs flex items-center gap-1"
                onClick={handleSaveDownload}
                disabled={!isModelReady}
                title={!isModelReady ? "Connect to chat model first" : "Save conversation as markdown"}
              >
                <Save size={14} /> <span>Save</span>
              </button>
            </div>
            <textarea
              ref={!isSidebarCollapsed ? inputRef : undefined}
              className={`w-full bg-primary border border-border rounded-lg p-3 text-sm focus:outline-none focus:border-brand-500 resize-y min-h-[120px] ${historyGhost ? 'opacity-60' : ''}`}
              placeholder={isConnecting ? "Connecting..." : "Type a message... (Shift+Enter for new line, Enter to send)"}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                if (historyGhost) {
                  setHistoryGhost(false);
                  setHistoryIndex(null);
                }
              }}
              onKeyDown={handleKeyDown}
              disabled={!isModelReady}
            />
          </div>

          {/* Send Button */}
          <button
            className="w-full bg-gradient-to-r from-brand-600 to-cyan-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-primary font-bold py-3 rounded-lg shadow-lg shadow-brand-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all"
            onClick={() => sendMessage()}
            disabled={!isModelReady || !input.trim()}
          >
            <Send size={18} /> Send
          </button>
        </div>
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col overflow-hidden bg-primary/30 min-h-[500px] lg:min-h-0 relative">
        {/* Expand button header when collapsed */}
        {isSidebarCollapsed && (
          <div className="flex items-center gap-2 p-4 border-b border-border">
            <button
              onClick={() => setIsSidebarCollapsed(false)}
              className="p-2 hover:bg-tertiary/50 rounded-lg transition-colors text-secondary hover:text-primary"
              title="Expand sidebar"
            >
              <ChevronRight size={20} />
            </button>
            <h2 className="text-lg font-bold flex items-center gap-2">
              <MessageSquare className="text-brand-400" size={18} /> Chat
            </h2>
          </div>
        )}

        <div className={`flex-1 overflow-y-auto p-6 space-y-4 ${isSidebarCollapsed ? 'pb-48' : ''}`}>
          {chatMessages.length === 0 && (
            <div className="text-center text-tertiary mt-20 px-4">

              {/* State 1: Loading / Connecting */}
              {((isConnecting || loadingLogs.length > 0) && !isModelReady && !loadError) && (
                <>
                  <Loader2 className="w-12 h-12 mx-auto mb-2 text-yellow-500/80 animate-spin" />
                  <p className="text-lg font-semibold text-secondary">Loading Model...</p>
                  <p className="text-sm mt-1 mb-6">Initializing {MODEL_DISPLAY_INFO[model]?.label || model}...</p>

                  <div className="flex items-center justify-between mb-2 w-full max-w-none">
                    <span className="text-xs font-mono text-tertiary uppercase tracking-wider">Server Logs</span>
                    {loadingLogs.length > 0 && (
                      <button
                        onClick={() => setLoadingLogs([])}
                        className="text-xs flex items-center gap-1 text-tertiary hover:text-secondary transition-colors"
                        title="Clear logs"
                      >
                        <Trash2 size={12} /> Clear
                      </button>
                    )}
                  </div>

                  {/* Embedded Logs */}
                  <div className="mx-auto w-full max-w-none bg-primary/50 p-4 rounded-lg text-left font-mono text-xs text-secondary max-h-64 overflow-y-auto border border-border/50 shadow-inner">
                    {loadingLogs.length === 0 && <span className="opacity-50 italic">Waiting for server...</span>}
                    {loadingLogs.map((log, i) => (
                      <div key={i} className="whitespace-pre-wrap">{log}</div>
                    ))}
                    <div ref={messagesEndRef} />
                  </div>
                </>
              )}

              {/* State 2: Ready */}
              {(!isConnecting && isModelReady) && (
                <>
                  <MessageSquare className="w-12 h-12 mx-auto mb-2 text-green-500/80" />
                  <p className="text-lg font-semibold text-secondary">Model Ready!</p>
                  <p className="text-sm mt-2">Type a message below to start chatting with {MODEL_DISPLAY_INFO[model]?.label || model}.</p>
                </>
              )}

              {/* State 3: Error */}
              {loadError && (
                <div className="mx-auto w-full max-w-2xl bg-red-950/20 border border-red-500/30 rounded-xl p-8 text-center animate-in fade-in zoom-in duration-300">
                  <AlertCircle className="w-16 h-16 mx-auto mb-4 text-red-500" />
                  <h3 className="text-xl font-bold text-red-100 mb-2">Model Loading Failed</h3>
                  <div className="bg-primary/50 p-4 rounded-lg text-red-200 font-mono text-sm text-left mb-6 border border-red-500/20 whitespace-pre-wrap">
                    {loadError}
                  </div>
                  <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                    <button
                      onClick={handleRetry}
                      className="btn-primary bg-red-600 hover:bg-red-500 border-red-500 flex items-center gap-2 px-8 py-3"
                    >
                      <RefreshCw size={18} className={isConnecting ? 'animate-spin' : ''} />
                      {isConnecting ? 'Retrying...' : 'Retry Loading'}
                    </button>
                    <button
                      onClick={handleClear}
                      className="btn-secondary text-secondary hover:text-primary"
                    >
                      Choose Different Model
                    </button>
                  </div>
                </div>
              )}

              {/* State 4: Disconnected / Initial */}
              {(!isConnecting && !isModelReady && !loadError && loadingLogs.length === 0) && (
                <>
                  <MessageSquare className="w-12 h-12 mx-auto mb-2 opacity-20" />
                  <p className="text-lg font-medium">Connect to start chatting with AI</p>
                  <p className="text-sm mt-2 opacity-70">Select a model above to get started</p>
                </>
              )}
            </div>
          )}

          {/* (Removed old localized loading logs block) */}

          {chatMessages.map((msg, i) => (
            <React.Fragment key={i}>
              <div className="mb-6 font-mono">
                {/* Header */}
                {msg.role !== 'system' && (
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`font-bold ${msg.role === 'user' ? 'text-blue-400' : 'text-green-400'}`}>
                      {msg.role === 'user' ? 'You:' : 'Bot:'}
                    </span>
                  </div>
                )}

                {/* Content */}
                <div className={`pl-0 ${msg.role === 'system' ? 'text-tertiary italic bg-secondary/30 p-2 rounded border-l-2 border-border' : 'text-secondary'}`}>
                  {msg.role === 'assistant' ? (
                    <ThinkingMessage content={msg.content} originalContent={msg.originalContent} reasoning={msg.reasoning} thinkingTime={msg.thinkingTime} />
                  ) : msg.role === 'user' ? (
                    <UserMessage content={msg.content} translatedInput={msg.translatedInput} />
                  ) : (
                    <div className="text-sm font-sans flex items-center gap-2">
                      {msg.content}
                    </div>
                  )}
                </div>
              </div>
            </React.Fragment>
          ))}

          {/* Render thinking indicator AFTER the last user message in history */}
          {(() => {
            if (!isProcessing) return null;

            // Find the index of the most recent user message that should have a thinking indicator
            // We look from the end of the array backwards
            let lastUserIndex = -1;
            for (let j = chatMessages.length - 1; j >= 0; j--) {
              if (chatMessages[j].role === 'user') {
                lastUserIndex = j;
                break;
              }
            }

            if (lastUserIndex === -1) return null;

            return (
              <div className="flex justify-start mb-6 -mt-2">
                <div className="bg-tertiary/50 rounded-lg px-4 py-2 flex items-center gap-2 animate-in fade-in slide-in-from-left-2 duration-300">
                  <Loader2 className="animate-spin text-primary-400" size={16} />
                  <span className="text-secondary italic font-light">
                    {statusMessage || 'Thinking...'}
                    {elapsedTime && <span className="ml-2 text-tertiary text-sm font-mono">{elapsedTime}</span>}
                  </span>
                </div>
              </div>
            );
          })()}

          {/* Render Queued (Pending) Messages */}
          {queuedMessages.map((content, i) => (
            <div key={`queue-${i}`} className="mb-6 font-mono opacity-60 relative group">
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-blue-400/70">You:</span>
                  <span className="bg-tertiary/50 text-[10px] uppercase px-1.5 py-0.5 rounded text-secondary tracking-wider font-bold border border-border/30">Pending</span>
                </div>
                <button
                  onClick={() => cancelQueuedMessage(i)}
                  className="p-1 hover:bg-red-500/20 hover:text-red-400 text-tertiary rounded transition-all"
                  title="Remove from queue"
                >
                  <Trash2 size={14} />
                </button>
              </div>
              <div className="pl-0 text-secondary italic">
                <UserMessage content={content} />
              </div>
            </div>
          ))}

          <div ref={messagesEndRef} />
        </div>

        {/* Collapsed mode input bar at bottom */}
        {isSidebarCollapsed && (
          <div className="absolute bottom-0 left-0 right-0 bg-primary/95 backdrop-blur-sm border-t border-border p-4">
            <div className="flex flex-wrap gap-2 mb-2">
              <button
                className="btn-secondary px-3 py-1 text-xs flex items-center gap-1"
                onClick={handleReadFileClick}
                disabled={!isModelReady}
                title="Read a file"
              >
                <FileText size={14} /> <span>Read File</span>
              </button>
              <input
                type="file"
                ref={isSidebarCollapsed ? fileInputRef : undefined}
                className="hidden"
                onChange={handleFileChange}
                accept=".txt,.md,.json,.py,.js,.ts,.tsx,.jsx,.html,.css,.scss,.xml,.yaml,.yml,.sh,.bat,.c,.cpp,.h,.java,.rs,.go,.php,.rb,.pl,.lua,.sql,.log,.ini,.conf,.env,.dockerfile,makefile,text/*"
              />
              <button
                className="btn-secondary px-3 py-1 text-xs flex items-center gap-1"
                onClick={() => setShowSearchModal(true)}
                disabled={!isModelReady}
                title="Deep Research"
              >
                <Globe size={14} /> <span>Search</span>
              </button>
              <button
                className="btn-secondary px-3 py-1 text-xs flex items-center gap-1"
                onClick={handleSaveDownload}
                disabled={!isModelReady}
                title="Save conversation"
              >
                <Save size={14} /> <span>Save</span>
              </button>
            </div>
            <div className="flex gap-2">
              <textarea
                ref={isSidebarCollapsed ? inputRef : undefined}
                className={`flex-1 bg-primary border border-border rounded-lg p-3 text-sm focus:outline-none focus:border-brand-500 resize-none min-h-[60px] max-h-[120px] ${historyGhost ? 'opacity-60' : ''}`}
                placeholder="Type a message... (Shift+Enter for new line, Enter to send)"
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  if (historyGhost) {
                    setHistoryGhost(false);
                    setHistoryIndex(null);
                  }
                }}
                onKeyDown={handleKeyDown}
                disabled={!isModelReady}
              />
              <button
                className="bg-gradient-to-r from-brand-600 to-cyan-600 hover:brightness-110 text-primary font-bold px-6 rounded-lg shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all"
                onClick={() => sendMessage()}
                disabled={!isModelReady || !input.trim()}
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        )}
      </div>

      <SearchModal
        isOpen={showSearchModal}
        onClose={() => setShowSearchModal(false)}
        onSearch={(query, limit) => sendMessage(`/search|${limit} ${query}`)}
      />

      <FilePreviewModal
        isOpen={showFileModal}
        onClose={() => setShowFileModal(false)}
        file={fileData}
        onConfirm={confirmFileRead}
      />

      <SaveModal
        isOpen={showSaveModal}
        onClose={() => setShowSaveModal(false)}
        onSave={handleExport}
        lastMessage={getLastMessageInfo().content}
        fullChat={getFullChatMarkdown()}
        suggestedFilename={getLastMessageInfo().suggestedFilename || undefined}
      />
    </div>
  );
}




