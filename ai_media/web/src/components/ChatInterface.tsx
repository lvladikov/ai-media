import React, { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../store';
import { fetchModels } from '../hooks/useApi';
import { API_BASE_URL as API_BASE } from '../config';
import { MessageSquare, Send, Loader2, LogOut, FileText, Save, Globe, ChevronRight, ChevronDown, Copy, Check, Trash2, Image, AlertCircle, RefreshCw } from 'lucide-react';
import domToImage from 'dom-to-image';
import { NumberInput } from './common/NumberInput';
import { MarkdownWithAnsi, MarkdownWithAnsiNoHtml } from './common/AnsiRenderer';
import { SaveModal } from './SaveModal';
import type { SaveOptions } from './SaveModal';
import { formatDuration } from '../utils/formatTime';

// Clipboard CSS mapping for rich-text copy (inline styles for external apps)
const TAILWIND_TO_CSS: Record<string, string> = {
    'text-slate-600': 'color: #475569;',
    'text-red-400': 'color: #f87171;',
    'text-green-400': 'color: #4ade80;',
    'text-yellow-400': 'color: #facc15;',
    'text-blue-400': 'color: #60a5fa;',
    'text-purple-400': 'color: #c084fc;',
    'text-cyan-400': 'color: #22d3ee;',
    'text-slate-200': 'color: #e2e8f0;',
    'text-slate-400': 'color: #94a3b8;',
    'text-red-300': 'color: #fca5a5;',
    'text-green-300': 'color: #86efac;',
    'text-yellow-300': 'color: #fde047;',
    'text-blue-300': 'color: #93c5fd;',
    'text-purple-300': 'color: #d8b4fe;',
    'text-cyan-300': 'color: #67e8f9;',
    'text-white': 'color: #ffffff;',
    'font-bold': 'font-weight: bold;',
    'italic': 'font-style: italic;',
    'underline': 'text-decoration: underline;',
    'bg-slate-800': 'background-color: #1e293b;',
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



interface ModelInfo {
  name: string;
  model_id: string;
  vram_required: number | null;
  ram_required: number | null;
  max_resolution: [number, number] | null;
  is_default?: boolean;
}

// Display names matching CLI
const MODEL_DISPLAY_INFO: Record<string, { label: string; vram: string }> = {
  'deepseek-r1-qwen-7b': { label: 'DeepSeek R1 Qwen 7B (Reasoning)', vram: '~7GB' },
  'deepseek-r1-qwen-14b': { label: 'DeepSeek R1 Qwen 14B (Reasoning)', vram: '~14GB' },
  'deepseek-r1-qwen-32b': { label: 'DeepSeek R1 Qwen 32B (Reasoning)', vram: '~24GB' },
  'deepseek-r1-llama-8b': { label: 'DeepSeek R1 Llama 8B (Reasoning)', vram: '~8GB' },
  'deepseek-r1-llama-70b': { label: 'DeepSeek R1 Llama 70B (Reasoning)', vram: '~40GB' },
  'llama-3.1-8b': { label: 'Llama 3.1 8B (Fast & Stable)', vram: '~8GB' },
  'mistral-nemo-12b': { label: 'Mistral Nemo 12B', vram: '~12GB' },
  'qwen-2.5-14b': { label: 'Qwen 2.5 14B Instruct', vram: '~14GB' },
  'qwen3-coder-30b': { label: 'Qwen3 Coder 30B (MoE, 3.3B active)', vram: '~10GB' },
  'qwen-coder-32b': { label: 'Qwen 2.5 Coder 32B (⚠️ 120GB RAM)', vram: '~24GB' },
  'qwen-coder-14b': { label: 'Qwen 2.5 Coder 14B', vram: '~12GB' },
  'qwen-coder-7b': { label: 'Qwen 2.5 Coder 7B', vram: '~6GB' },
  'qwen-vl': { label: 'Qwen3-VL 8B (Vision)', vram: '~16GB' },
  'qwen3-vl-4b': { label: 'Qwen3-VL 4B (Vision)', vram: '~8GB' },
  'qwen3-vl-2b': { label: 'Qwen3-VL 2B (Vision)', vram: '~4GB' },
};

const MODEL_ORDER = [
  'deepseek-r1-qwen-7b', 'deepseek-r1-qwen-14b', 'deepseek-r1-qwen-32b',
  'deepseek-r1-llama-8b', 'deepseek-r1-llama-70b',
  'llama-3.1-8b', 'mistral-nemo-12b', 'qwen-2.5-14b',
  'qwen3-coder-30b', 'qwen-coder-32b', 'qwen-coder-14b', 'qwen-coder-7b',
  'qwen-vl', 'qwen3-vl-4b', 'qwen3-vl-2b'
];

const ReasoningAccordion = ({ reasoning }: { reasoning: string }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="border-l-2 border-slate-600/50 pl-3 my-1 bg-slate-800/10 py-1 pr-2 rounded-r">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 font-semibold text-xs opacity-50 text-slate-300 mb-0.5 hover:opacity-100 transition-opacity w-full text-left"
      >
        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span>💭 Reasoning</span>
      </button>
      
      {isOpen && (
        <div className="prose prose-invert max-w-none text-slate-400/90 leading-tight [&>p]:!text-xs [&>p]:italic [&>p]:my-0.5 [&>pre]:not-italic [&>pre]:my-1 mt-2 animate-in fade-in slide-in-from-top-1 duration-200">
           <MarkdownWithAnsiNoHtml>{reasoning}</MarkdownWithAnsiNoHtml>
        </div>
      )}
    </div>
  );
};

const ThinkingMessage = React.memo(({ content, reasoning, thinkingTime }: { content: string, reasoning?: string, thinkingTime?: string }) => {
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
    } catch (err) {
      console.error('Failed to copy rich text:', err);
      navigator.clipboard.writeText(content);
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
    <div ref={answerRef} className="prose prose-invert prose-sm max-w-none">
      <MarkdownWithAnsi>{markdown}</MarkdownWithAnsi>
    </div>
  );

  // Case 1: Structured reasoning provided
  if (reasoning) {
    return (
      <div className="font-sans font-normal whitespace-pre-wrap">
        <ReasoningAccordion reasoning={reasoning} />
        {thinkingTime && (
          <div className="mt-2 mb-2 text-xs text-slate-500 italic">
            Thought for {thinkingTime}
          </div>
        )}
        <div className="mt-4">
          <div className="flex items-center justify-between mb-1">
            <div className="font-bold text-slate-100 italic">Answer:</div>
            <div className="flex gap-1">
              <button 
                onClick={copyAsImage}
                className="p-1 hover:bg-slate-700/50 rounded transition-colors text-slate-400 hover:text-white"
                title="Copy answer as image"
              >
                {copiedImage ? <Check size={14} className="text-green-400" /> : <Image size={14} />}
              </button>
              <button 
                onClick={copyRichText}
                className="p-1 hover:bg-slate-700/50 rounded transition-colors text-slate-400 hover:text-white"
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
          <div className="mt-2 mb-2 text-xs text-slate-500 italic">
            Thought for {thinkingTime}
          </div>
        )}
        {a && (
          <div className="mt-4">
            <div className="flex items-center justify-between mb-1">
              <div className="font-bold text-slate-100 italic">Answer:</div>
              <div className="flex gap-1">
                <button 
                  onClick={copyAsImage}
                  className="p-1 hover:bg-slate-700/50 rounded transition-colors text-slate-400 hover:text-white"
                  title="Copy answer as image"
                >
                  {copiedImage ? <Check size={14} className="text-green-400" /> : <Image size={14} />}
                </button>
                <button 
                  onClick={copyRichText}
                  className="p-1 hover:bg-slate-700/50 rounded transition-colors text-slate-400 hover:text-white"
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
          className="p-1 bg-slate-800/80 hover:bg-slate-700 rounded transition-colors text-slate-400 hover:text-white border border-slate-700/50"
          title="Copy as image"
        >
          {copiedImage ? <Check size={14} className="text-green-400" /> : <Image size={14} />}
        </button>
        <button 
          onClick={copyRichText}
          className="p-1 bg-slate-800/80 hover:bg-slate-700 rounded transition-colors text-slate-400 hover:text-white border border-slate-700/50"
          title="Copy formatted text"
        >
          {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
        </button>
      </div>
      {/* Show search time if applicable */}
      {thinkingTime && content.includes('🌍') && (
        <div className="mb-2 text-xs text-slate-500 italic">
          Searched for {thinkingTime}
        </div>
      )}
      {renderContent(content)}
    </div>
  );
});


const UserMessage = ({ content }: { content: string }) => {
  // Check if message is a slash command
  if (content.trim().startsWith('/')) {
    const parts = content.trim().split(' ');
    const command = parts[0];
    const args = parts.slice(1).join(' ');
    
    return (
      <div className="prose prose-invert prose-sm max-w-none">
        <span className="text-yellow-400 font-mono italic font-bold">{command}</span>
        {' '}
        <span className="text-slate-300">
           {args}
        </span>
      </div>
    );
  }
  
  return (
    <div className="prose prose-invert prose-sm max-w-none">
      <MarkdownWithAnsi>{content}</MarkdownWithAnsi>
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
      <div className="bg-secondary border border-border rounded-xl shadow-2xl w-full max-w-2xl p-6 animate-in fade-in zoom-in-95 duration-200 flex flex-col max-h-[80vh]" onClick={e => e.stopPropagation()}>
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
  const { chatSessionId, chatMessages, setChatSessionId, addChatMessage, clearChat } = useAppStore();
  const [model, setModel] = useState(''); // Initial value empty, set after fetch
  const [defaultModelId, setDefaultModelId] = useState(''); // Store default model ID from backend
  const [input, setInput] = useState('');
  const [isConnecting, setIsConnecting] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [thinkingStartTime, setThinkingStartTime] = useState<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState<string>('');
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [isModelReady, setIsModelReady] = useState(false);
  const [loadingLogs, setLoadingLogs] = useState<string[]>([]);
  const [showSearchModal, setShowSearchModal] = useState(false);
  const [showFileModal, setShowFileModal] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [fileData, setFileData] = useState<{ name: string; content: string } | null>(null);
  const [queuedMessages, setQueuedMessages] = useState<string[]>([]);

  const socketRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const thinkingStartTimeRef = useRef<number | null>(null);

  // History navigation state
  const [historyIndex, setHistoryIndex] = useState<number | null>(null);
  const [historyGhost, setHistoryGhost] = useState(false);

  // Fetch models on mount
  useEffect(() => {
    // Session is lost on refresh since backend generates new ID on connect
    setChatSessionId(''); 
    
    fetchModels()
      .then((data) => {
        if (data.text) {
          const models = data.text;
          setAvailableModels(models);
          
          // Find default model based on backend flag
          let initialModel = '';
          const defaultModel = models.find((m: ModelInfo) => m.is_default);
          
          if (defaultModel) {
             initialModel = defaultModel.name;
          } else if (models.length > 0) {
             // Fallback if no default flag
             initialModel = models[0].name;
          }
          
          setModel(initialModel);
          setDefaultModelId(initialModel);
        }
      })
      .catch((err) => console.error('Failed to fetch models:', err));

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
  }, [chatMessages, loadingLogs, queuedMessages]); // Added loadingLogs and queuedMessages to dependencies

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
    
    // Set immediate thinking state
    setIsProcessing(true);
    setStatusMessage("Thinking...");
    const now = Date.now();
    setThinkingStartTime(now);  // Start elapsed time timer
    thinkingStartTimeRef.current = now;
    
    // Send to server
    socketRef.current?.send(JSON.stringify({
      type: 'message',
      content: contentToSend,
      model: model,
      session_id: chatSessionId
    }));
    
    // Only clear input if we sent from the text area
    if (!isOverride) {
      setInput('');
    }
  };

  const sendMessage = (overrideContent?: string) => {
    const contentToSend = overrideContent || input;
    if (!contentToSend.trim() || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN || !isModelReady) return;
    
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
    
    socketRef.current = new WebSocket(`${API_BASE.replace('http', 'ws')}/ws/chat?model=${model}`); // Pass model param!
    
    socketRef.current.onopen = () => {
      // Trigger model loading immediately
      socketRef.current?.send(JSON.stringify({ type: 'load', model }));
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
          reasoning: data.reasoning, // Store reasoning from server
          thinkingTime: finalThinkingTime
        });
        setIsProcessing(false);
        thinkingStartTimeRef.current = null;
      } else if (data.type === 'command_response') {
        // Handle structural responses (like file added) without clearing isProcessing
        addChatMessage({ 
          role: 'system', 
          content: data.content 
        });
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
      const response = await fetch(`${API_BASE}/api/text/export`, {
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
    name === 'default' || availableModels.some(m => m.name === name)
  );


  return (
    <div className="h-full flex flex-col w-full">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <MessageSquare className="text-primary-400" />
          Chat
        </h1>
        <div className="flex items-center gap-2">
          <select 
            className="select w-96" 
            value={model} 
            onChange={(e) => setModel(e.target.value)}
            disabled={!!chatSessionId} // Disable model switching during active session
          >
            {sortedModels.map((name) => {
              const info = MODEL_DISPLAY_INFO[name];
              return (
                <option key={name} value={name}>
                  {info ? `${info.label} ${info.vram}` : name}
                </option>
              );
            })}
          </select>
          {(!chatSessionId || isConnecting) ? (
            <button className="btn-primary" onClick={connect} disabled={isConnecting}>
              {isConnecting ? 'Connecting...' : 'Connect'}
            </button>
          ) : (
            <button className="btn-secondary flex items-center gap-2" onClick={handleClear} title="Disconnect">
              <LogOut size={16} />
              <span className="hidden sm:inline">Disconnect</span>
            </button>
          )}
        </div>
      </div>

      <div className="card flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {chatMessages.length === 0 && (
            <div className="text-center text-slate-500 mt-20 px-4">
              
              {/* State 1: Loading / Connecting */}
              {((isConnecting || loadingLogs.length > 0) && !isModelReady && !loadError) && (
                <>
                  <Loader2 className="w-12 h-12 mx-auto mb-2 text-yellow-500/80 animate-spin" />
                  <p className="text-lg font-semibold text-slate-300">Loading Model...</p>
                  <p className="text-sm mt-1 mb-6">Initializing {MODEL_DISPLAY_INFO[model]?.label || model}...</p>
                  
                  <div className="flex items-center justify-between mb-2 w-full max-w-none">
                    <span className="text-xs font-mono text-slate-500 uppercase tracking-wider">Server Logs</span>
                    {loadingLogs.length > 0 && (
                      <button 
                        onClick={() => setLoadingLogs([])}
                        className="text-xs flex items-center gap-1 text-slate-500 hover:text-slate-300 transition-colors"
                        title="Clear logs"
                      >
                        <Trash2 size={12} /> Clear
                      </button>
                    )}
                  </div>
                  
                  {/* Embedded Logs */}
                   <div className="mx-auto w-full max-w-none bg-slate-900/50 p-4 rounded-lg text-left font-mono text-xs text-slate-400 max-h-64 overflow-y-auto border border-slate-700/50 shadow-inner">
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
                  <p className="text-lg font-semibold text-slate-300">Model Ready!</p>
                  <p className="text-sm mt-2">Type a message below to start chatting with {MODEL_DISPLAY_INFO[model]?.label || model}.</p>
                </>
              )}

              {/* State 3: Error */}
              {loadError && (
                <div className="mx-auto w-full max-w-2xl bg-red-950/20 border border-red-500/30 rounded-xl p-8 text-center animate-in fade-in zoom-in duration-300">
                  <AlertCircle className="w-16 h-16 mx-auto mb-4 text-red-500" />
                  <h3 className="text-xl font-bold text-red-100 mb-2">Model Loading Failed</h3>
                  <div className="bg-slate-950/50 p-4 rounded-lg text-red-200 font-mono text-sm text-left mb-6 border border-red-500/20 whitespace-pre-wrap">
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
                      className="btn-secondary text-slate-400 hover:text-white"
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
                <div className={`pl-0 ${msg.role === 'system' ? 'text-slate-500 italic bg-slate-800/30 p-2 rounded border-l-2 border-slate-700' : 'text-slate-300'}`}>
                  {msg.role === 'assistant' ? (
                    <ThinkingMessage content={msg.content} reasoning={msg.reasoning} thinkingTime={msg.thinkingTime} />
                  ) : msg.role === 'user' ? (
                    <UserMessage content={msg.content} />
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
                <div className="bg-slate-700/50 rounded-lg px-4 py-2 flex items-center gap-2 animate-in fade-in slide-in-from-left-2 duration-300">
                  <Loader2 className="animate-spin text-primary-400" size={16} />
                  <span className="text-slate-400 italic font-light">
                    {statusMessage || 'Thinking...'}
                    {elapsedTime && <span className="ml-2 text-slate-500 text-sm font-mono">{elapsedTime}</span>}
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
                  <span className="bg-slate-700/50 text-[10px] uppercase px-1.5 py-0.5 rounded text-slate-400 tracking-wider font-bold border border-slate-600/30">Pending</span>
                </div>
                <button 
                  onClick={() => cancelQueuedMessage(i)}
                  className="p-1 hover:bg-red-500/20 hover:text-red-400 text-slate-500 rounded transition-all"
                  title="Remove from queue"
                >
                  <Trash2 size={14} />
                </button>
              </div>
              <div className="pl-0 text-slate-400 italic">
                <UserMessage content={content} />
              </div>
            </div>
          ))}

          <div ref={messagesEndRef} />
        </div>

        <div className="border-t border-slate-700 p-4">
          {/* Command Toolbar */}
          <div className="flex gap-2 mb-2">
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

          <div className="flex gap-2">
            <textarea
            ref={inputRef}
            className={`input resize-none h-[42px] py-2.5 overflow-hidden ${historyGhost ? 'opacity-60' : ''}`}
            placeholder={isConnecting ? "Connecting..." : "Type a message... (Shift+Enter for new line)"}
            value={input}
            onChange={(e) => {
                setInput(e.target.value);
                if (historyGhost) {
                    setHistoryGhost(false);
                    setHistoryIndex(null);
                }
            }}
            onKeyDown={handleKeyDown}
            disabled={isConnecting || (!chatSessionId && model !== 'llama-3.1-8b')} // Disable during load/connect
            rows={1}
            style={{ minHeight: '42px' }}
          />
          <button
              className="btn-primary px-4"
              onClick={() => sendMessage()}
              disabled={!isModelReady || !input.trim()}
            >
              <Send size={18} />
            </button>
          </div>
        </div>
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
