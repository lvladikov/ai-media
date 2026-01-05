import { useState, useEffect, useRef } from 'react';
import { X, Download, Columns, FileText, Image, WrapText, Copy, Check, Eye } from 'lucide-react';
import { API_BASE_URL } from '../../config';
import { MarkdownWithAnsi } from './AnsiRenderer';

interface VisionPreviewModalProps {
    isOpen: boolean;
    onClose: () => void;
    originalPath: string;  // Path to the original image/video
    resultPath: string;    // Path to the description .txt file
    resultText?: string | null; // Optionally pass the text content directly
    fileName: string;
    originalIsVideo?: boolean;
}

type ViewTab = 'description' | 'original' | 'sideBySide';

export function VisionPreviewModal({
    isOpen,
    onClose,
    originalPath,
    resultPath,
    resultText,
    fileName,
    originalIsVideo = false
}: VisionPreviewModalProps) {
    const [activeTab, setActiveTab] = useState<ViewTab>('description');
    const [content, setContent] = useState<string | null>(resultText || null);
    const [wordWrap, setWordWrap] = useState(true);
    const [copied, setCopied] = useState(false);
    const [loading, setLoading] = useState(!resultText);
    const contentRef = useRef<HTMLDivElement>(null);

    // Build URLs
    const isFullUrl = originalPath && (originalPath.startsWith('data:') || originalPath.startsWith('blob:') || originalPath.startsWith('http'));
    const originalUrl = isFullUrl ? originalPath : `${API_BASE_URL()}/api/files/${originalPath}`;
    const resultUrl = `${API_BASE_URL()}/api/files/${resultPath}`;

    // Fetch text content if not provided
    useEffect(() => {
        if (resultText) {
            setContent(resultText);
            setLoading(false);
            return;
        }

        setLoading(true);
        fetch(resultUrl)
            .then(res => res.text())
            .then(text => {
                setContent(text);
                setLoading(false);
            })
            .catch(() => {
                setContent('Failed to load description');
                setLoading(false);
            });
    }, [resultUrl, resultText]);

    if (!isOpen) return null;

    const handleDownload = () => {
        const a = document.createElement('a');
        a.href = `${resultUrl}?download=true`;
        a.download = fileName.replace(/\.[^.]+$/, '') + '_description.txt';
        a.click();
    };

    const handleCopy = () => {
        if (content) {
            navigator.clipboard.writeText(content);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    const tabs: { id: ViewTab; label: string; icon: React.ReactNode }[] = [
        { id: 'description', label: 'Description', icon: <FileText size={14} /> },
        { id: 'original', label: 'Original', icon: <Image size={14} /> },
        { id: 'sideBySide', label: 'Side by Side', icon: <Columns size={14} /> },
    ];

    const renderOriginalMedia = (className?: string) => {
        if (originalIsVideo) {
            return (
                <video
                    src={originalUrl}
                    controls
                    className={className || "max-w-full max-h-full object-contain rounded-lg shadow-xl"}
                />
            );
        }
        return (
            <img
                src={originalUrl}
                alt="Original"
                className={className || "max-w-full max-h-full object-contain rounded-lg shadow-xl"}
            />
        );
    };

    const renderDescription = (fullHeight = true) => (
        <div className={`relative group ${fullHeight ? 'h-full' : ''} flex flex-col bg-zinc-100 dark:bg-zinc-900 rounded-lg border border-border overflow-hidden`}>
            {/* Text actions */}
            <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity z-10">
                <button
                    onClick={() => setWordWrap(!wordWrap)}
                    className={`p-2 rounded-lg transition-colors ${wordWrap ? 'bg-brand-500/20 text-brand-400' : 'bg-tertiary text-primary'}`}
                    title={wordWrap ? 'Disable word wrap' : 'Enable word wrap'}
                >
                    <WrapText size={16} />
                </button>
                <button
                    onClick={handleCopy}
                    className="p-2 bg-tertiary hover:bg-tertiary rounded-lg text-primary"
                    title="Copy to clipboard"
                >
                    {copied ? <Check size={16} className="text-green-400" /> : <Copy size={16} />}
                </button>
            </div>

            <div ref={contentRef} className="flex-1 overflow-auto scrollbar-themed p-4">
                {loading ? (
                    <div className="flex items-center justify-center h-full text-tertiary">Loading...</div>
                ) : (
                    <div
                        className={`text-sm text-secondary leading-relaxed prose prose-sm dark:prose-invert max-w-none ${wordWrap ? '' : 'whitespace-pre overflow-x-auto'}`}
                        style={{ minWidth: wordWrap ? undefined : 'max-content' }}
                    >
                        <MarkdownWithAnsi>{content || ''}</MarkdownWithAnsi>
                    </div>
                )}
            </div>
        </div>
    );

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4" onClick={onClose}>
            <div
                className="relative bg-secondary rounded-xl p-4 w-[90vw] h-[90vh] flex flex-col"
                onClick={(e) => e.stopPropagation()}
                style={{ maxWidth: '90vw', maxHeight: '90vh' }}
            >
                {/* Header */}
                <div className="flex items-center justify-between mb-0 flex-shrink-0 px-4 pt-4">
                    <div className="flex items-center gap-3 pb-4">
                        <h3 className="text-lg font-semibold truncate max-w-md flex items-center gap-2 text-primary">
                            <Eye size={18} className="text-brand-400" />
                            Vision Analysis
                        </h3>
                    </div>

                    <div className="flex items-end gap-4 h-full relative top-[1px]">
                        {/* Tab buttons */}
                        <div className="flex items-end">
                            {tabs.map((tab, idx) => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`px-4 py-2 rounded-t-lg text-sm font-medium flex items-center gap-2 transition-all border-t border-x ${idx > 0 ? 'ml-[-1px]' : ''} ${activeTab === tab.id
                                        ? 'bg-zinc-100 dark:bg-zinc-950 text-primary border-zinc-300 dark:border-zinc-800 border-b-zinc-100 dark:border-b-zinc-950 z-10'
                                        : 'bg-secondary text-secondary border-transparent hover:bg-tertiary hover:text-primary border-b-zinc-300 dark:border-b-zinc-800'
                                        }`}
                                >
                                    {tab.icon} {tab.label}
                                </button>
                            ))}
                        </div>

                        <div className="flex items-center gap-2 pb-3 mb-1 pl-4 border-l border-border/50">
                            <button className="btn-secondary p-1.5 h-8 w-8 flex items-center justify-center" onClick={handleDownload} title="Download Description">
                                <Download size={16} />
                            </button>
                            <button className="btn-secondary p-1.5 h-8 w-8 flex items-center justify-center hover:bg-red-500/20 hover:text-red-600 dark:hover:text-red-200 hover:border-red-500/50" onClick={onClose} title="Close">
                                <X size={16} />
                            </button>
                        </div>
                    </div>
                </div>

                {/* Content */}
                <div className="relative flex-1 min-h-0 overflow-hidden bg-zinc-100 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-b-lg rounded-tr-lg flex items-center justify-center p-4">
                    {activeTab === 'description' && (
                        <div className="w-full h-full">
                            {renderDescription()}
                        </div>
                    )}

                    {activeTab === 'original' && (
                        <div className="flex items-center justify-center h-full">
                            {renderOriginalMedia()}
                        </div>
                    )}

                    {activeTab === 'sideBySide' && (
                        <div className="flex flex-col md:flex-row gap-4 w-full h-full">
                            {/* Original side */}
                            <div className="w-full md:w-1/2 flex flex-col items-center gap-2 h-full justify-start min-h-0">
                                <div className="flex flex-col items-center shadow-sm">
                                    <span className="text-xs font-medium text-secondary bg-zinc-100 dark:bg-zinc-800 px-2 py-1 rounded-t border border-b-0 border-border">Original</span>
                                </div>
                                <div className="flex-1 flex items-center justify-center w-full min-h-0 overflow-hidden rounded-lg border border-border bg-zinc-50 dark:bg-zinc-900 p-2">
                                    {renderOriginalMedia("max-w-full max-h-full object-contain rounded")}
                                </div>
                            </div>

                            {/* Description side */}
                            <div className="w-full md:w-1/2 flex flex-col items-center gap-2 h-full justify-start min-h-0">
                                <div className="flex flex-col items-center shadow-sm">
                                    <span className="text-xs font-medium text-brand-600 dark:text-brand-400 bg-zinc-100 dark:bg-zinc-800 px-2 py-1 rounded-t border border-b-0 border-brand-200 dark:border-brand-500/30">Description</span>
                                </div>
                                <div className="flex-1 w-full min-h-0 overflow-hidden">
                                    {renderDescription(true)}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
