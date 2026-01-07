import { useEffect, useState, useMemo } from 'react';
import { MAJOR_LANGUAGES } from '../../data/languages';
import { Check, Globe, LayoutList, Languages, Info, Zap, Clock } from 'lucide-react';
import { ModelHelpLink } from './ModelHelpLink';
import { API_BASE_URL } from '../../config';

// Sort MAJOR_LANGUAGES (fallback for LLM models)
const MAJOR_LANGUAGES_SORTED = [...MAJOR_LANGUAGES].sort((a, b) => a.label.localeCompare(b.label));

// Module-level cache for API-fetched languages
let cachedApiLanguages: Array<{ label: string, value: string }> | null = null;
let fetchPromise: Promise<Array<{ label: string, value: string }>> | null = null;

async function fetchLanguagesFromApi(): Promise<Array<{ label: string, value: string }>> {
    // Return cached if available
    if (cachedApiLanguages) return cachedApiLanguages;

    // If already fetching, wait for that promise
    if (fetchPromise) return fetchPromise;

    // Start fetching
    fetchPromise = (async () => {
        try {
            const response = await fetch(`${API_BASE_URL()}/api/system/languages`);
            if (!response.ok) throw new Error('Failed to fetch languages');
            const data = await response.json();
            cachedApiLanguages = data.languages || [];
            return cachedApiLanguages!;
        } catch (error) {
            console.error('Failed to fetch languages from API, using fallback:', error);
            // Fallback to MAJOR_LANGUAGES if API fails
            return MAJOR_LANGUAGES_SORTED;
        } finally {
            fetchPromise = null;
        }
    })();

    return fetchPromise;
}

export const TRANSLATE_MODELS = [
    { value: 'nllb-200-3.3b', label: 'NLLB 200 (High Quality)', description: 'Fast, efficient translation for 200+ languages.', supportsAutoDetect: true },
    { value: 'nllb-200-distilled', label: 'NLLB 200 (Fast)', description: 'Lightweight and fast, great for quick translations.', supportsAutoDetect: true },
    { value: 'alma-13b', label: 'ALMA (13B)', description: 'Better nuance and tone - ideal for professional content.', supportsAutoDetect: false },
    { value: 'qwen3-8b', label: 'Qwen 3 8B', description: 'Natural-sounding output, great for conversational text.', supportsAutoDetect: false },
    { value: 'qwen3-14b', label: 'Qwen 3 14B', description: 'Superior context awareness for nuanced, creative translations.', supportsAutoDetect: false },
    { value: 'llama-3.1-8b', label: 'Llama 3.1 (8B)', description: 'Handles idioms and cultural nuances naturally.', supportsAutoDetect: false },
];

interface TranslateOptionsProps {
    /** Whether translation is enabled (checkbox checked) */
    enabled: boolean;
    /** Callback when enabled state changes */
    onEnabledChange: (enabled: boolean) => void;
    /** Currently selected translation model ID */
    selectedModel: string;
    /** Callback when model changes */
    onModelChange: (model: string) => void;
    /** Currently selected target language code */
    targetLanguage: string;
    /** Callback when target language changes */
    onLanguageChange: (language: string) => void;
    /** Whether to show the parent checkbox/toggle. If false, shows controls directly. */
    showToggle?: boolean;
    /** Optional title for the section (defaults to "Translate Output") */
    title?: string;
    /** Whether to hide the language selector (e.g. for input auto-detect) */
    hideLanguageSelector?: boolean;
    /** Whether to hide the model selector */
    hideModelSelector?: boolean;
    /** Whether to show source language selector instead of target (for input translation) */
    showSourceLanguage?: boolean;
    /** Currently selected source language code (for input translation, "auto" for auto-detect) */
    sourceLanguage?: string;
    /** Callback when source language changes */
    onSourceLanguageChange?: (language: string) => void;
    /** Info message to display above the options */
    infoMessage?: string;
}

export function TranslateOptions({
    enabled,
    onEnabledChange,
    selectedModel,
    onModelChange,
    targetLanguage,
    onLanguageChange,
    showToggle = true,
    title = "Translate Output",
    hideLanguageSelector = false,
    hideModelSelector = false,
    showSourceLanguage = false,
    sourceLanguage = "auto",
    onSourceLanguageChange,
    infoMessage,
}: TranslateOptionsProps) {

    // State for API-fetched languages
    const [apiLanguages, setApiLanguages] = useState<Array<{ label: string, value: string }>>(
        cachedApiLanguages || MAJOR_LANGUAGES_SORTED
    );

    // Fetch languages from API on mount (cached)
    useEffect(() => {
        fetchLanguagesFromApi().then(setApiLanguages);
    }, []);

    // Dynamic Language Options based on Model (NLLB uses full API list, LLMs use major languages)
    const languageOptions = useMemo(() => {
        if (selectedModel.includes('nllb')) {
            return apiLanguages;
        }
        return MAJOR_LANGUAGES_SORTED;
    }, [selectedModel, apiLanguages]);

    // Check if selected model supports auto-detect
    const selectedModelInfo = TRANSLATE_MODELS.find(m => m.value === selectedModel);
    const supportsAutoDetect = selectedModelInfo?.supportsAutoDetect ?? true;

    // Ensure target language is valid when switching models/lists
    useEffect(() => {
        // Only valid if enabled
        if (!enabled) return;

        const isValid = languageOptions.some(opt => opt.value === targetLanguage);
        if (!isValid) {
            // Default to English if current selection is invalid for new model
            onLanguageChange(languageOptions.find(l => l.value === 'eng_Latn')?.value || languageOptions[0]?.value);
        }
    }, [selectedModel, languageOptions, targetLanguage, enabled, onLanguageChange]);

    // Auto-switch from "auto" to first language when model doesn't support auto-detect
    useEffect(() => {
        if (showSourceLanguage && sourceLanguage === 'auto' && !supportsAutoDetect && onSourceLanguageChange) {
            // Switch to first available language
            onSourceLanguageChange(languageOptions[0]?.value || 'eng_Latn');
        }
    }, [selectedModel, supportsAutoDetect, sourceLanguage, showSourceLanguage, onSourceLanguageChange, languageOptions]);

    return (
        <div className="space-y-4">
            {/* Main Toggle */}
            {showToggle ? (
                <div
                    className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${enabled
                        ? 'bg-blue-50 dark:bg-blue-500/10 border-blue-200 dark:border-blue-500/50'
                        : 'bg-primary border-border hover:bg-secondary/50'}`}
                    onClick={() => onEnabledChange(!enabled)}
                >
                    <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${enabled
                        ? 'bg-blue-500 border-blue-500 text-white'
                        : 'border-tertiary bg-primary'}`}>
                        {enabled && <Check size={14} />}
                    </div>
                    <div className="flex-1">
                        <div className="font-medium text-sm flex items-center gap-2">
                            <Globe size={16} className={enabled ? 'text-blue-500' : 'text-tertiary'} />
                            {title}
                        </div>
                    </div>
                </div>
            ) : null}

            {/* Options (Conditional Render) */}
            {(enabled || !showToggle) && (
                <div className={`space-y-4 ${showToggle ? 'pl-2 border-l-2 border-border ml-2 animate-in slide-in-from-top-2 fade-in duration-200' : ''}`}>

                    {/* Info Message */}
                    {infoMessage && (
                        <div className="text-[11px] text-tertiary bg-blue-500/5 border border-blue-500/20 rounded-lg p-2.5 leading-relaxed flex items-start gap-2">
                            <Info size={13} className="mt-0.5 flex-shrink-0 text-blue-500/70" />
                            <span>{infoMessage}</span>
                        </div>
                    )}

                    {/* Model Selection - NOW FIRST */}
                    {!hideModelSelector && (
                        <div className="space-y-2">
                            <label className="text-xs font-bold text-tertiary uppercase tracking-wider flex items-center gap-2">
                                <LayoutList size={14} />
                                Translation Model
                                <ModelHelpLink section="translate#models" />
                            </label>
                            <div className="relative">
                                <select
                                    value={selectedModel}
                                    onChange={(e) => onModelChange(e.target.value)}
                                    className="w-full text-sm p-3 rounded-lg border border-border bg-primary text-secondary focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-shadow appearance-none cursor-pointer"
                                >
                                    {TRANSLATE_MODELS.map(model => (
                                        <option key={model.value} value={model.value}>{model.label}</option>
                                    ))}
                                </select>
                                <div className="absolute right-3 top-3 pointer-events-none text-tertiary">
                                    <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
                                        <path d="M2.5 4.5L6 8L9.5 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                                    </svg>
                                </div>
                            </div>
                            {/* Selected Model Description with performance indicator */}
                            <div className="text-[10px] text-tertiary px-1 flex items-center gap-1.5">
                                {selectedModelInfo?.supportsAutoDetect ? (
                                    <Zap size={11} className="text-green-500 flex-shrink-0" />
                                ) : (
                                    <Clock size={11} className="text-yellow-500 flex-shrink-0" />
                                )}
                                <span>{selectedModelInfo?.description}</span>
                            </div>
                        </div>
                    )}

                    {/* Source Language Selection - NOW SECOND (for input translation) */}
                    {showSourceLanguage && onSourceLanguageChange && (
                        <div className="space-y-2">
                            <label className="text-xs font-bold text-tertiary uppercase tracking-wider flex items-center gap-2">
                                <Languages size={14} />
                                Source Language
                            </label>
                            <div className="relative">
                                <select
                                    value={sourceLanguage}
                                    onChange={(e) => onSourceLanguageChange(e.target.value)}
                                    className="w-full text-sm p-3 rounded-lg border border-border bg-primary text-secondary focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-shadow appearance-none cursor-pointer"
                                >
                                    {/* Only show Auto Detect if model supports it */}
                                    {supportsAutoDetect && <option value="auto">🔍 Auto Detect</option>}
                                    {languageOptions.map(opt => (
                                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                                    ))}
                                </select>
                                <div className="absolute right-3 top-3 pointer-events-none text-tertiary">
                                    <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
                                        <path d="M2.5 4.5L6 8L9.5 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                                    </svg>
                                </div>
                            </div>
                            <div className="text-[10px] text-tertiary px-1">
                                {sourceLanguage === 'auto'
                                    ? `Auto-detect enabled (${languageOptions.length} languages supported)`
                                    : `Translating from ${languageOptions.find(l => l.value === sourceLanguage)?.label || 'selected language'}`
                                }
                            </div>
                        </div>
                    )}

                    {/* Target Language Selection */}
                    {!hideLanguageSelector && (
                        <div className="space-y-2">
                            <label className="text-xs font-bold text-tertiary uppercase tracking-wider flex items-center gap-2">
                                <Languages size={14} />
                                Target Language
                            </label>
                            <div className="relative">
                                <select
                                    value={targetLanguage}
                                    onChange={(e) => onLanguageChange(e.target.value)}
                                    className="w-full text-sm p-3 rounded-lg border border-border bg-primary text-secondary focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-shadow appearance-none cursor-pointer"
                                >
                                    {languageOptions.map(opt => (
                                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                                    ))}
                                </select>
                                <div className="absolute right-3 top-3 pointer-events-none text-tertiary">
                                    <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
                                        <path d="M2.5 4.5L6 8L9.5 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                                    </svg>
                                </div>
                            </div>
                            <div className="text-[10px] text-tertiary px-1 flex justify-between">
                                <span>{languageOptions.length} languages available</span>
                            </div>
                        </div>
                    )}

                </div>
            )}
        </div>
    );
}






