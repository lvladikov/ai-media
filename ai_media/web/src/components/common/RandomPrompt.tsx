import { Dices } from 'lucide-react';
import { PROMPTS } from '../../data/prompts';
import { Tooltip } from './Tooltip';

type PromptType = keyof typeof PROMPTS;

interface RandomPromptProps {
  type: PromptType;
  onPromptSelect: (prompt: string) => void;
  className?: string;
  /** Strip Bark-specific tokens like [laughs], [gasps], etc. */
  stripBarkTokens?: boolean;
}

/** Removes Bark special tokens (e.g. [laughs], [gasps]) from a prompt */
function removeBarkTokens(prompt: string): string {
  // Match common Bark tokens: [word] or [word word] patterns
  return prompt
    .replace(/\[(laughter|laughs?|cheers?|music|sighs?|gasps?|groans?|coughs?|clears throat|nervous laugh|shouts?|whispers?|hesitates?|alarm|buzzer|explosion|beep|wind|Short pause)\]/gi, '')
    .replace(/\s{2,}/g, ' ')  // Clean up double spaces
    .trim();
}

export function RandomPrompt({ type, onPromptSelect, className = "", stripBarkTokens = false }: RandomPromptProps) {
  const handleRandomClick = () => {
    const list = PROMPTS[type];
    if (list && list.length > 0) {
      const randomIndex = Math.floor(Math.random() * list.length);
      let prompt = list[randomIndex];

      // Strip Bark tokens if requested
      if (stripBarkTokens) {
        prompt = removeBarkTokens(prompt);
      }

      onPromptSelect(prompt);
    }
  };

  return (
    <div className={`inline-flex items-center ${className}`}>
      <Tooltip content="Click here to use a random prompt" align="right">
        <button
          onClick={handleRandomClick}
          className="p-1.5 text-primary-400 hover:text-primary-300 hover:bg-primary-500/10 rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500/50"
          type="button"
          aria-label="Generate Random Prompt"
        >
          <Dices size={18} />
        </button>
      </Tooltip>
      <span className="sr-only">Random Prompt</span>
    </div>
  );
}
