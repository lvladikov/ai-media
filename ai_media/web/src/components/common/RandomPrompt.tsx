import { Dices } from 'lucide-react';
import { PROMPTS } from '../../data/prompts';
import { Tooltip } from './Tooltip';

type PromptType = keyof typeof PROMPTS;

interface RandomPromptProps {
  type: PromptType;
  onPromptSelect: (prompt: string) => void;
  className?: string;
}

export function RandomPrompt({ type, onPromptSelect, className = "" }: RandomPromptProps) {
  const handleRandomClick = () => {
    const list = PROMPTS[type];
    if (list && list.length > 0) {
      const randomIndex = Math.floor(Math.random() * list.length);
      onPromptSelect(list[randomIndex]);
    }
  };

  return (
    <div className={`inline-flex items-center ${className}`}>
      <Tooltip content="Click here to use a random prompt">
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
