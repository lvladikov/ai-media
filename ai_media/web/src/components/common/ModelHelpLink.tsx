import { HelpCircle } from 'lucide-react';
import { useAppStore } from '../../store';

interface ModelHelpLinkProps {
  section: 'image' | 'video' | 'audio' | 'text' | 'transform' | 'upscale' | 'multimedia' | 'vision';
}

export function ModelHelpLink({ section }: ModelHelpLinkProps) {
  const { openHelpSection } = useAppStore();

  return (
    <button
      onClick={() => openHelpSection(section)}
      className="inline-flex items-center justify-center ml-1 p-0.5 hover:bg-primary-500/20 rounded transition-colors cursor-pointer group"
      title="Click to get help choosing the right model"
    >
      <HelpCircle
        size={15}
        className="text-primary-400 group-hover:text-primary-300 transition-colors"
      />
    </button>
  );
}
