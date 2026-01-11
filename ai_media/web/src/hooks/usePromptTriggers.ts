import { useState, useEffect } from 'react';
import { fetchPromptsConfig } from '../api/config';

export function usePromptTriggers() {
  const [triggers, setTriggers] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    const loadTriggers = async () => {
      const config = await fetchPromptsConfig();
      if (mounted) {
        setTriggers(config.triggers);
        setIsLoading(false);
      }
    };

    loadTriggers();

    return () => {
      mounted = false;
    };
  }, []);

  return { triggers, isLoading };
}
