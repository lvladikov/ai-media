import { API_BASE_URL } from '../config';

export interface PromptsConfig {
  triggers: string[];
}

// Simple in-memory cache
let cachedPromptsConfig: PromptsConfig | null = null;

export const fetchPromptsConfig = async (): Promise<PromptsConfig> => {
  if (cachedPromptsConfig) {
    return cachedPromptsConfig;
  }

  try {
    const response = await fetch(`${API_BASE_URL()}/api/config/prompts`);
    if (!response.ok) {
      throw new Error(`Failed to fetch prompts config: ${response.statusText}`);
    }
    const data = await response.json();
    cachedPromptsConfig = data;
    return data;
  } catch (error) {
    console.error('Error fetching prompts config:', error);
    // Return empty config if API fails - do not fallback to hardcoded strings
    return {
      triggers: []
    };
  }
};
