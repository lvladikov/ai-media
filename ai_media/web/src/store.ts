import { create } from 'zustand';

// --- Types ---

export type TabId =
  | 'image' | 'video' | 'audio' | 'article' | 'code' | 'chat' | 'vision'
  | 'transform' | 'convert' | 'upscale' | 'jobs' | 'settings';

export interface Job {
  job_id: string;
  type: string;
  status: 'pending' | 'loading' | 'generating' | 'complete' | 'failed' | 'cancelled';
  progress: number;
  phase: string;
  message: string;
  logs?: string[];  // Accumulated log messages for display
  result_path: string | null;
  is_multi_file?: boolean;  // True if result is a multi-file project (ZIP)
  generated_files?: string[];  // List of files in multi-file project
  error: string | null;
  created_at: string;
  updated_at: string;
  // Optional metadata for display
  prompt?: string;
  model?: string;
  params?: Record<string, string | number | boolean>;
  generation_started_at?: string;
  reasoning?: string; // Optional reasoning/thinking block from R1 models
}


export interface ResourceStats {
  global: {
    cpu_percent: number;
    ram_used_gb: number;
    ram_total_gb: number;
    swap_used_gb?: number;
    swap_total_gb?: number;
    vram_used_gb: number;
    vram_total_gb: number;
    gpu_percent: number;
  };
  process: {
    pid: number;
    cpu_percent: number;
    ram_used_gb: number;
    vram_used_gb?: number;
    gpu_percent?: number;
  };
  timestamp: string;
}

export interface SystemInfo {
  device: string;
  dtype: string;
  cuda_available: boolean;
  mps_available: boolean;
  gpu_name: string | null;
  vram_total_gb: number | null;
  ram_total_gb: number;
  platform: string;
  python_version: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoning?: string; // Optional reasoning block from R1 models
  thinkingTime?: string; // Time taken to generate response (e.g., "45s", "2m 30s")
  timestamp?: string;
}

// --- Store ---

interface AppState {
  // Navigation
  activeTab: TabId;
  setActiveTab: (tab: TabId) => void;

  // System
  systemInfo: SystemInfo | null;
  setSystemInfo: (info: SystemInfo) => void;

  // Resources
  resources: ResourceStats | null;
  setResources: (stats: ResourceStats) => void;

  // Jobs
  jobs: Job[];
  setJobs: (jobs: Job[]) => void;
  addJob: (job: Job) => void;
  updateJob: (jobId: string, updates: Partial<Job>) => void;
  removeJob: (jobId: string) => void;

  // Chat
  chatMessages: ChatMessage[];
  chatSessionId: string | null;
  addChatMessage: (message: ChatMessage) => void;
  setChatSessionId: (id: string) => void;
  clearChat: () => void;

  // Help
  isHelpOpen: boolean;
  helpSection: string | null;
  toggleHelp: () => void;
  openHelpSection: (section: string) => void;

  // Mobile Menu
  isMobileMenuOpen: boolean;
  toggleMobileMenu: () => void;
  setMobileMenuOpen: (isOpen: boolean) => void;

  // Server connection
  isConnected: boolean;
  setConnected: (connected: boolean) => void;
}

// Helper to get initial tab from URL
const getInitialTab = (): TabId => {
  if (typeof window === 'undefined') return 'image';
  const params = new URLSearchParams(window.location.search);
  const tab = params.get('tab');
  const validTabs: TabId[] = [
    'image', 'video', 'vision', 'audio', 'article', 'code', 'chat',
    'transform', 'convert', 'upscale', 'jobs', 'settings'
  ];
  return (validTabs.includes(tab as TabId) ? tab as TabId : 'image');
};

export const useAppStore = create<AppState>((set) => ({
  // Navigation
  activeTab: getInitialTab(),
  setActiveTab: (tab) => set({ activeTab: tab }),

  // System
  systemInfo: null,
  setSystemInfo: (info) => set({ systemInfo: info }),

  // Resources
  resources: null,
  setResources: (stats) => set({ resources: stats }),

  // Jobs
  // Jobs
  jobs: [],
  setJobs: (jobs) => set({ jobs }),
  addJob: (job) => set((state) => {
    if (state.jobs.some(j => j.job_id === job.job_id)) return state;
    return { jobs: [job, ...state.jobs] };
  }),
  updateJob: (jobId, updates) => set((state) => ({
    jobs: state.jobs.map((j) => j.job_id === jobId ? { ...j, ...updates } : j),
  })),
  removeJob: (jobId) => set((state) => ({
    jobs: state.jobs.filter((j) => j.job_id !== jobId),
  })),

  // Chat
  chatMessages: [],
  chatSessionId: null,
  addChatMessage: (message) => set((state) => ({
    chatMessages: [...state.chatMessages, { ...message, timestamp: new Date().toISOString() }],
  })),
  setChatSessionId: (id) => set({ chatSessionId: id }),
  clearChat: () => set({ chatMessages: [], chatSessionId: null }),

  // Help
  isHelpOpen: false,
  helpSection: null,
  toggleHelp: () => set((state) => ({ isHelpOpen: !state.isHelpOpen, helpSection: state.isHelpOpen ? null : state.helpSection })),
  openHelpSection: (section) => set({ isHelpOpen: true, helpSection: section }),

  // Mobile Menu
  isMobileMenuOpen: false,
  toggleMobileMenu: () => set((state) => ({ isMobileMenuOpen: !state.isMobileMenuOpen })),
  setMobileMenuOpen: (isOpen) => set({ isMobileMenuOpen: isOpen }),

  // Server connection
  isConnected: false,
  setConnected: (connected) => set({ isConnected: connected }),
}));
