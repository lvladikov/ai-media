import { useEffect, useRef } from 'react';
import { useAppStore } from '../store';
import type { ResourceStats } from '../store';
import { API_BASE_URL as API_BASE } from '../config';

export interface ModelInfo {
  name: string;
  is_default?: boolean;
}

/**
 * Hook to connect to the SSE resource stream
 * Uses fast health check first, then establishes SSE
 */
export function useResourceMonitor() {
  const { setResources, setConnected } = useAppStore();
  const eventSourceRef = useRef<EventSource | null>(null);
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    // Quick health check to set connected status fast
    const quickHealthCheck = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/health`, { method: 'GET' });
        if (res.ok) {
          setConnected(true);
        }
      } catch {
        // Server not ready yet, SSE will retry
      }
    };

    const connect = () => {
      // Clear any pending retry
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
        retryTimeoutRef.current = null;
      }

      eventSourceRef.current = new EventSource(`${API_BASE}/sse/resources`);

      eventSourceRef.current.onopen = () => {
        setConnected(true);
      };

      eventSourceRef.current.onmessage = (event) => {
        try {
          const data: ResourceStats = JSON.parse(event.data);
          setResources(data);
          setConnected(true); // Ensure connected on every message
        } catch (e) {
          console.error('Failed to parse resource data:', e);
        }
      };

      eventSourceRef.current.onerror = () => {
        setConnected(false);
        eventSourceRef.current?.close();
        // Retry connection faster (1 second)
        retryTimeoutRef.current = setTimeout(connect, 1000);
      };
    };

    // Start with quick health check, then open SSE
    quickHealthCheck();
    connect();

    return () => {
      eventSourceRef.current?.close();
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
    };
  }, [setResources, setConnected]);
}

/**
 * Hook to fetch system info on mount
 */
export function useSystemInfo() {
  const { setSystemInfo } = useAppStore();

  useEffect(() => {
    fetch(`${API_BASE}/api/system`)
      .then((res) => res.json())
      .then((data) => setSystemInfo(data))
      .catch((err) => console.error('Failed to fetch system info:', err));
  }, [setSystemInfo]);
}

/**
 * Hook to connect to the Job WebSocket
 */
export function useJobSocket() {
  const { setJobs, updateJob, addJob } = useAppStore();
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let isUnmounting = false;

    const connect = () => {
      // Clear any pending retry
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }

      // Create WebSocket connection
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      // Use string cast to avoid TS getting confused if API_BASE is empty literal
      const baseStr = (API_BASE as string);
      const wsUrl = baseStr 
        ? `${baseStr.replace('http', 'ws')}/ws/jobs`
        : `${wsProtocol}//${window.location.host}/ws/jobs`;
      const ws = new WebSocket(wsUrl);
      socketRef.current = ws;

      ws.onopen = () => {
        // console.log("Job socket connected");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'job_list') {
             // Initial load
             setJobs(data.jobs);
          } else if (data.type === 'job_update') {
             const job = data.job;
             // Check if we already have this job
             const exists = useAppStore.getState().jobs.some(j => j.job_id === job.job_id);
             if (exists) {
                updateJob(job.job_id, job);
             } else {
                addJob(job);
             }
          }
        } catch (e) {
          console.error("Failed to parse job socket message:", e);
        }
      };

      ws.onerror = (err) => {
        // Only log if not immediately closed (suppress dev warning)
        if (ws.readyState !== WebSocket.CLOSED && ws.readyState !== WebSocket.CLOSING) {
            console.error("Job socket error:", err);
        }
      };

      ws.onclose = () => {
        console.log("Job socket closed");
        socketRef.current = null;
        // Only reconnect if we haven't been unmounted/cleaned up
        if (!isUnmounting) {
            console.log("Reconnecting job socket in 2s...");
            reconnectTimeoutRef.current = setTimeout(connect, 2000);
        }
      };
    };

    // Small delay to prevent double-connect in Strict Mode
    connectTimeoutRef.current = setTimeout(connect, 100);

    return () => {
      isUnmounting = true;
      if (socketRef.current) {
        // Prevent event handlers from firing during cleanup
        socketRef.current.onopen = null;
        socketRef.current.onmessage = null;
        socketRef.current.onerror = null;
        socketRef.current.onclose = null;
        
        socketRef.current.close();
        socketRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (connectTimeoutRef.current) {
        clearTimeout(connectTimeoutRef.current);
      }
    };
  }, [setJobs, updateJob, addJob]);
}

/**
 * API helper for generation requests
 */
export async function generateImage(params: {
  prompt: string;
  model?: string;
  width?: number;
  height?: number;
  steps?: number;
  guidance_scale?: number;
  force?: boolean;
}) {
  const response = await fetch(`${API_BASE}/api/generate/image`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return response.json();
}

export async function generateVideo(params: {
  prompt: string;
  model?: string;
  width?: number;
  height?: number;
  duration?: number;
}) {
  const response = await fetch(`${API_BASE}/api/generate/video`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return response.json();
}

export async function generateAudio(params: {
  prompt: string;
  model?: string;
  duration?: number;
}) {
  const response = await fetch(`${API_BASE}/api/generate/audio`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return response.json();
}



export async function generateArticle(params: {
  topic: string;
  model?: string;
  format?: string;
  online?: boolean; // For research mode
  length?: string;  // short, medium, long
  research_iterations?: number;
  max_images?: number;
  output_filename?: string;
}) {
  const response = await fetch(`${API_BASE}/api/generate/article`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return response.json();
}

export async function generateCode(params: {
  prompt: string;
  output_name?: string;
  model?: string;
}) {
  const response = await fetch(`${API_BASE}/api/generate/code`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return response.json();
}

export async function getJobStatus(jobId: string) {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  return response.json();
}

export async function cancelJob(jobId: string) {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error('Failed to cancel job');
  }
  return response.json();
}

export async function fetchModels() {
  const response = await fetch(`${API_BASE}/api/models`);
  return response.json();
}

/**
 * Hook to apply theme from config
 */
export function useConfig() {
  useEffect(() => {
    fetch(`${API_BASE}/api/config`)
      .then(res => res.json())
      .then(config => {
        if (config.preferences?.theme === 'dark') {
          document.documentElement.classList.add('dark');
        } else {
          document.documentElement.classList.remove('dark');
        }
      })
      .catch(err => console.error("Failed to load config:", err));
  }, []);
}

export async function updateConfig(params: { theme?: string }) {
  const response = await fetch(`${API_BASE}/api/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  
  if (response.ok) {
     const data = await response.json();
     // Apply immediately
     if (params.theme) {
        if (params.theme === 'dark') {
          document.documentElement.classList.add('dark');
        } else {
          document.documentElement.classList.remove('dark');
        }
     }
     return data;
  }
  throw new Error('Failed to update config');
}
