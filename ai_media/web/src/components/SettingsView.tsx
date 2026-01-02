import { useState, useEffect } from 'react';
import { useAppStore } from '../store';
import { updateConfig } from '../hooks/useApi';
import { Settings as SettingsIcon, Monitor, Cpu, Palette, Moon, Sun } from 'lucide-react';
import { API_BASE_URL as API_BASE } from '../config';

export function SettingsView() {
  const { systemInfo } = useAppStore();
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Check current theme from DOM or fetch
    fetch(`${API_BASE}/api/config`)
      .then(res => res.json())
      .then(config => {
         if (config.preferences?.theme) {
             setTheme(config.preferences.theme);
         }
      })
      .catch(console.error);
  }, []);

  const handleThemeChange = async (newTheme: 'dark' | 'light') => {
    setLoading(true);
    try {
        await updateConfig({ theme: newTheme });
        setTheme(newTheme);
    } catch (e) {
        console.error(e);
    } finally {
        setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl">
      <h1 className="text-2xl font-bold mb-6 flex items-center gap-2 text-primary">
        <SettingsIcon className="text-brand-400" />
        Settings
      </h1>

      <div className="space-y-6">
      
        {/* Appearance */}
        <div className="card">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-primary">
                <Palette size={20} />
                Appearance
            </h2>
            
            <div className="flex items-center justify-between p-4 bg-tertiary rounded-lg border border-border">
                <div className="flex items-center gap-3">
                    {theme === 'dark' ? <Moon size={20} className="text-brand-400" /> : <Sun size={20} className="text-yellow-500" />}
                    <div>
                        <p className="font-medium text-primary">Interface Theme</p>
                        <p className="text-sm text-secondary">Select your preferred color scheme</p>
                    </div>
                </div>
                
                <div className="flex gap-2">
                    <button 
                        onClick={() => handleThemeChange('light')}
                        disabled={loading}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${theme === 'light' ? 'bg-brand-600 text-white' : 'bg-primary hover:bg-secondary text-secondary'}`}
                    >
                        Light
                    </button>
                    <button 
                        onClick={() => handleThemeChange('dark')}
                        disabled={loading}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${theme === 'dark' ? 'bg-brand-600 text-white' : 'bg-primary hover:bg-secondary text-secondary'}`}
                    >
                        Dark
                    </button>
                </div>
            </div>
        </div>
      
        {/* System Info */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-primary">
            <Monitor size={20} />
            System Information
          </h2>
          
          {systemInfo ? (
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-secondary">Platform:</span>
                <span className="ml-2 text-primary">{systemInfo.platform}</span>
              </div>
              <div>
                <span className="text-secondary">Python:</span>
                <span className="ml-2 text-primary">{systemInfo.python_version}</span>
              </div>
              <div>
                <span className="text-secondary">Device:</span>
                <span className="ml-2 text-primary uppercase">{systemInfo.device}</span>
              </div>
              <div>
                <span className="text-secondary">Dtype:</span>
                <span className="ml-2 text-primary">{systemInfo.dtype}</span>
              </div>
              <div>
                <span className="text-secondary">GPU:</span>
                <span className="ml-2 text-primary">{systemInfo.gpu_name || 'N/A'}</span>
              </div>
              <div>
                <span className="text-secondary">VRAM:</span>
                <span className="ml-2 text-primary">
                  {systemInfo.vram_total_gb ? `${systemInfo.vram_total_gb} GB` : 'N/A'}
                </span>
              </div>
              <div>
                <span className="text-secondary">RAM:</span>
                <span className="ml-2 text-primary">{systemInfo.ram_total_gb} GB</span>
              </div>
              <div>
                <span className="text-secondary">CUDA:</span>
                <span className={`ml-2 ${systemInfo.cuda_available ? 'text-green-500' : 'text-secondary'}`}>
                  {systemInfo.cuda_available ? 'Available' : 'Not Available'}
                </span>
              </div>
              <div>
                <span className="text-secondary">MPS (Apple):</span>
                <span className={`ml-2 ${systemInfo.mps_available ? 'text-green-500' : 'text-secondary'}`}>
                  {systemInfo.mps_available ? 'Available' : 'Not Available'}
                </span>
              </div>
            </div>
          ) : (
            <p className="text-secondary">Loading system information...</p>
          )}
        </div>

        {/* Server Settings */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-primary">
            <Cpu size={20} />
            Server Connection
          </h2>
          <div className="text-sm">
            <p className="text-secondary">API Server: <span className="text-primary">http://localhost:8000</span></p>
            <p className="text-secondary mt-2">WebSocket: <span className="text-primary">ws://localhost:8000/ws/chat</span></p>
          </div>
        </div>
      </div>
    </div>
  );
}
