import { useAppStore } from '../store';
import type { TabId } from '../store';
import {
  Image,
  Film,
  Music,
  FileText,
  Code,
  MessageSquare,
  Wand2,
  RefreshCw,
  TrendingUp,
  History,
  Settings,
  Zap,
  Book,
} from 'lucide-react';

const navItems: { id: TabId; label: string; icon: React.ReactNode; section: string }[] = [
  { id: 'image', label: 'Image', icon: <Image size={18} />, section: 'Generate' },
  { id: 'video', label: 'Video', icon: <Film size={18} />, section: 'Generate' },
  { id: 'audio', label: 'Audio', icon: <Music size={18} />, section: 'Generate' },
  { id: 'article', label: 'Article', icon: <FileText size={18} />, section: 'Generate' },
  { id: 'code', label: 'Code', icon: <Code size={18} />, section: 'Generate' },
  { id: 'chat', label: 'Chat', icon: <MessageSquare size={18} />, section: 'Generate' },
  { id: 'transform', label: 'Transform', icon: <Wand2 size={18} />, section: 'Edit' },
  { id: 'convert', label: 'Convert', icon: <RefreshCw size={18} />, section: 'Edit' },
  { id: 'upscale', label: 'Upscale', icon: <TrendingUp size={18} />, section: 'Edit' },
  { id: 'jobs', label: 'Jobs', icon: <History size={18} />, section: 'History' },
  { id: 'settings', label: 'Settings', icon: <Settings size={18} />, section: 'System' },
];

export function Sidebar() {
  const { activeTab, setActiveTab, isConnected } = useAppStore();
  
  // Group items by section
  const sections = navItems.reduce((acc, item) => {
    if (!acc[item.section]) acc[item.section] = [];
    acc[item.section].push(item);
    return acc;
  }, {} as Record<string, typeof navItems>);

  return (
    <div className="sidebar">

      {/* Logo */}
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <Zap className="w-6 h-6 text-primary-400" />
          <span className="text-lg font-semibold">AI-Media</span>
        </div>
        <div className="mt-1 flex items-center gap-2 text-xs text-slate-400">
          <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-yellow-400 animate-pulse'}`} />
          {isConnected ? 'Connected' : 'Connecting...'}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-2">
        {Object.entries(sections).map(([section, items]) => (
          <div key={section}>
            <div className="nav-section">{section}</div>
            {items.map((item) => (
              <div
                key={item.id}
                className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
                onClick={() => setActiveTab(item.id)}
              >
                {item.icon}
                <span>{item.label}</span>
              </div>
            ))}
          </div>
        ))}
         {/* Help Item */}
         <div>
            <div className="nav-section">Help</div>
            <div
                className="nav-item"
                onClick={() => useAppStore.getState().toggleHelp()}
            >
              <Book size={18} />
              <span>Help Guide</span>
            </div>
          </div>
      </nav>

      {/* Version */}
      <div className="p-4 border-t border-slate-700 text-xs text-slate-500">
        v1.0.0
      </div>
    </div>
  );
}
