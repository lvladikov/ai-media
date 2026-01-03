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
  const { activeTab, setActiveTab, isConnected, isHelpOpen, isMobileMenuOpen, setMobileMenuOpen } = useAppStore();
  
  // Close menu on navigation on mobile
  const handleNavClick = (id: TabId | 'help') => {
      if (id === 'help') useAppStore.getState().toggleHelp();
      else setActiveTab(id as TabId);
      
      if (window.innerWidth < 768) {
          setMobileMenuOpen(false);
      }
  };

  // Group items by section
  const sections = navItems.reduce((acc, item) => {
    if (!acc[item.section]) acc[item.section] = [];
    acc[item.section].push(item);
    return acc;
  }, {} as Record<string, typeof navItems>);

  return (
    <>
        {/* Mobile Overlay */}
        {isMobileMenuOpen && (
            <div 
                className="fixed inset-0 bg-black/50 z-30 md:hidden"
                onClick={() => setMobileMenuOpen(false)}
            />
        )}

        <div className={`
            sidebar 
            ${isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
            fixed md:relative z-40 h-full md:h-auto
            border-r md:border-none border-border shadow-xl md:shadow-none
        `}>
          {/* The Separator Line (Desktop only) */}
          <div className="hidden md:block sidebar-border" />
    
          {/* Logo (Hidden on mobile as it's in header) */}
          <div className="hidden md:block p-4 border-b border-border z-10 relative">
            <div className="flex items-center gap-2">
              <Zap className="w-6 h-6 text-primary-400" />
              <span className="text-lg font-semibold text-primary">AI-Media</span>
            </div>
            <div className="mt-1 flex items-center gap-2 text-xs text-secondary">
              <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-yellow-400 animate-pulse'}`} />
              {isConnected ? 'Connected' : 'Connecting...'}
            </div>
          </div>
    
          {/* Mobile Status (since logo is hidden) */}
          <div className="md:hidden p-4 border-b border-border z-10 relative bg-secondary">
             <div className="flex items-center gap-2 text-xs text-secondary">
              <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-yellow-400 animate-pulse'}`} />
              {isConnected ? 'Connected' : 'Connecting...'}
            </div>
          </div>
    
          {/* Navigation */}
          <nav className="flex-1 overflow-y-auto pt-2 z-10 relative scrollbar-themed">
            {Object.entries(sections).map(([section, items]) => (
              <div key={section}>
                <div className="nav-section">{section}</div>
                {items.map((item) => (
                  <div
                    key={item.id}
                    className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
                    onClick={() => handleNavClick(item.id)}
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
                    className={`nav-item ${isHelpOpen ? 'active' : ''}`}
                    onClick={() => handleNavClick('help')}
                >
                  <Book size={18} />
                  <span>Help Guide</span>
                </div>
              </div>
          </nav>
    
          {/* Version */}
          <div className="p-4 border-t border-border text-xs text-tertiary z-10 relative bg-secondary">
            v1.0.0
          </div>
        </div>
    </>
  );
}
