import { useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ResourceBar } from './components/ResourceBar';
import { ImageGenerator } from './components/ImageGenerator';
import { VideoGenerator } from './components/VideoGenerator';
import { AudioGenerator } from './components/AudioGenerator';
import { ChatInterface } from './components/ChatInterface';
import { ArticleGenerator } from './components/ArticleGenerator';
import { CodeGenerator } from './components/CodeGenerator';
import { JobsView } from './components/JobsView';
import { SettingsView } from './components/SettingsView';
import { HelpModal } from './components/HelpModal';
import { useResourceMonitor, useSystemInfo, useJobSocket, useConfig } from './hooks/useApi';
import { useAppStore } from './store';
import { TitleBar } from './components/TitleBar';
import './index.css';

import { TransformView } from './components/TransformView';
import { ConvertView } from './components/ConvertView';
import { UpscaleView } from './components/UpscaleView';
import { VisionView } from './components/VisionView';

function MainContent() {
  const { activeTab } = useAppStore();

  const renderContent = () => {
    switch (activeTab) {
      case 'image':
        return <ImageGenerator />;
      case 'video':
        return <VideoGenerator />;
      case 'audio':
        return <AudioGenerator />;
      case 'article':
        return <ArticleGenerator />;
      case 'code':
        return <CodeGenerator />;
      case 'chat':
        return <ChatInterface />;
      case 'transform':
        return <TransformView />;
      case 'convert':
        return <ConvertView />;
      case 'upscale':
        return <UpscaleView />;
      case 'vision':
        return <VisionView />;
      case 'jobs':
        return <JobsView />;
      case 'settings':
        return <SettingsView />;
      default:
        return <ImageGenerator />;
    }
  };

  return <div className="main-content">{renderContent()}</div>;
}

import { Menu, X } from 'lucide-react';

function MobileHeader() {
  const { toggleMobileMenu, isMobileMenuOpen } = useAppStore();

  return (
    <div className="md:hidden h-14 bg-secondary border-b border-border flex items-center justify-between px-4 shrink-0 z-20 relative">
      <div className="flex items-center gap-2">
        {/* Hamburger */}
        <button onClick={toggleMobileMenu} className="text-secondary hover:text-primary p-1">
          {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
        <span className="font-semibold text-lg">AI-Media</span>
      </div>
    </div>
  );
}

function App() {
  // Initialize hooks
  useResourceMonitor();
  useSystemInfo();
  useJobSocket();
  useConfig();
  useConfig();
  const { activeTab, setActiveTab } = useAppStore();

  // Sync activeTab to URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('tab') !== activeTab) {
      const newUrl = `${window.location.pathname}?tab=${activeTab}`;
      window.history.pushState({ path: newUrl }, '', newUrl);
    }
  }, [activeTab]);

  // Listen for Electron navigation events (Menu)
  useEffect(() => {
    if ((window as any).electronAPI?.onNavigate) {
      (window as any).electronAPI.onNavigate((id: string) => {
        if (id === 'help') {
          useAppStore.getState().toggleHelp();
        } else {
          setActiveTab(id as any);
        }
      });
    }
  }, [setActiveTab]);

  return (
    <div className="flex flex-col h-screen bg-primary text-primary overflow-hidden">
      <TitleBar />
      <MobileHeader />
      <div className="flex flex-1 overflow-hidden relative">
        <Sidebar />
        <div className="flex flex-col flex-1 min-w-0 h-full overflow-hidden">
          <MainContent />
          <HelpModal />
        </div>
      </div>
      <ResourceBar />
    </div>
  );
}

export default App;
