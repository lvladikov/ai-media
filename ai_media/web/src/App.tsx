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

function App() {
  // Initialize hooks
  useResourceMonitor();
  useSystemInfo();
  useJobSocket();
  useConfig();
  const { activeTab } = useAppStore();

  // Sync activeTab to URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get('tab') !== activeTab) {
      const newUrl = `${window.location.pathname}?tab=${activeTab}`;
      window.history.pushState({ path: newUrl }, '', newUrl);
    }
  }, [activeTab]);

  return (
    <div className="flex flex-col h-screen bg-primary text-primary">
      <TitleBar />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <div className="flex flex-col flex-1 min-w-0">
          <MainContent />
          <HelpModal />
        </div>
      </div>
      <ResourceBar />
    </div>
  );
}

export default App;
