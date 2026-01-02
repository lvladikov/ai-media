import { useState, useEffect, useCallback, useRef } from 'react';
import { X, Folder, FileCode, ChevronRight, ChevronDown, Loader2, Code, File as FileIcon, GripVertical, Download } from 'lucide-react';
import JSZip from 'jszip';
import Prism from 'prismjs';
import 'prismjs/themes/prism-tomorrow.css';
import { API_BASE_URL } from '../config';

// Ensure languages are loaded (reusing imports from PreviewModal implies they are globally available or we import again)
// Import simplified set for this component or rely on global Prism if loaded elsewhere.
// Better to be safe and import common ones.
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-css';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-markdown';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-go';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-jsx';
import 'prismjs/components/prism-tsx';

interface ProjectPreviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  resultPath: string; // Path to zip on server
  zipUrl?: string;    // Optional direct URL
  sidebarWidth?: number;
  onSidebarWidthChange?: (width: number) => void;
}

interface FileNode {
  name: string;
  path: string;
  isDir: boolean;
  children: FileNode[];
  content?: string; // Loaded on demand
}

const buildFileTree = (files: string[]): FileNode[] => {
  const root: FileNode[] = [];
  
  files.forEach(path => {
    const parts = path.split('/');
    let currentLevel = root;
    
    parts.forEach((part, index) => {
      // If last part, it's a file
      const isFile = index === parts.length - 1;
      const existing = currentLevel.find(n => n.name === part);
      
      if (existing) {
        if (!isFile) {
             currentLevel = existing.children;
        }
      } else {
        const newNode: FileNode = {
          name: part,
          path: parts.slice(0, index + 1).join('/'),
          isDir: !isFile,
          children: []
        };
        currentLevel.push(newNode);
        if (!isFile) {
           currentLevel = newNode.children;
        }
      }
    });
  });
  
  // Sort: folders first, then files
  const sortNodes = (nodes: FileNode[]) => {
      nodes.sort((a, b) => {
          if (a.isDir === b.isDir) return a.name.localeCompare(b.name);
          return a.isDir ? -1 : 1;
      });
      nodes.forEach(n => {
          if (n.children.length > 0) sortNodes(n.children);
      });
  };
  
  sortNodes(root);
  return root;
};

const getFileIcon = (fileName: string) => {
    const ext = fileName.split('.').pop()?.toLowerCase();
    switch(ext) {
        case 'js': case 'jsx': case 'ts': case 'tsx': return <Code size={14} className="text-yellow-400" />;
        case 'css': case 'scss': return <Code size={14} className="text-blue-400" />;
        case 'html': return <Code size={14} className="text-orange-400" />;
        case 'json': return <Code size={14} className="text-yellow-200" />;
        case 'md': return <FileIcon size={14} className="text-slate-300" />;
        case 'py': return <Code size={14} className="text-blue-300" />;
        case 'go': return <Code size={14} className="text-cyan-400" />;
        default: return <FileCode size={14} className="text-slate-400" />;
    }
};

const getLanguage = (fileName: string) => {
  const ext = fileName.split('.').pop()?.toLowerCase() || '';
  const map: Record<string, string> = {
    'js': 'javascript', 'jsx': 'jsx',
    'ts': 'typescript', 'tsx': 'tsx',
    'py': 'python', 'md': 'markdown',
    'json': 'json', 'css': 'css',
    'html': 'html', 'go': 'go',
    'sh': 'bash', 'rs': 'rust'
  };
  return map[ext] || 'text';
};

export function ProjectPreviewModal({ 
  isOpen, 
  onClose, 
  resultPath, 
  sidebarWidth, 
  onSidebarWidthChange 
}: ProjectPreviewModalProps) {
  const [loading, setLoading] = useState(true);
  const [zip, setZip] = useState<JSZip | null>(null);
  const [fileTree, setFileTree] = useState<FileNode[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string>('');
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [isResizing, setIsResizing] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Use props if provided, otherwise local fallback
  const currentSidebarWidth = sidebarWidth || 256;
  const updateSidebarWidth = onSidebarWidthChange || (() => {});

  // Init: Fetch and Parsing
  useEffect(() => {
    if (isOpen && resultPath) {
        setLoading(true);
        const url = `${API_BASE_URL}/api/files/zip?path=${encodeURIComponent(resultPath)}`;
        
        fetch(url)
            .then(res => res.blob())
            .then(blob => JSZip.loadAsync(blob))
            .then(loadedZip => {
                setZip(loadedZip);
                
                // Build tree
                const files: string[] = [];
                loadedZip.forEach((path, entry) => {
                    if (!entry.dir) files.push(path);
                });
                
                const tree = buildFileTree(files);
                setFileTree(tree);

                // Open root folders by default
                const newExpanded = new Set<string>();
                tree.filter(n => n.isDir).forEach(n => newExpanded.add(n.path));
                setExpandedFolders(newExpanded);

                // Auto select first file
                const findFirstFile = (nodes: FileNode[]): string | null => {
                    for (const node of nodes) {
                        if (!node.isDir) return node.path;
                        const childFile = findFirstFile(node.children);
                        if (childFile) return childFile;
                    }
                    return null;
                };
                
                const first = findFirstFile(tree);
                if (first) {
                    handleFileSelect(first, loadedZip);
                } else {
                    setLoading(false);
                }
            })
            .catch(err => {
                console.error("Failed to load project zip:", err);
                setFileContent("Failed to load project files.");
                setLoading(false);
            });
    }
  }, [isOpen, resultPath]);

  // Prism Highlight
  useEffect(() => {
    if (!loading && fileContent && selectedFile) {
        Prism.highlightAll();
    }
  }, [fileContent, selectedFile, loading]);

  const handleFileSelect = async (path: string, zipInstance = zip) => {
      if (!zipInstance) return;
      
      setSelectedFile(path);
      const file = zipInstance.file(path);
      if (file) {
          try {
              const text = await file.async("string");
              setFileContent(text);
          } catch (e) {
              setFileContent("Error reading file content.");
          }
      }
      setLoading(false);
  };
  
  const toggleFolder = (path: string) => {
      const newSet = new Set(expandedFolders);
      if (newSet.has(path)) {
          newSet.delete(path);
      } else {
          newSet.add(path);
      }
      setExpandedFolders(newSet);
  };

  const handleDownloadZip = () => {
    if (!resultPath) return;
    const url = `${API_BASE_URL}/api/files/zip?path=${encodeURIComponent(resultPath)}&download=true`;
    const a = document.createElement('a');
    a.href = url;
    a.download = resultPath.split('/').pop() || 'project.zip';
    a.click();
  };

  const startResizing = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  const stopResizing = useCallback(() => {
    setIsResizing(false);
  }, []);

  const resize = useCallback((e: MouseEvent) => {
    if (isResizing && containerRef.current) {
      const containerRect = containerRef.current.getBoundingClientRect();
      const newWidth = e.clientX - containerRect.left;
      
      // Safety bounds: 10% min, 70% max (approx 150px min)
      const minWidth = Math.max(150, containerRect.width * 0.1);
      const maxWidth = containerRect.width * 0.7;
      
      if (newWidth >= minWidth && newWidth <= maxWidth) {
        updateSidebarWidth(newWidth);
      }
    }
  }, [isResizing, updateSidebarWidth]);

  useEffect(() => {
    if (isResizing) {
      window.addEventListener('mousemove', resize);
      window.addEventListener('mouseup', stopResizing);
    } else {
      window.removeEventListener('mousemove', resize);
      window.removeEventListener('mouseup', stopResizing);
    }
    return () => {
      window.removeEventListener('mousemove', resize);
      window.removeEventListener('mouseup', stopResizing);
    };
  }, [isResizing, resize, stopResizing]);

  const renderTree = (nodes: FileNode[], depth = 0) => {
      return (
          <div className="pl-2">
              {nodes.map(node => (
                  <div key={node.path}>
                      <div 
                        className={`
                            flex items-center gap-1.5 py-1 px-2 rounded cursor-pointer text-sm select-none
                            ${node.isDir ? 'text-slate-300 hover:text-white' : 
                              selectedFile === node.path ? 'bg-primary/20 text-primary' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'}
                        `}
                        onClick={() => node.isDir ? toggleFolder(node.path) : handleFileSelect(node.path)}
                        style={{ paddingLeft: `${depth * 12 + 8}px` }}
                      >
                          {node.isDir && (
                             expandedFolders.has(node.path) ? <ChevronDown size={14} /> : <ChevronRight size={14} />
                          )}
                          {!node.isDir && <span className="w-3.5" />} {/* Spacer for indent */}
                          
                          {node.isDir ? <Folder size={14} className="text-indigo-400" /> : getFileIcon(node.name)}
                          <span className="truncate">{node.name}</span>
                      </div>
                      
                      {node.isDir && expandedFolders.has(node.path) && (
                          <div>{renderTree(node.children, depth + 1)}</div>
                      )}
                  </div>
              ))}
          </div>
      );
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 rounded-xl border border-slate-700 w-full max-w-6xl h-[85vh] flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-700 bg-slate-900">
          <h2 className="text-lg font-semibold flex items-center gap-2">
             <Code size={20} className="text-primary" />
             Project Preview
          </h2>
          <div className="flex items-center gap-2">
            <button 
              onClick={handleDownloadZip}
              className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-400 hover:text-white flex items-center gap-2 text-sm"
              title="Download project ZIP"
            >
              <Download size={18} />
              <span className="hidden sm:inline text-xs font-medium">Download ZIP</span>
            </button>
            <button onClick={onClose} className="p-1 hover:bg-slate-800 rounded-lg transition-colors text-slate-400 hover:text-white">
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Body - Split View */}
        <div className="flex flex-1 min-h-0 relative" ref={containerRef}>
           
           {/* Sidebar - File Tree */}
           <div 
             className="border-r border-slate-700 bg-slate-900/50 flex flex-col min-h-0 shrink-0"
             style={{ width: `${currentSidebarWidth}px`, flexBasis: `${currentSidebarWidth}px` }}
           >
               <div className="p-3 text-xs font-semibold text-slate-500 uppercase tracking-wider border-b border-slate-800/50">
                   Explorer
               </div>
               <div className="flex-1 overflow-y-auto p-2 scrollbar-themed text-slate-400">
                   {loading && !zip ? (
                       <div className="flex flex-col items-center justify-center h-40 text-slate-500">
                           <Loader2 size={24} className="animate-spin mb-2" />
                           <span className="text-xs">Loading structure...</span>
                       </div>
                   ) : (
                       renderTree(fileTree)
                   )}
               </div>
           </div>

           {/* Resize Handle */}
           <div 
             className={`
                w-1.5 cursor-col-resize hover:bg-primary/50 transition-colors flex items-center justify-center
                ${isResizing ? 'bg-primary/70' : 'bg-transparent'}
                active:bg-primary/80 group z-10
             `}
             onMouseDown={startResizing}
           >
              <div className="w-[1px] h-8 bg-slate-700 group-hover:bg-primary/50 invisible group-hover:visible flex items-center justify-center">
                 <GripVertical size={10} className="text-slate-500 ml-[-4px]" />
              </div>
           </div>
           
           {/* Main - Code View */}
           <div className="flex-1 flex flex-col min-h-0 bg-[#1e1e1e]">
              
              {/* File Tab */}
              {selectedFile && (
                  <div className="bg-[#2d2d2d] px-4 py-2 text-sm text-slate-200 border-b border-black/20 flex items-center gap-2 shadow-sm">
                      {getFileIcon(selectedFile)}
                      <span>{selectedFile.split('/').pop()}</span>
                      <span className="text-xs text-slate-500 ml-2 opacity-50">{selectedFile}</span>
                  </div>
              )}
              
              {/* Content */}
              <div className="flex-1 overflow-auto relative scrollbar-themed">
                  {loading && !fileContent ? (
                       <div className="flex items-center justify-center h-full text-slate-500 gap-2">
                           <Loader2 size={24} className="animate-spin" />
                           <span>Loading file...</span>
                       </div>
                  ) : (
                      selectedFile ? (
                        <pre className="!m-0 !p-6 !bg-transparent text-sm min-h-full font-mono !font-normal">
                            <code className={`language-${getLanguage(selectedFile)}`}>
                                {fileContent}
                            </code>
                        </pre>
                      ) : (
                          <div className="flex items-center justify-center h-full text-slate-600 select-none">
                              Select a file to preview
                          </div>
                      )
                  )}
              </div>
           </div>
        </div>
      </div>
    </div>
  );
}
