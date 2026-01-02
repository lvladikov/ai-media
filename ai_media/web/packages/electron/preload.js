const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Config
  getConfig: () => ipcRenderer.invoke('get-config'),
  saveConfig: (config) => ipcRenderer.invoke('save-config', config),
  
  // File operations
  openFolder: (filePath) => ipcRenderer.invoke('open-folder', filePath),
  openFile: (filePath) => ipcRenderer.invoke('open-file', filePath),
  selectDirectory: () => ipcRenderer.invoke('select-directory'),
  
  // Platform info
  platform: process.platform,
  isElectron: true,
});
