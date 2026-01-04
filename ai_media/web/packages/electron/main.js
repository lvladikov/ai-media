const { app, BrowserWindow, shell, ipcMain, dialog, Menu } = require('electron');
const path = require('path');

// Set App Name explicitly
if (!app.isPackaged) {
    app.setName('AI-Media');
}

// Suppress security warnings in dev
process.env['ELECTRON_DISABLE_SECURITY_WARNINGS'] = 'true';

const fs = require('fs');
const { spawn } = require('child_process');
const { loadConfig, saveConfig, getDefaultConfig } = require('./config-loader');

let mainWindow;
let serverProcess = null;

// Determine if we're in development or production
const isDev = !app.isPackaged;

let aboutWindow = null;

function createAboutWindow() {
  if (aboutWindow) {
    aboutWindow.focus();
    return;
  }

  aboutWindow = new BrowserWindow({
    width: 400,
    height: 380,
    title: 'About AI-Media',
    resizable: false,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,
    webPreferences: {
      nodeIntegration: true, // Enabled for this local static file to simplify logic
      contextIsolation: false
    },
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#0f172a',
    parent: mainWindow || null,
    modal: true,
    show: false
  });

  aboutWindow.loadFile(path.join(__dirname, 'about.html'));

  aboutWindow.once('ready-to-show', () => {
    // Inject dynamic versions
    const script = `
      document.getElementById('app-version').innerText = 'Version ${app.getVersion()}';
      document.getElementById('electron-version').innerText = 'Electron ${process.versions.electron}';
      document.getElementById('year').innerText = new Date().getFullYear();
    `;
    aboutWindow.webContents.executeJavaScript(script);
    aboutWindow.show();
  });

  // Handle external links in About window
  aboutWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  aboutWindow.on('closed', () => {
    aboutWindow = null;
  });
}

function createAppMenu() {
  const isMac = process.platform === 'darwin';

  const template = [
    // { role: 'appMenu' }
    ...(isMac
      ? [{
          label: app.name,
          submenu: [
            { 
              label: `About ${app.name}`,
              click: createAboutWindow
            },
            { type: 'separator' },
            { role: 'services' },
            { type: 'separator' },
            { role: 'hide' },
            { role: 'hideOthers' },
            { role: 'unhide' },
            { type: 'separator' },
            { role: 'quit' }
          ]
        }]
      : []),
    // { role: 'fileMenu' }
    {
      label: 'File',
      submenu: [
        isMac ? { role: 'close' } : { role: 'quit' }
      ]
    },
    // ... existing menus ...

    // { role: 'editMenu' }
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        ...(isMac
          ? [
              { role: 'pasteAndMatchStyle' },
              { role: 'delete' },
              { role: 'selectAll' },
              { type: 'separator' },
              {
                label: 'Speech',
                submenu: [
                  { role: 'startSpeaking' },
                  { role: 'stopSpeaking' }
                ]
              }
            ]
          : [
              { role: 'delete' },
              { type: 'separator' },
              { role: 'selectAll' }
            ])
      ]
    },
    // { role: 'viewMenu' }
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },

    // Tools Menu (Custom)
    {
      label: 'Tools',
      submenu: [
        { label: 'Generate', enabled: false },
        { 
            label: '🖼️ Image', 
            click: () => mainWindow?.webContents.send('navigate-to', 'image') 
        },
        { 
            label: '🎥 Video', 
            click: () => mainWindow?.webContents.send('navigate-to', 'video') 
        },
        { 
            label: '🎵 Audio', 
            click: () => mainWindow?.webContents.send('navigate-to', 'audio') 
        },
        { 
            label: '📄 Article', 
            click: () => mainWindow?.webContents.send('navigate-to', 'article') 
        },
        { 
            label: '💻 Code', 
            click: () => mainWindow?.webContents.send('navigate-to', 'code') 
        },
        { 
            label: '💬 Chat', 
            click: () => mainWindow?.webContents.send('navigate-to', 'chat') 
        },
        { type: 'separator' },
        { label: 'Edit', enabled: false },
        { 
            label: '✨ Transform', 
            click: () => mainWindow?.webContents.send('navigate-to', 'transform') 
        },
        { 
            label: '🔄 Convert', 
            click: () => mainWindow?.webContents.send('navigate-to', 'convert') 
        },
        { 
            label: '📈 Upscale', 
            click: () => mainWindow?.webContents.send('navigate-to', 'upscale') 
        },
        { type: 'separator' },
        { label: 'History', enabled: false },
        { 
            label: '⏱️ Jobs', 
            click: () => mainWindow?.webContents.send('navigate-to', 'jobs') 
        },
        { type: 'separator' },
        { label: 'System', enabled: false },
        { 
            label: '⚙️ Settings', 
            click: () => mainWindow?.webContents.send('navigate-to', 'settings') 
        }
      ]
    },
    // { role: 'windowMenu' }
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'zoom' },
        ...(isMac
          ? [
              { type: 'separator' },
              { role: 'front' },
              { type: 'separator' },
              { role: 'window' }
            ]
          : [
              { role: 'close' }
            ])
      ]
    },
    // Help Menu (Custom)
    {
      role: 'help',
      submenu: [
        {
          label: 'Help Guide',
          click: async () => {
             // Always target the main window for app-level navigation
             if (mainWindow && !mainWindow.isDestroyed()) {
                 mainWindow.webContents.send('navigate-to', 'help');
                 mainWindow.focus(); // Bring main window to front
             }
          }
        }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 16, y: 16 },
    backgroundColor: '#0f172a',
    show: false,
  });

  // Load the React app
  if (isDev) {
    const webPort = process.env.VITE_WEB_PORT || '5173';
    mainWindow.loadURL(`http://localhost:${webPort}`);
    // mainWindow.webContents.openDevTools();
  } else {
    const appPath = path.join(process.resourcesPath, 'app', 'index.html');
    mainWindow.loadFile(appPath);
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow.maximize();
    mainWindow.show();
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

async function startPythonServer(config) {
  let pythonPath, serverScript, serverCwd;

  if (app.isPackaged) {
    // Production: use bundled server source
    const venvPath = config.paths.python_venv || path.join(path.dirname(app.getPath('exe')), '.venv');

    if (process.platform === 'win32') {
      pythonPath = path.join(venvPath, 'Scripts', 'python.exe');
    } else {
      pythonPath = path.join(venvPath, 'bin', 'python');
    }

    serverScript = path.join(process.resourcesPath, 'server', 'ai-media.py');
    serverCwd = path.join(process.resourcesPath, 'server');
  } else {
    // Development: use config paths
    if (!config.paths.python_venv || !config.paths.ai_media) {
      console.log('Python paths not configured, skipping server start');
      return;
    }

    if (process.platform === 'win32') {
      pythonPath = path.join(config.paths.python_venv, 'Scripts', 'python.exe');
    } else {
      pythonPath = path.join(config.paths.python_venv, 'bin', 'python');
    }

    serverScript = path.join(config.paths.ai_media, 'ai-media.py');
    serverCwd = config.paths.ai_media;
  }

  if (!fs.existsSync(pythonPath)) {
    console.error('Python not found at:', pythonPath);
    return;
  }

  if (!fs.existsSync(serverScript)) {
    console.error('Server script not found at:', serverScript);
    return;
  }

  const host = config.server?.host || '127.0.0.1';
  const port = config.server?.port || 8000;

  serverProcess = spawn(pythonPath, [serverScript, '--serve-no-client', '--host', host, '--port', String(port)], {
    cwd: serverCwd,
    env: {
      ...process.env,
      HF_HOME: config.paths.hf_home || '',
    },
  });

  serverProcess.stdout.on('data', (data) => {
    console.log(`[Server] ${data}`);
  });

  serverProcess.stderr.on('data', (data) => {
    console.error(`[Server Error] ${data}`);
  });

  serverProcess.on('close', (code) => {
    console.log(`Server exited with code ${code}`);
    serverProcess = null;
  });
}

function stopPythonServer() {
  if (serverProcess) {
    serverProcess.kill();
    serverProcess = null;
  }
}

// IPC Handlers
ipcMain.handle('get-config', async () => {
  return loadConfig();
});

ipcMain.handle('save-config', async (event, config) => {
  return saveConfig(config);
});

ipcMain.handle('open-folder', async (event, filePath) => {
  shell.showItemInFolder(filePath);
});

ipcMain.handle('open-file', async (event, filePath) => {
  shell.openPath(filePath);
});

ipcMain.handle('select-directory', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
  });
  return result.canceled ? null : result.filePaths[0];
});

// App lifecycle
app.whenReady().then(async () => {
  const config = loadConfig();

  // Check if first run (config doesn't exist or is incomplete)
  if (!config.paths.python_venv || !config.paths.ai_media) {
    // Will show setup wizard in React app
    console.log('First run - setup wizard will be shown');
  }

  // Ensure media_output directory exists
  if (config.paths.media_output) {
    let mediaOutputPath = config.paths.media_output;

    // Resolve relative paths based on app location
    if (!path.isAbsolute(mediaOutputPath)) {
      if (app.isPackaged) {
        mediaOutputPath = path.join(path.dirname(app.getPath('exe')), mediaOutputPath);
      } else {
        // Dev: resolve relative to project root
        mediaOutputPath = path.resolve(mediaOutputPath);
      }
    }

    if (!fs.existsSync(mediaOutputPath)) {
      try {
        fs.mkdirSync(mediaOutputPath, { recursive: true });
        console.log('Created media_output directory:', mediaOutputPath);
      } catch (error) {
        console.error('Failed to create media_output directory:', error);
      }
    }
  }

  // Auto-start server if configured
  // BUT skip if we are in dev mode and parent process provided API port (meaning it's managing the server)
  if (config.server?.auto_start !== false && !process.env.VITE_API_PORT) {
    await startPythonServer(config);
  } else if (process.env.VITE_API_PORT) {
    // Main process managing server (VITE_API_PORT set), skipping internal start.
  }

  // Set Dock Icon for dev mode (macOS)
  if (isDev && process.platform === 'darwin') {
      const iconPath = path.join(__dirname, 'assets', 'icon.png');
      app.dock.setIcon(iconPath);
  }

  createAppMenu();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopPythonServer();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopPythonServer();
});
