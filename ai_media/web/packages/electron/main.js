const { app, BrowserWindow, shell, ipcMain, dialog } = require('electron');
const path = require('path');

// Suppress security warnings in dev
process.env['ELECTRON_DISABLE_SECURITY_WARNINGS'] = 'true';

const fs = require('fs');
const { spawn } = require('child_process');
const { loadConfig, saveConfig, getDefaultConfig } = require('./config-loader');

let mainWindow;
let serverProcess = null;

// Determine if we're in development or production
const isDev = !app.isPackaged;

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
    mainWindow.webContents.openDevTools();
  } else {
    const appPath = path.join(process.resourcesPath, 'app', 'index.html');
    mainWindow.loadFile(appPath);
  }

  mainWindow.once('ready-to-show', () => {
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
  if (!config.paths.python_venv || !config.paths.ai_media) {
    console.log('Python paths not configured, skipping server start');
    return;
  }

  const pythonPath = path.join(config.paths.python_venv, 'bin', 'python');
  const serverScript = path.join(config.paths.ai_media, 'ai-media.py');

  if (!fs.existsSync(pythonPath) || !fs.existsSync(serverScript)) {
    console.error('Python or server script not found');
    return;
  }

  const host = config.server?.host || '127.0.0.1';
  const port = config.server?.port || 8000;

  serverProcess = spawn(pythonPath, [serverScript, '--serve-no-client', '--host', host, '--port', String(port)], {
    cwd: config.paths.ai_media,
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

  // Auto-start server if configured
  if (config.server?.auto_start !== false) {
    await startPythonServer(config);
  }

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
