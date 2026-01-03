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
    // mainWindow.webContents.openDevTools();
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
