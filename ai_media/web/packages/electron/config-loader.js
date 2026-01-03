const fs = require('fs');
const path = require('path');
const { app } = require('electron');

// Config file path (next to the executable in production, or in project root in dev)
function getConfigPath() {
  if (app.isPackaged) {
    if (process.platform === 'darwin') {
      // Production macOS: config.json is next to the AI-Media.app bundle
      // Executable is at AI-Media.app/Contents/MacOS/AI-Media
      return path.join(path.dirname(app.getPath('exe')), '..', '..', '..', 'config.json');
    }
    // Production Windows/Linux: config.json is next to the executable
    return path.join(path.dirname(app.getPath('exe')), 'config.json');
  } else {
    // Development: use project root (electron -> packages -> web -> ai_media -> root)
    return path.join(__dirname, '..', '..', '..', '..', 'config.json');
  }
}

function getDefaultConfig() {
  return {
    paths: {
      hf_home: '',
      python_venv: '',
      ai_media: '',
      ffmpeg: '',
      media_output: '',
    },
    server: {
      host: '127.0.0.1',
      port: 8000,
      auto_start: true,
    },
    preferences: {
      theme: 'dark',
    },
  };
}

function loadConfig() {
  const configPath = getConfigPath();

  try {
    if (fs.existsSync(configPath)) {
      const content = fs.readFileSync(configPath, 'utf-8');
      const config = JSON.parse(content);
      // Merge with defaults to ensure all fields exist
      return { ...getDefaultConfig(), ...config };
    }
  } catch (error) {
    console.error('Failed to load config:', error);
  }

  return getDefaultConfig();
}

function saveConfig(config) {
  const configPath = getConfigPath();

  try {
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
    return true;
  } catch (error) {
    console.error('Failed to save config:', error);
    return false;
  }
}

module.exports = {
  loadConfig,
  saveConfig,
  getDefaultConfig,
  getConfigPath,
};
