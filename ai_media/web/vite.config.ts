import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  let apiHost: string | undefined;
  let apiPort: number | undefined;

  // 1. Try to load values from ../../config.json
  try {
    const configPath = path.resolve(__dirname, '../../config.json');
    if (fs.existsSync(configPath)) {
      const data = fs.readFileSync(configPath, 'utf-8');
      const config = JSON.parse(data);
      if (config.client) {
         apiHost = config.client.host;
         apiPort = config.client.port;
      }
    }
  } catch (e) {
    // Error loading/parsing config
    console.error("Error loading config.json:", e);
  }

  // 2. Allow ENV overrides (passed from python script), but NO hardcoded defaults here.
  const env = loadEnv(mode, process.cwd(), '');
  
  // Python script passes VITE_API_PORT and VITE_WEB_PORT.
  // We prefer ENV if provided, otherwise fall back to file.
  
  const finalApiPort = env.VITE_API_PORT ? parseInt(env.VITE_API_PORT) : apiPort;
  const finalApiHost = apiHost; 

  if (!finalApiPort) {
     throw new Error("STRICT CONFIG ERROR: Could not determine API PORT. config.json missing or VITE_API_PORT env not set.");
  }
  if (!finalApiHost) {
     throw new Error("STRICT CONFIG ERROR: Could not determine API HOST from config.json.");
  }

  // Web Port
  const webPort = env.VITE_WEB_PORT ? parseInt(env.VITE_WEB_PORT) : 5173;
  
  // Proxy setup
  const backendUrl = `http://${finalApiHost}:${finalApiPort}`;
  const wsBackendUrl = `ws://${finalApiHost}:${finalApiPort}`;

  console.log(`[Vite] Configured Proxy -> API: ${backendUrl}`);

  return {
    base: './',
    plugins: [react()],
    server: {
      host: finalApiHost, // Bind to the configured host
      port: webPort,      // Bind to web port (from env or default)
      strictPort: true,
      proxy: {
        '/api': backendUrl,
        '/ws': {
          target: wsBackendUrl,
          ws: true,
        },
        '/sse': backendUrl,
      }
    }
  }
})
