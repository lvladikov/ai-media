import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

// Try to load config.json from root
let serverConfig = {
  host: "127.0.0.1",
  port: 5173
}

try {
  const configPath = path.resolve(__dirname, '../config.json')
  if (fs.existsSync(configPath)) {
    const data = fs.readFileSync(configPath, 'utf-8')
    const config = JSON.parse(data)
    if (config.client) {
      serverConfig = {
        host: config.client.host || serverConfig.host,
        port: config.client.port || serverConfig.port
      }
    }
  }
} catch (e) {
  console.warn("Could not load ../config.json", e)
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: serverConfig.host,
    port: process.env.VITE_WEB_PORT ? parseInt(process.env.VITE_WEB_PORT) : serverConfig.port,
    proxy: {
      '/api': `http://localhost:${process.env.VITE_API_PORT || 8000}`,
      '/ws': {
        target: `ws://localhost:${process.env.VITE_API_PORT || 8000}`,
        ws: true,
      },
      '/sse': `http://localhost:${process.env.VITE_API_PORT || 8000}`,
    }
  }
})
