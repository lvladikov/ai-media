import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import { initApiConfig } from './config'

const root = createRoot(document.getElementById('root')!);

initApiConfig()
  .then(() => {
    root.render(
      <StrictMode>
        <App />
      </StrictMode>,
    )
  })
  .catch((err) => {
    console.error("Critical: Failed to load configuration", err);
    root.render(
      <div style={{ padding: '2rem', fontFamily: 'sans-serif', color: '#ef4444', background: '#1e293b', height: '100vh' }}>
        <h1 style={{ marginBottom: '1rem' }}>Configuration Error</h1>
        <p>Could not load <code>config.json</code>.</p>
        <p>Please ensure <code>config.json</code> exists in the application root or usage directory.</p>
        <pre style={{ marginTop: '1rem', background: '#0f172a', padding: '1rem', borderRadius: '0.5rem', overflow: 'auto' }}>
          {err.message || String(err)}
        </pre>
      </div>
    )
  });
