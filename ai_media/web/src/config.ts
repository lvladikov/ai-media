// Automatically detect API base URL
// Must strictly load from config.json. No hardcoded localhost/8000 allowed.

let apiBase = '';
let isConfigLoaded = false;

export const API_BASE_URL = () => apiBase;

export async function initApiConfig() {
  if (isConfigLoaded) return;

  try {
    // In Electron/File protocol, we need to find where config.json is relative to index.html
    if (typeof window !== 'undefined' && window.location.protocol === 'file:') {
       // In dev, it might be at project root (../../config.json from dist/index.html)
       // In prod, it might be adjacent
       // We'll try a few locations
       const locations = [
         '../../config.json', // Dev (from web/dist to root)
         '../config.json',    // Prod (adjacent to resources)
         './config.json'
       ];
       
       for (const loc of locations) {
         try {
           const res = await fetch(loc);
           if (res.ok) {
             const data = await res.json();
             if (data.client && data.client.host && data.client.port) {
                // Construct URL from config
                const host = data.client.host;
                const port = data.client.port;
                apiBase = `http://${host}:${port}`;
                console.log('Loaded config from', loc, apiBase);
                isConfigLoaded = true;
                return;
             }
           }
         } catch (e) {
           // Continue to next location
         }
       }
       throw new Error("Could not find or parse config.json in any expected location.");

    } else {
       // Web / Vite Dev
       // Try fetching from the API (proxied by Vite)
       const res = await fetch('/api/config');
       if (res.ok) {
           const contentType = res.headers.get('content-type');
           if (contentType && contentType.includes('text/html')) {
              throw new Error("Received HTML instead of JSON. API proxy might be misconfigured.");
           }
           
           let data;
           try {
             data = await res.json();
           } catch (parseErr) {
             throw new Error("Failed to parse config from API: " + String(parseErr));
           }

           // backend /api/config returns the specialized config structure
           // It might return { client: { host, port } ... } OR just the config object directly.
           // Let's assume it returns the full config object similar to config.json.
           if (data.client && data.client.host && data.client.port) {
              const host = data.client.host;
              const port = data.client.port;
              apiBase = `http://${host}:${port}`;
              isConfigLoaded = true;
              return;
           } else {
               // Fallback: if API is reachable but config format differs,
               // we might assume apiBase is relative (current origin) if we are in browser.
               // But strict mode requires validation.
               throw new Error("API config missing client.host or client.port");
           }
       } else {
           throw new Error(`Could not fetch /api/config (Status: ${res.status})`);
       }
    }
  } catch (err) {
    console.error("Failed to init config", err);
    throw err; // Propagate error to main.tsx
  }
}
