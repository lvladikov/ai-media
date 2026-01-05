import { X, Book, Lock, ExternalLink, Image, Film, Music, FileText, Wand2, ScanEye, Monitor, Cpu, Terminal, Palette, FileType, MessageSquare, TrendingUp, Sparkles, Info, ArrowRight } from 'lucide-react';
import { useAppStore } from '../store';
import { useState, useEffect } from 'react';

// --- Shared UI Components ---

function ExternalLinkBtn({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="text-brand-600 dark:text-brand-400 hover:text-brand-500 dark:hover:text-brand-300 inline-flex items-center gap-1 hover:underline"
    >
      {children} <ExternalLink size={12} />
    </a>
  );
}

function CodeBadge({ children }: { children: React.ReactNode }) {
  return (
    <code className="bg-secondary px-1.5 py-0.5 rounded text-amber-700 dark:text-amber-200 text-sm font-mono border border-border">
      {children}
    </code>
  );
}

function GatedLock({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onClick(); }}
      className="inline-flex items-center justify-center w-5 h-5 hover:bg-yellow-500/20 rounded transition-colors cursor-pointer ml-1 relative top-[1px]"
      title="Requires Setup: Click to read Gated Models guide"
    >
      <Lock size={12} className="text-yellow-500" />
    </button>
  );
}

function SectionTitle({ icon, children }: { icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <h3 className="text-lg font-semibold text-primary mb-4 flex items-center gap-2 border-b border-border pb-2 mt-8 first:mt-0">
      <span className="text-brand-600 dark:text-brand-400">{icon}</span>
      {children}
    </h3>
  );
}

function InfoCard({ title, children, icon }: { title: string; children: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <div className="bg-primary/50 p-4 rounded-lg border border-border">
      <h4 className="text-md font-medium text-primary mb-2 flex items-center gap-2">
        {icon || <Book size={16} className="text-tertiary" />}
        {title}
      </h4>
      <div className="text-sm text-secondary space-y-2 pl-6">
        {children}
      </div>
    </div>
  );
}

function Table({ headers, rows }: { headers: string[], rows: (string | React.ReactNode)[][] }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-border my-4">
      <table className="w-full text-left text-xs">
        <thead className="bg-tertiary dark:bg-primary text-primary font-semibold">
          <tr>
            {headers.map((h, i) => <th key={i} className="p-3 border-b border-border">{h}</th>)}
          </tr>
        </thead>
        <tbody className="divide-y divide-border bg-secondary/50">
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => <td key={j} className="p-3 text-secondary">{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface HelpSectionProps {
  onNavigate: (id: string) => void;
}

// --- Section Components ---

function HelpGatedModels() {
  return (
    <div className="space-y-6 max-w-3xl text-secondary">
      <p className="text-lg text-primary">
        Some state-of-the-art models (like <CodeBadge>FLUX.1</CodeBadge> and <CodeBadge>SD 3.5</CodeBadge>) require Hugging Face authentication, but correspond to free-to-use research licenses.
      </p>

      <div className="space-y-4">
        <div className="bg-primary/50 p-4 rounded-lg border border-border">
          <h3 className="text-primary font-semibold flex items-center gap-2 mb-2">
            <span className="flex items-center justify-center w-6 h-6 rounded-full bg-secondary text-xs border border-border">1</span>
            Create Account
          </h3>
          <p className="text-sm ml-8">
            Sign up for a free <ExternalLinkBtn href="https://huggingface.co/join">Hugging Face Account</ExternalLinkBtn>.
          </p>
        </div>

        <div className="bg-primary/50 p-4 rounded-lg border border-border">
          <h3 className="text-primary font-semibold flex items-center gap-2 mb-2">
            <span className="flex items-center justify-center w-6 h-6 rounded-full bg-secondary text-xs border border-border">2</span>
            Accept Licenses
          </h3>
          <p className="text-sm ml-8 mb-4">
            Visit each model page below and click <span className="text-primary font-medium">"Agree and access repository"</span>:
          </p>
          <div className="ml-8 grid gap-2 text-sm">
            {[
              { name: 'FLUX.1-schnell', url: 'https://huggingface.co/black-forest-labs/FLUX.1-schnell' },
              { name: 'FLUX.1-dev', url: 'https://huggingface.co/black-forest-labs/FLUX.1-dev' },
              { name: 'SD 3.5 Medium', url: 'https://huggingface.co/stabilityai/stable-diffusion-3.5-medium' },
              { name: 'SD 3.5 Large/Turbo', url: 'https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo' },
              { name: 'Stable Audio Open', url: 'https://huggingface.co/stabilityai/stable-audio-open-1.0' },
              { name: 'Llama 3.1 8B', url: 'https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct' }
            ].map(model => (
              <div key={model.name} className="flex items-center justify-between bg-secondary/40 px-3 py-2 rounded border border-border/50 hover:bg-secondary/60 transition-colors">
                <span className="font-mono text-secondary">{model.name}</span>
                <ExternalLinkBtn href={model.url}>Accept License</ExternalLinkBtn>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-primary/50 p-4 rounded-lg border border-border">
          <h3 className="text-primary font-semibold flex items-center gap-2 mb-2">
            <span className="flex items-center justify-center w-6 h-6 rounded-full bg-secondary text-xs border border-border">3</span>
            Create Token
          </h3>
          <ul className="list-disc ml-8 space-y-1 text-sm">
            <li>Go to <ExternalLinkBtn href="https://huggingface.co/settings/tokens">Settings → Access Tokens</ExternalLinkBtn>.</li>
            <li>Create a new token with <strong>Read</strong> permissions.</li>
          </ul>
        </div>

        <div className="bg-primary/50 p-4 rounded-lg border border-border">
          <h3 className="text-primary font-semibold flex items-center gap-2 mb-2">
            <span className="flex items-center justify-center w-6 h-6 rounded-full bg-secondary text-xs border border-border">4</span>
            Login
          </h3>
          <p className="text-sm ml-8">
            Run <CodeBadge>hf auth login</CodeBadge> in your terminal, paste your token, and answer <CodeBadge>n</CodeBadge> to git credentials.
          </p>
        </div>
      </div>
    </div>
  );
}

function HelpImage({ onNavigate }: HelpSectionProps) {
  return (
    <div className="space-y-6 max-w-4xl text-secondary text-sm">
      <p className="text-base">
        Generate high-quality images using state-of-the-art diffusion models running locally.
        Supports SDXL, SD 1.5, SD 3.5, Flux, and Qwen-Image.
      </p>

      {/* Quick Pick Recommendations */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-emerald-500/10 border border-emerald-500/20 p-3 rounded-lg">
          <div className="text-emerald-600 dark:text-emerald-400 font-medium text-xs mb-1">🎯 First Timer</div>
          <div className="text-primary font-bold text-sm">SDXL Turbo</div>
          <div className="text-secondary text-xs">No login, good quality, fast</div>
        </div>
        <div className="bg-blue-500/10 border border-blue-500/20 p-3 rounded-lg">
          <div className="text-blue-600 dark:text-blue-400 font-medium text-xs mb-1">🍎 Mac User</div>
          <div className="text-primary font-bold text-sm">SD 3.5 Turbo</div>
          <div className="text-secondary text-xs">Best Mac performance</div>
        </div>
        <div className="bg-purple-500/10 border border-purple-500/20 p-3 rounded-lg">
          <div className="text-purple-600 dark:text-purple-400 font-medium text-xs mb-1">🎮 CUDA 24GB+</div>
          <div className="text-primary font-bold text-sm">Flux Schnell</div>
          <div className="text-secondary text-xs">SOTA realism</div>
        </div>
        <div className="bg-amber-500/10 border border-amber-500/20 p-3 rounded-lg">
          <div className="text-amber-600 dark:text-amber-400 font-medium text-xs mb-1">🔤 Text in Image</div>
          <div className="text-primary font-bold text-sm">Qwen-Image</div>
          <div className="text-secondary text-xs">Best text rendering</div>
        </div>
      </div>

      <SectionTitle icon={<Cpu size={20} />}>Model Comparison</SectionTitle>
      <Table
        headers={['Model', 'VRAM', 'RAM', 'Speed', 'Platform', 'Best For']}
        rows={[
          [
            <span className="font-bold text-primary flex items-center">SD 3.5 Turbo <GatedLock onClick={() => onNavigate('gated-models')} /></span>,
            '~19GB', '32GB+', <span className="text-emerald-600 dark:text-emerald-400">⚡ 4 steps</span>, 'Mac/CUDA', 'Default. High quality, very fast'
          ],
          [
            <span className="font-bold text-primary">SDXL Turbo</span>,
            '~8GB', '16GB+', <span className="text-emerald-600 dark:text-emerald-400">⚡ Fast</span>, 'Mac/CUDA', 'No login, good all-rounder'
          ],
          [
            <span className="font-bold text-primary">SD 1.5</span>,
            '~4GB', '8GB+', <span className="text-emerald-600 dark:text-emerald-400">⚡ Fast</span>, 'Mac/CUDA', 'Low VRAM, artistic styles'
          ],
          [
            <span className="font-bold text-primary flex items-center">SD 3.5 Medium <GatedLock onClick={() => onNavigate('gated-models')} /></span>,
            '~10GB', '24GB+', <span className="text-yellow-600 dark:text-yellow-400">🕐 30 steps</span>, 'Mac/CUDA', 'Balanced quality/speed'
          ],
          [
            <span className="font-bold text-primary flex items-center">SD 3.5 Large <GatedLock onClick={() => onNavigate('gated-models')} /></span>,
            '~19GB', '32GB+', <span className="text-orange-600 dark:text-orange-400">🕐 40 steps</span>, 'Mac/CUDA', 'Best SD quality'
          ],
          [
            <span className="font-bold text-primary flex items-center">Flux Schnell <GatedLock onClick={() => onNavigate('gated-models')} /></span>,
            '~12GB', '24GB+', <span className="text-red-600 dark:text-red-400">🐢 Slow on Mac</span>, 'CUDA best', 'SOTA realism'
          ],
          [
            <span className="font-bold text-primary flex items-center">Flux Dev <GatedLock onClick={() => onNavigate('gated-models')} /></span>,
            '~16GB', '32GB+', <span className="text-red-600 dark:text-red-400">🐢 Very slow</span>, 'CUDA best', 'Professional quality'
          ],
          [
            <span className="font-bold text-primary">Qwen-Image Auto</span>,
            '~20-40GB', '48GB+', <span className="text-yellow-600 dark:text-yellow-400">🕐 30 steps</span>, 'CUDA', 'Best text rendering'
          ],
          [
            <span className="font-bold text-primary">Qwen-Image Lightning</span>,
            '~40GB', '64GB+', <span className="text-emerald-600 dark:text-emerald-400">⚡ 4 steps</span>, 'CUDA', 'Fast text rendering'
          ],
          [
            <span className="font-bold text-primary">Qwen-Image 4-bit</span>,
            '~20GB', '32GB+', <span className="text-emerald-600 dark:text-emerald-400">⚡ Fast</span>, 'CUDA only', 'Text rendering, less VRAM'
          ],
          [
            <span className="font-bold text-primary flex items-center">FLUX.2 4-bit <GatedLock onClick={() => onNavigate('gated-models')} /></span>,
            '~12GB', '24GB+', <span className="text-yellow-600 dark:text-yellow-400">🕐 Medium</span>, 'CUDA only', 'Quantized Flux'
          ],
          [
            <span className="font-bold text-primary flex items-center">FLUX.2 Full <GatedLock onClick={() => onNavigate('gated-models')} /></span>,
            '~65GB', '128GB+', <span className="text-red-600 dark:text-red-400">🐢 Slow</span>, 'CUDA only', <span className="text-red-600 dark:text-red-400">⚠️ Extreme RAM</span>
          ],
        ]}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Key Features" icon={<Image size={16} className="text-blue-600 dark:text-blue-400" />}>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>Text-to-Image</strong>: Detailed wallpapers, art, photos.</li>
            <li><strong>Negative Prompt</strong>: List items to exclude (e.g., "blur, text").</li>
            <li><strong>Steps/CFG</strong>: Turbo models use 4 steps & 0 CFG. Standard use 30+ steps.</li>
          </ul>
        </InfoCard>

        <InfoCard title="Supported Formats" icon={<Monitor size={16} className="text-green-600 dark:text-green-400" />}>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>Resolutions</strong>: 720p, 1080p, 4k, 8k, HD, UHD.</li>
            <li><strong>Custom</strong>: Width x Height (e.g. 1024x1024).</li>
            <li><strong>Files</strong>: PNG (lossless), JPG (compressed).</li>
          </ul>
        </InfoCard>
      </div>

      <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-lg mt-4">
        <h4 className="text-blue-600 dark:text-blue-400 font-medium mb-2 flex items-center gap-2"><Monitor size={16} /> Smart Multi-Stage Strategy</h4>
        <p className="mb-2">
          For high-res requests (&gt; 6 Megapixels), we auto-generate at ~3K base then AI upscale to target.
        </p>
      </div>
    </div>
  );
}

function HelpVideo() {
  return (
    <div className="space-y-6 max-w-4xl text-secondary text-sm">
      <p className="text-base">
        Create engaging short clips using models like Zeroscope, LTX-Video, and Wan 2.2.
      </p>

      {/* Quick Pick Recommendations */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-emerald-500/10 border border-emerald-500/20 p-3 rounded-lg">
          <div className="text-emerald-600 dark:text-emerald-400 font-medium text-xs mb-1">🍎 Mac Champion</div>
          <div className="text-primary font-bold text-sm">LTX-Video</div>
          <div className="text-secondary text-xs">Best native Apple Silicon</div>
        </div>
        <div className="bg-blue-500/10 border border-blue-500/20 p-3 rounded-lg">
          <div className="text-blue-600 dark:text-blue-400 font-medium text-xs mb-1">⚡ Instant Preview</div>
          <div className="text-primary font-bold text-sm">Zeroscope</div>
          <div className="text-secondary text-xs">Fastest for quick loops</div>
        </div>
        <div className="bg-purple-500/10 border border-purple-500/20 p-3 rounded-lg">
          <div className="text-purple-600 dark:text-purple-400 font-medium text-xs mb-1">🎮 CUDA Power</div>
          <div className="text-primary font-bold text-sm">Wan 2.2</div>
          <div className="text-secondary text-xs">Photorealism (24GB VRAM)</div>
        </div>
        <div className="bg-amber-500/10 border border-amber-500/20 p-3 rounded-lg">
          <div className="text-amber-600 dark:text-amber-400 font-medium text-xs mb-1">🌊 High Motion</div>
          <div className="text-primary font-bold text-sm">Mochi 1</div>
          <div className="text-secondary text-xs">Best physics & fluidity</div>
        </div>
      </div>

      <SectionTitle icon={<Film size={20} />}>Model Comparison</SectionTitle>
      <Table
        headers={['Model', 'VRAM', 'RAM', 'Speed', 'Platform', 'Best For']}
        rows={[
          [
            <span className="font-bold text-primary">LTX-Video</span>,
            '~12GB', '16GB+', <span className="text-emerald-600 dark:text-emerald-400">⚡ Fast</span>, 'Mac/CUDA', 'Perfect speed/motion balance'
          ],
          [
            <span className="font-bold text-primary">Zeroscope</span>,
            '~4GB', '8GB+', <span className="text-emerald-600 dark:text-emerald-400">⚡ Fast</span>, 'Mac/CUDA', 'Stable, no watermark loops'
          ],
          [
            <span className="font-bold text-primary">Wan 2.2</span>,
            '~24GB', '32GB+', <span className="text-yellow-600 dark:text-yellow-400">🕐 Medium</span>, 'CUDA Best', 'State-of-the-Art realism'
          ],
          [
            <span className="font-bold text-primary">Mochi 1</span>,
            '~19GB', '32GB+', <span className="text-yellow-600 dark:text-yellow-400">🕐 Medium</span>, 'CUDA Best', 'Fluid physics & high motion'
          ],
          [
            <span className="font-bold text-primary">SVD</span>,
            '~8GB', '16GB+', <span className="text-red-600 dark:text-red-400">🐢 Slow on Mac</span>, 'Mac/CUDA', 'Image-to-Video specialist'
          ],
          [
            <span className="font-bold text-primary">CogVideoX</span>,
            '~38GB', '48GB+', <span className="text-red-600 dark:text-red-400">🐢 Heavy</span>, 'CUDA Best', 'High fidelity production'
          ],
          [
            <span className="font-bold text-primary">Hunyuan</span>,
            '~80GB+', '64GB+', <span className="text-red-600 dark:text-red-400">🐢 Very heavy</span>, 'CUDA only', 'Cinematic scale (13B)'
          ]
        ]}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8">
        <InfoCard title="Generation Modes" icon={<Film size={16} className="text-purple-600 dark:text-purple-400" />}>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>Text-to-Video</strong>: Create from prompts ("A forest drone shot").</li>
            <li><strong>Image-to-Video</strong>: Animate a starting frame image (`-ii`).</li>
            <li><strong>Audio-Reactive</strong>: Generates matching audio automatically.</li>
          </ul>
        </InfoCard>

        <InfoCard title="Platform Performance" icon={<Cpu size={16} className="text-yellow-600 dark:text-yellow-400" />}>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>Mac (MPS)</strong>: Use LTX-Video or Zeroscope for best results.</li>
            <li><strong>NVIDIA (CUDA)</strong>: Wan 2.2 and Mochi thrive on 24GB+ cards.</li>
            <li><strong>Memory</strong>: SVD/Hunyuan require high RAM regardless of GPU.</li>
          </ul>
        </InfoCard>
      </div>

      <div className="bg-secondary/50 p-4 rounded-lg border border-border mt-6">
        <h4 className="text-primary-600 dark:text-primary-400 font-medium mb-2">Zeroscope Dynamic Upscaling Pipeline</h4>
        <p className="mb-2">When generating &gt; 576x320 with Zeroscope:</p>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="bg-secondary px-2 py-1 rounded">1. Native 576x320</span>
          <span>→</span>
          <span className="bg-secondary px-2 py-1 rounded">2. Temporal Upscale (XL)</span>
          <span>→</span>
          <span className="bg-secondary px-2 py-1 rounded">3. Real-ESRGAN 2x-4x</span>
          <span>→</span>
          <span className="bg-secondary px-2 py-1 rounded">4. Final HQ MP4</span>
        </div>
      </div>
    </div>
  );
}

function HelpAudio({ onNavigate }: HelpSectionProps) {
  return (
    <div className="space-y-6 max-w-4xl text-secondary text-sm">
      <p className="text-base">
        Compose music, SFX, and realistic speech entirely offline.
      </p>

      {/* Quick Pick Recommendations */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-emerald-500/10 border border-emerald-500/20 p-3 rounded-lg">
          <div className="text-emerald-600 dark:text-emerald-400 font-medium text-xs mb-1">🎹 Music Composer</div>
          <div className="text-primary font-bold text-sm">MusicGen</div>
          <div className="text-secondary text-xs">High-fidelity melodies</div>
        </div>
        <div className="bg-blue-500/10 border border-blue-500/20 p-3 rounded-lg">
          <div className="text-blue-600 dark:text-blue-400 font-medium text-xs mb-1">🔊 SFX Specialist</div>
          <div className="text-primary font-bold text-sm">AudioLDM 2</div>
          <div className="text-secondary text-xs">Foley, weather & nature</div>
        </div>
        <div className="bg-purple-500/10 border border-purple-500/20 p-3 rounded-lg">
          <div className="text-purple-600 dark:text-purple-400 font-medium text-xs mb-1">🗣️ Voice Engine</div>
          <div className="text-primary font-bold text-sm">Bark</div>
          <div className="text-secondary text-xs">Speech with emotion</div>
        </div>
        <div className="bg-amber-500/10 border border-amber-500/20 p-3 rounded-lg flex flex-col justify-between">
          <div>
            <div className="text-amber-600 dark:text-amber-400 font-medium text-xs mb-1 flex items-center justify-between">
              <span>🎬 Cinema Score</span>
              <GatedLock onClick={() => onNavigate('gated-models')} />
            </div>
            <div className="text-primary font-bold text-sm">Stable Audio</div>
            <div className="text-secondary text-xs">Professional textures</div>
          </div>
        </div>
      </div>

      <SectionTitle icon={<Music size={20} />}>Model Comparison</SectionTitle>
      <Table
        headers={['Model', 'VRAM', 'RAM', 'Speed', 'Type', 'Best For']}
        rows={[
          [
            <span className="font-bold text-primary">MusicGen (Small)</span>,
            '~4GB', '8GB+', <span className="text-emerald-600 dark:text-emerald-400">⚡ Fast</span>, 'Music', 'Quick melodic ideas'
          ],
          [
            <span className="font-bold text-primary">MusicGen (Medium)</span>,
            '~8GB', '16GB+', <span className="text-yellow-600 dark:text-yellow-400">🕐 Medium</span>, 'Music', 'High-quality loopable music'
          ],
          [
            <span className="font-bold text-primary">AudioLDM 2</span>,
            '~6GB', '12GB+', <span className="text-emerald-600 dark:text-emerald-400">⚡ Fast</span>, 'SFX', 'Soundscapes and Foley'
          ],
          [
            <span className="font-bold text-primary">Bark</span>,
            '~6GB', '12GB+', <span className="text-yellow-600 dark:text-yellow-400">🕐 Medium</span>, 'Speech', 'Natural TTS with prosody'
          ],
          [
            <span className="font-bold text-primary">Stable Audio <GatedLock onClick={() => onNavigate('gated-models')} /></span>,
            '~12GB', '16GB+', <span className="text-emerald-600 dark:text-emerald-400">⚡ Fast</span>, 'All-in-one', 'Long textures & full songs'
          ]
        ]}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8">
        <InfoCard title="Audio Features" icon={<Wand2 size={16} className="text-violet-600 dark:text-violet-400" />}>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>Auto-Chunking</strong>: Models generate up to 30s at once; this tool automatically chains prompts for unlimited length (Bark/MusicGen).</li>
            <li><strong>Visual-to-Audio</strong>: Uses an internal "Vision" description of your image/video to prompt the audio model automatically.</li>
            <li><strong>Voice Presets</strong>: Support for 100+ native speakers in Bark (`v2/en_speaker_x`).</li>
          </ul>
        </InfoCard>

        <div className="bg-primary/50 p-4 rounded-lg border border-border">
          <h4 className="text-primary font-bold mb-2 flex items-center gap-2">
            <MessageSquare size={16} className="text-primary-600 dark:text-primary-400" />
            Bark Special Tokens
          </h4>
          <div className="flex flex-wrap gap-1.5 mb-3">
            {['[laughter]', '[cheers]', '[music]', '[sighs]', '[gasps]', '[clears throat]', '♪ lyrics ♪'].map(token => (
              <span key={token} className="px-2 py-0.5 bg-secondary border border-border rounded text-[10px] font-mono text-primary-400 dark:text-primary-300">
                {token}
              </span>
            ))}
          </div>
          <p className="text-[11px] text-tertiary">
            To trigger these sounds, simply include the token in your prompt.
            Note: Audio length depends strictly on text volume in Bark.
          </p>
        </div>
      </div>
    </div>
  );
}

function HelpText() {
  return (
    <div className="space-y-6 max-w-4xl text-secondary text-sm">
      <p className="text-base">
        Unified hub for <strong>Articles</strong>, <strong>Code</strong>, and <strong>Deep Research</strong> using LLMs.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Capabilities" icon={<FileText size={16} className="text-emerald-600 dark:text-emerald-400" />}>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>Article</strong>: Context-aware offline writing.</li>
            <li><strong>Deep Research</strong>: Autonomous web search (DuckDuckGo).</li>
            <li><strong>Chat</strong>: Local ChatGPT style with `/read` commands.</li>
            <li><strong>Code</strong>: Project scaffolding and script generation.</li>
          </ul>
        </InfoCard>

        <InfoCard title="DeepSeek R1" icon={<Cpu size={16} className="text-blue-600 dark:text-blue-400" />}>
          <p>
            We support <strong>DeepSeek R1 Distilled</strong> models (Qwen/Llama).
            These are "Reasoning" models that output a <code className="text-amber-200">&lt;think&gt;</code> process before answering.
            Excellent for complex logic, math, and coding.
          </p>
        </InfoCard>
      </div>

      <SectionTitle icon={<Terminal size={20} />}>Chat Commands</SectionTitle>
      <Table
        headers={['Command', 'Description']}
        rows={[
          [<CodeBadge>/read [file]</CodeBadge>, 'Load local file into context (e.g. valid code, logs).'],
          [<CodeBadge>/search [query]</CodeBadge>, 'Perform live web search and add results to context.'],
          [<CodeBadge>/save [name]</CodeBadge>, 'Save conversation or generated content to file.']
        ]}
      />
    </div>
  );
}

function HelpTransform() {
  return (
    <div className="space-y-6 max-w-4xl text-secondary text-sm">
      <p className="text-base">
        Modify existing images using AI-powered instructional commands or computer vision tasks.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Tools" icon={<Palette size={16} className="text-pink-600 dark:text-pink-400" />}>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>InstructPix2Pix</strong>: Stylistic edits ("Make it anime").</li>
            <li><strong>Qwen-Image-Edit</strong>: Precision text/object edits.</li>
            <li><strong>Background Removal</strong>: RMBG-1.4 (Transparent PNG).</li>
          </ul>
        </InfoCard>

        <InfoCard title="Guidance Scale" icon={<Wand2 size={16} className="text-purple-600 dark:text-purple-400" />}>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>&lt; 1.2</strong>: Creative, loose adherence to original.</li>
            <li><strong>&gt; 1.5</strong>: Strict, keeps original structure.</li>
            <li>Default is <strong>1.5</strong>.</li>
          </ul>
        </InfoCard>
      </div>

      <SectionTitle icon={<Book size={20} />}>Recipe Book</SectionTitle>
      <Table
        headers={['Goal', 'Prompt / Command']}
        rows={[
          ['Anime Style', <CodeBadge>Make it look like an anime drawing</CodeBadge>],
          ['Pixar / Disney', <CodeBadge>Make it look like a 3D Pixar character</CodeBadge>],
          ['Background Removal', <CodeBadge>--remove-background</CodeBadge>],
          ['Text Edit (Qwen)', <CodeBadge>Change text to "Hello World"</CodeBadge>],
          ['Object Removal', <CodeBadge>Remove the person in the background</CodeBadge>]
        ]}
      />
    </div>
  )
}

function HelpDescription() {
  return (
    <div className="space-y-6 max-w-4xl text-secondary text-sm">
      <p className="text-base">
        Use Vision-Language Models (VLMs) to give eyes to your AI. Describes images or videos.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Image Captioning" icon={<Image size={16} className="text-blue-600 dark:text-blue-400" />}>
          <p className="mb-2">Two models available:</p>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>Florence-2</strong> (Default): Rich detail, spatial awareness.</li>
            <li><strong>BLIP</strong>: Short, concise captions ("A dog on a bench").</li>
          </ul>
        </InfoCard>

        <InfoCard title="Video Analysis" icon={<Film size={16} className="text-purple-600 dark:text-purple-400" />}>
          <p>
            The script intelligently samples <strong>10 frames</strong> from the video, analyzes each, and synthesizes a full summary of action, setting, and flow.
          </p>
        </InfoCard>
      </div>
    </div>
  )
}

function HelpMultimedia() {
  return (
    <div className="space-y-6 max-w-4xl text-secondary text-sm">
      <p className="text-base">
        Instantly convert images, videos, audio, and documents between formats (No AI required).
      </p>

      <SectionTitle icon={<FileType size={20} />}>Document Conversion Matrix</SectionTitle>
      <Table
        headers={['From ▼ | To ▶', 'MD', 'HTML', 'PDF', 'DOCX', 'TXT', 'JSON', 'Image']}
        rows={[
          ['MD', '-', '✅', '✅', '✅', '✅', '✅', '✅'],
          ['HTML', '✅', '-', '✅', '✅', '✅', '✅', '✅'],
          ['PDF', '⚠️', '⚠️', '-', '⚠️', '✅', '⚠️', '✅'],
          ['DOCX', '✅', '✅', '✅', '-', '✅', '✅', '✅'],
          ['Image', '📷', '📷', '📷', '📷', '📷', '📷', '📷']
        ]}
      />
      <p className="text-xs text-tertiary mt-2">
        📷 = <strong>OCR (Optical Character Recognition)</strong>: Extract text from images/scans.
      </p>
      <div className="bg-primary/50 p-4 rounded-lg border border-border mt-2">
        <h4 className="text-xs font-bold text-primary mb-2 uppercase tracking-tight">OCR Model Options</h4>
        <div className="space-y-3">
          <div>
            <div className="flex justify-between items-center mb-1">
              <span className="text-xs font-bold text-primary">Qwen-VL (Default)</span>
              <span className="text-[10px] text-blue-600 dark:text-blue-400 bg-blue-400/10 px-1.5 py-0.5 rounded border border-blue-400/20">PRECISE</span>
            </div>
            <p className="text-[11px] text-secondary">~30GB RAM usage. High-precision extraction of code, paths, and 🐍 emojis.</p>
          </div>
          <div className="pt-2 border-t border-border">
            <div className="flex justify-between items-center mb-1">
              <span className="text-xs font-bold text-primary">Florence-2 (Fast Choice)</span>
              <span className="text-[10px] text-green-600 dark:text-green-400 bg-green-400/10 px-1.5 py-0.5 rounded border border-green-400/20">FAST</span>
            </div>
            <p className="text-[11px] text-secondary italic">Lightweight (~1.5GB RAM). Best for quick scans and general text drafts.</p>
          </div>
        </div>
      </div>
    </div>
  )
}


function HelpUpscale() {
  return (
    <div className="space-y-6 max-w-4xl text-secondary text-sm">
      <p className="text-base font-medium text-primary">
        Enhance the resolution and quality of images and videos using AI-powered upscaling.
        Supports multi-stage pipelines for massive billboard-sized outputs.
      </p>

      {/* Quick Pick Recommendations */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        <div className="bg-emerald-500/10 border border-emerald-500/20 p-3 rounded-lg">
          <div className="text-emerald-600 dark:text-emerald-400 font-medium text-xs mb-1 text-center">⚡ The Speed King</div>
          <div className="text-primary font-bold text-sm text-center">Fast (Real-ESRGAN)</div>
          <p className="text-secondary text-xs mt-1">Best for sharp photos and digital art. 10x faster than Latent models. Reliable and sharp.</p>
        </div>
        <div className="bg-purple-500/10 border border-purple-500/20 p-3 rounded-lg">
          <div className="text-purple-600 dark:text-purple-400 font-medium text-xs mb-1 text-center">✨ The Detailer</div>
          <div className="text-primary font-bold text-sm text-center">Creative (Latent)</div>
          <p className="text-secondary text-xs mt-1">Uses Stable Diffusion to "re-imagine" textures. Best for removing jpeg noise and adding realism.</p>
        </div>
        <div className="bg-blue-500/10 border border-blue-500/20 p-3 rounded-lg">
          <div className="text-blue-600 dark:text-blue-400 font-medium text-xs mb-1 text-center">📏 The Purist</div>
          <div className="text-primary font-bold text-sm text-center">Simple (Lanczos)</div>
          <p className="text-secondary text-xs mt-1">Mathematical scaling. No AI artifacts or hallucinations. Best when you just need a bigger container.</p>
        </div>
      </div>

      <SectionTitle icon={<TrendingUp size={20} />}>Method Comparison</SectionTitle>
      <Table
        headers={['Method', 'ModelType', 'Steps', 'Resources', 'Best For']}
        rows={[
          [
            <span className="font-bold text-primary">Fast Mode</span>,
            'Real-ESRGAN x4', '1 pass', 'Low VRAM', 'Standard photos, cleaning up art'
          ],
          [
            <span className="font-bold text-primary">Latent x2</span>,
            'SD Latent x2', <span className="text-yellow-600 dark:text-yellow-400">50 steps</span>, 'Medium VRAM', '2x detail enhancement'
          ],
          [
            <span className="font-bold text-primary">Latent x4</span>,
            'SD x4 Upscaler', <span className="text-orange-600 dark:text-orange-400">75 steps</span>, 'High VRAM', 'Massive detail, texture generation'
          ],
          [
            <span className="font-bold text-primary">Simple Mode</span>,
            'Lanczos', 'N/A', 'Low Memory', 'Clean scaling, no AI hallucinations'
          ],
        ]}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Creative Strength" icon={<Palette size={16} className="text-pink-600 dark:text-pink-400" />}>
          <p className="mb-2">The "Upscale Strength" slider controls the <strong>noise level</strong> for Latent models (Creative Mode):</p>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>0.0 - 0.1</strong>: Strict adherence to original shapes.</li>
            <li><strong>0.2 - 0.4</strong>: Rebuilds textures and adds high-frequency details.</li>
            <li><strong>&gt; 0.5</strong>: Can slightly modify faces or objects for better aesthetics.</li>
          </ul>
        </InfoCard>

        <InfoCard title="The x4 Upscaling Formula" icon={<Sparkles size={16} className="text-amber-600 dark:text-amber-400" />}>
          <p className="mb-2 text-xs">To ensure maximum quality, "Latent" upscaling uses specific iteration counts:</p>
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between text-xs bg-secondary/50 p-2 rounded border border-border">
              <span className="font-bold text-primary">x4 Pass</span>
              <span className="bg-orange-500/20 text-orange-600 dark:text-orange-400 px-2 rounded">75 Steps</span>
            </div>
            <div className="flex items-center justify-between text-xs bg-secondary/50 p-2 rounded border border-border">
              <span className="font-bold text-primary">x2 Pass</span>
              <span className="bg-yellow-500/20 text-yellow-600 dark:text-yellow-400 px-2 rounded">50 Steps</span>
            </div>
          </div>
        </InfoCard>
      </div>

      <SectionTitle icon={<Monitor size={20} />}>Smart Multi-Stage Architecture</SectionTitle>
      <div className="bg-primary/50 p-4 rounded-xl border border-border">
        <p className="mb-4 text-sm leading-relaxed">
          AI-Media uses a <strong>Proactive Strategy</strong> to prevent OOM (Out of Memory) crashes on massive targets.
          When upscaling by high factors (e.g. 8x), it intelligently sequences passes:
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs text-center">
          <div className="bg-secondary p-3 rounded-lg border border-border">
            <div className="font-bold text-primary mb-1">Pass 1: x4 AI</div>
            <div className="text-tertiary">75 Rendering Steps</div>
          </div>
          <div className="flex items-center justify-center text-tertiary">→</div>
          <div className="bg-secondary p-3 rounded-lg border border-border">
            <div className="font-bold text-primary mb-1">Pass 2: x2 AI</div>
            <div className="text-tertiary">Result is now 8x</div>
          </div>
        </div>
        <p className="mt-4 text-[11px] text-tertiary italic">
          Tip: For ultra-fast results on 4K/8K, use "Fast Mode" instead. It bypasses the multi-step diffusion process while maintaining excellent clarity.
        </p>
      </div>
    </div>
  );
}

function HelpGeneral({ onNavigate }: HelpSectionProps) {
  const tocItems = [
    { id: 'image', title: 'Image Generation', icon: <Image className="text-emerald-600 dark:text-emerald-400" size={24} />, desc: 'Flux, SDXL, SD 3.5' },
    { id: 'video', title: 'Video Generation', icon: <Film className="text-purple-600 dark:text-purple-400" size={24} />, desc: 'Wan 2.2, LTX, Zeroscope' },
    { id: 'audio', title: 'Audio Generation', icon: <Music className="text-blue-600 dark:text-blue-400" size={24} />, desc: 'MusicGen, Bark, AudioLDM' },
    { id: 'text', title: 'Chat, Articles & Research', icon: <FileText className="text-amber-600 dark:text-amber-400" size={24} />, desc: 'DeepSeek, Llama, Web Search' },
    { id: 'transform', title: 'Transformations', icon: <Wand2 className="text-pink-600 dark:text-pink-400" size={24} />, desc: 'Edit, BG Removal' },
    { id: 'upscale', title: 'Upscaling', icon: <TrendingUp className="text-cyan-600 dark:text-cyan-400" size={24} />, desc: 'AI & Lanczos enlargement' },
  ];

  return (
    <div className="space-y-8 max-w-4xl text-secondary text-sm">
      <div className="space-y-3">
        <h3 className="text-2xl font-bold text-primary flex items-center gap-3">
          <Sparkles className="text-primary-600 dark:text-primary-400" size={28} />
          Welcome to AI-Media
        </h3>
        <p className="text-base leading-relaxed text-secondary">
          AI-Media is a comprehensive local generative AI application designed to run state-of-the-art open source models entirely on your hardware.
          No subscriptions, no cloud latency, and 100% privacy.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Hardware Acceleration" icon={<Cpu size={16} className="text-primary-600 dark:text-primary-400" />}>
          <div className="space-y-4 pt-1">
            <div className="flex items-start gap-3">
              <div className="bg-secondary p-2 rounded shrink-0">🍏</div>
              <div>
                <div className="text-primary font-bold text-xs uppercase tracking-wider mb-1">Apple Silicon (MPS)</div>
                <p className="text-xs text-secondary">Optimized for M-Series unified memory. Massive models use sequential offloading to run on consumer RAM.</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="bg-secondary p-2 rounded shrink-0">🟢</div>
              <div>
                <div className="text-primary font-bold text-xs uppercase tracking-wider mb-1">NVIDIA (CUDA)</div>
                <p className="text-xs text-secondary">Full CUDA & Tensor Core support. Automatically uses BFloat16 on RTX 30xx+ for maximum fidelity.</p>
              </div>
            </div>
          </div>
        </InfoCard>

        <div className="bg-primary-500/5 border border-primary-500/20 p-5 rounded-xl flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 bg-primary-500/10 rounded-lg">
                <Lock className="text-primary-600 dark:text-primary-400" size={20} />
              </div>
              <h4 className="text-primary font-bold text-lg">Gated Models</h4>
            </div>
            <p className="text-secondary mb-4 whitespace-normal">
              State-of-the-art models like <strong>FLUX.1</strong> and <strong>Llama 3.1</strong> require a one-time
              license acceptance on Hugging Face.
            </p>
          </div>
          <button
            onClick={() => onNavigate('gated-models')}
            className="flex items-center justify-between w-full p-3 bg-primary-500/10 hover:bg-primary-500/20 border border-primary-500/30 rounded-lg text-primary-600 dark:text-primary-400 font-bold transition-all group"
          >
            Setup Guide
            <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </div>

      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Book size={18} className="text-blue-600 dark:text-blue-400" />
          <h4 className="text-primary font-bold text-lg">Table of Contents</h4>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {tocItems.map(item => (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className="flex items-center gap-4 p-4 bg-secondary/40 hover:bg-secondary/80 border border-border/50 rounded-xl transition-all text-left group"
            >
              <div className="shrink-0 transition-transform group-hover:scale-110">
                {item.icon}
              </div>
              <div>
                <div className="text-primary font-bold">{item.title}</div>
                <div className="text-[10px] text-tertiary uppercase tracking-wider">{item.desc}</div>
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="bg-primary/40 p-5 rounded-xl border border-border flex items-start gap-4">
        <div className="p-2 bg-blue-500/10 rounded-lg mt-1">
          <Info size={18} className="text-blue-600 dark:text-blue-400" />
        </div>
        <div>
          <h4 className="text-primary font-bold mb-1">Local Model Cache</h4>
          <p className="text-secondary leading-relaxed">
            All AI models are downloaded once and stored in your local cache (`HF_HOME`).
            Ensure you have enough disk space (50GB+ recommended) for a comfortable experience.
          </p>
        </div>
      </div>
    </div>
  );
}

// --- Main Modal ---

const SECTIONS = [
  { id: 'general', label: 'General Info', icon: <Book size={18} />, component: HelpGeneral },
  { id: 'gated-models', label: 'Gated Models', icon: <Lock size={18} />, component: HelpGatedModels },
  { id: 'image', label: 'Image Generation', icon: <Image size={18} />, component: HelpImage },
  { id: 'video', label: 'Video Generation', icon: <Film size={18} />, component: HelpVideo },
  { id: 'audio', label: 'Audio Generation', icon: <Music size={18} />, component: HelpAudio },
  { id: 'text', label: 'Chat, Articles & Research', icon: <FileText size={18} />, component: HelpText },
  { id: 'transform', label: 'Transformations', icon: <Wand2 size={18} />, component: HelpTransform },
  { id: 'upscale', label: 'Upscaling', icon: <TrendingUp size={18} />, component: HelpUpscale },
  { id: 'description', label: 'Vision / Description', icon: <ScanEye size={18} />, component: HelpDescription },
  { id: 'multimedia', label: 'Converters', icon: <FileType size={18} />, component: HelpMultimedia },
];

export function HelpModal() {
  const { isHelpOpen, helpSection, toggleHelp } = useAppStore();
  const [activeSection, setActiveSection] = useState(helpSection || 'general');

  // Sync activeSection when store's helpSection changes (e.g., from ModelHelpLink)
  useEffect(() => {
    if (helpSection && isHelpOpen) {
      setActiveSection(helpSection);
    }
  }, [helpSection, isHelpOpen]);

  if (!isHelpOpen) return null;

  const activeItem = SECTIONS.find(s => s.id === activeSection) || SECTIONS[2];
  const ActiveComponent = activeItem.component;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm" onClick={toggleHelp}>
      <div
        className="bg-secondary border border-border rounded-xl shadow-2xl w-full max-w-6xl h-[85vh] flex flex-col md:flex-row overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Sidebar / Top Nav */}
        <div className="w-full md:w-64 bg-primary border-b md:border-b-0 md:border-r border-border flex flex-row md:flex-col shrink-0 overflow-x-auto md:overflow-visible">
          <div className="p-4 border-r md:border-r-0 md:border-b border-border flex items-center gap-2 font-semibold text-primary shrink-0 sticky left-0 bg-primary z-10 h-16">
            <Book size={20} className="text-brand-600 dark:text-brand-400" />
            <span className="hidden md:inline">Help Guide</span>
            <span className="md:hidden">Help</span>
          </div>

          <div className="flex-1 md:overflow-y-auto p-2 flex md:block gap-2 md:gap-0 md:space-y-1">
            {SECTIONS.map((section) => (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`flex items-center gap-2 md:gap-3 px-3 py-2 rounded-lg text-sm transition-colors text-left shrink-0 md:w-full ${activeSection === section.id
                  ? 'bg-primary-500/10 text-primary-600 dark:text-primary-400 border border-primary-500/20'
                  : 'text-secondary hover:bg-tertiary hover:text-primary border border-transparent'
                  }`}
              >
                {section.icon}
                <span className="whitespace-nowrap md:truncate">{section.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 flex flex-col bg-secondary overflow-hidden min-w-0">
          <div className="flex items-center justify-between p-4 border-b border-border shrink-0 h-16">
            <h2 className="text-xl font-bold text-primary flex items-center gap-2 truncate">
              {activeItem.icon}
              <span className="truncate">{activeItem.label}</span>
            </h2>
            <button
              onClick={toggleHelp}
              className="p-1 hover:bg-tertiary rounded-lg text-secondary transition-colors shrink-0"
            >
              <X size={24} />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 md:p-8 custom-scrollbar">
            <ActiveComponent onNavigate={setActiveSection} />
          </div>
        </div>
      </div>
    </div>
  );
}
