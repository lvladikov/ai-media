import {
  X, Book, Lock, ExternalLink, Image, Film, Music, FileText, Wand2, ScanEye, Monitor, Cpu, Terminal, Palette, FileType, MessageSquare, TrendingUp, Sparkles, Info, ArrowRight, Globe, Languages, Search,
  Check,
  LayoutList,
  Mic,
  Code,
  Trash2,
  FolderOpen,
  HardDrive,
} from 'lucide-react';
import { useAppStore } from '../store';
import { useState, useEffect } from 'react';
import { ALL_LANGUAGES, SEAMLESS_LANGUAGES, LLM_LANGUAGES, ALMA_LANGUAGES } from '../data/languages';


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
        Supports SDXL, SD 1.5, SD 3.5, Flux, Z-Image, and Qwen-Image.
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
          <div className="text-primary font-bold text-sm">Z-Image Turbo</div>
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

      <SectionTitle icon={<Cpu size={20} />}>Platform Defaults</SectionTitle>
      <Table
        headers={['Platform', 'Default Precision', 'Default Framework', 'Notes']}
        rows={[
          [<span className="font-bold text-primary">CUDA (NVIDIA)</span>, <CodeBadge>float16</CodeBadge>, 'PyTorch', 'Standard backend'],
          [<span className="font-bold text-primary">MPS (Mac)</span>, <CodeBadge>float16</CodeBadge>, 'PyTorch', 'Standard backend'],
          [<span className="font-bold text-primary">MLX (Mac)</span>, <CodeBadge>int4</CodeBadge>, 'MLX (Native)', <span className="text-emerald-500 font-bold">Fastest for Z-Image</span>]
        ]}
      />

      <SectionTitle icon={<Cpu size={20} />}>Model Comparison</SectionTitle>
      <Table
        headers={['Model', 'VRAM', 'RAM', 'Speed', 'Platform', 'Best For']}
        rows={[
          [
            <span className="font-bold text-primary flex items-center">Z-Image Turbo</span>,
            '~8GB', '16GB+', <span className="text-emerald-600 dark:text-emerald-400">⚡ 9 steps</span>, 'MLX/CUDA/MPS', <span><strong>Default</strong>. Alibaba model, fast, high quality</span>
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
            <span className="font-bold text-primary flex items-center">SD 3.5 Turbo <GatedLock onClick={() => onNavigate('gated-models')} /></span>,
            '~19GB', '32GB+', <span className="text-emerald-600 dark:text-emerald-400">⚡ 4 steps</span>, 'Mac/CUDA', 'High quality, very fast'
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
            <li><strong>Negative Prompt</strong>: List items to exclude. <em>Note: Z-Image/Turbo/Flux ignore this and CFG — use "without/avoid" in your prompt.</em></li>
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

      <div className="bg-amber-500/10 border border-amber-500/20 p-4 rounded-lg mt-4 text-xs">
        <h4 className="text-amber-600 dark:text-amber-400 font-medium mb-2 flex items-center gap-2"><Cpu size={16} /> PyTorch Framework Note</h4>
        <p>
          For <strong>Z-Image Turbo</strong> and <strong>SD 3.5 Turbo</strong>, the <CodeBadge>bfloat16</CodeBadge> precision is automatically enforced when using the PyTorch framework to ensure numerical stability and avoid black image outputs on Apple Silicon (MPS).
        </p>
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

      <SectionTitle icon={<Cpu size={20} />}>Platform Defaults</SectionTitle>
      <Table
        headers={['Platform', 'Default Precision', 'Default Framework', 'Notes']}
        rows={[
          [<span className="font-bold text-primary">CUDA (NVIDIA)</span>, <CodeBadge>float16</CodeBadge>, 'PyTorch', 'Standard backend'],
          [<span className="font-bold text-primary">MPS (Mac)</span>, <CodeBadge>float16</CodeBadge>, 'PyTorch', 'Legacy default / Fallback'],
          [<span className="font-bold text-primary">MLX (Mac)</span>, <CodeBadge>int4</CodeBadge>, 'MLX (Native)', <span className="text-emerald-500 font-bold">Recommended for Speed</span>]
        ]}
      />

      <SectionTitle icon={<Film size={20} />}>Model Comparison</SectionTitle>
      <Table
        headers={['Model', 'Variant', 'MLX Support', 'VRAM', 'Best For']}
        rows={[
          [
            <span className="font-bold text-primary">Wan 2.2</span>,
            '14B (T2V)',
            <span className="text-emerald-500 font-bold">✅ Native (int4)</span>, '~16GB', 'SOTA Cinematic (Text-to-Video)'
          ],
          [
            <span className="font-bold text-primary">Wan 2.2</span>,
            '5B (T2V/I2V)',
            <span className="text-emerald-500 font-bold">✅ Native (int4)</span>, '~6-8GB', 'Fast, efficient, supports I2V'
          ],
          [
            <span className="font-bold text-primary">LTX-Video</span>,
            'Standard',
            <span className="text-emerald-500 font-bold">✅ Native (int4)</span>, '~12GB', 'Fastest Mac Native (DiT)'
          ],
          [
            <span className="font-bold text-primary">HunyuanVideo</span>,
            '13B (T2V)',
            <span className="text-emerald-500 font-bold">✅ Native (int4)</span>, '~24GB', 'Massive Scale / Production'
          ],
          [
            <span className="font-bold text-primary">CogVideoX</span>,
            '5B',
            <span className="text-emerald-500 font-bold">✅ Native (int4)</span>, '~18GB', 'High Fidelity'
          ],
          [
            <span className="font-bold text-primary">Mochi 1</span>,
            'Preview',
            <span className="text-red-500 font-bold">❌ No (PyTorch)</span>, '~20GB', 'Fluid Mechanics / High Motion'
          ],
          [
            <span className="font-bold text-primary">Zeroscope</span>,
            'v2 576w',
            <span className="text-red-500 font-bold">❌ No (PyTorch)</span>, '~4GB', 'Fast loops, no watermark'
          ],
          [
            <span className="font-bold text-primary">Zeroscope</span>,
            'v2 XL',
            <span className="text-red-500 font-bold">❌ No (PyTorch)</span>, '~6GB', 'High-Res Upscaling'
          ],
          [
            <span className="font-bold text-primary">SVD</span>,
            'XT 1.1',
            <span className="text-red-500 font-bold">❌ No (PyTorch)</span>, '~8GB', 'Dedicated Image-to-Video'
          ]
        ]}
      />

      <div className="space-y-4 mt-6">
        <div className="bg-amber-500/10 border border-amber-500/20 p-4 rounded-lg">
          <h4 className="text-amber-600 dark:text-amber-400 font-medium mb-2 flex items-center gap-2">
            <Info size={16} /> Mochi 1 & MLX
          </h4>
          <p className="mb-2">
            <strong>Why isn't Mochi 1 native on Mac?</strong><br />
            While a "Partial" MLX port exists in the community, it requires a complex hybrid setup (MLX DiT + PyTorch VAE) that causes severe memory swapping and instability.
            For reliability, we strictly use the <strong>unified PyTorch (MPS)</strong> pipeline for Mochi 1, which ensures it runs correctly, albeit slower than native MLX models.
          </p>
        </div>

        <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-lg">
          <h4 className="text-blue-600 dark:text-blue-400 font-medium mb-2 flex items-center gap-2">
            <Cpu size={16} /> Mac Performance Tip
          </h4>
          <p className="mb-2">
            For the best experience on Apple Silicon, use <strong>Wan 2.2</strong> or <strong>LTX-Video</strong>.
            These models run natively on the Neural Engine via <code>mlx-community</code> (4-bit), offering 2-3x faster speeds than PyTorch-based models like Zeroscope or Mochi.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
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
          <div className="text-primary font-bold text-sm">Bark & SpeechT5</div>
          <div className="text-secondary text-xs">Expressive TTS</div>
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
            '~12GB', '16GB+', <span className="text-yellow-600 dark:text-yellow-400">🕐 Medium</span>, 'TTS', 'Audio/FX & Emotion'
          ],
          [
            <span className="font-bold text-primary">SpeechT5</span>,
            '~2GB', '4GB+', <span className="text-emerald-600 dark:text-emerald-400">⚡ Fast</span>, 'TTS', 'Efficient / Low VRAM'
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
            <li><strong>Voice Control</strong>: Bark supports expressive tokens like [laugh], [sighs] to steer the style.</li>
            <li><strong>Auto-Chunking</strong>: Automatically handles long text inputs for TTS (Bark, SpeechT5) by splitting into stable chunks, ensuring consistent voice quality for long scripts.</li>
            <li><strong>Visual-to-Audio</strong>: Uses an internal "Vision" description of your image/video to prompt the audio model automatically.</li>
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

function HelpChat({ onNavigate }: HelpSectionProps) {
  return (
    <div className="space-y-8 max-w-4xl text-secondary text-sm">
      {/* 1. Intro */}
      <div className="bg-primary/5 p-6 rounded-xl border border-border">
        <h4 className="text-xl font-bold text-primary mb-2">Chat Interface</h4>
        <p className="text-base leading-relaxed">
          The <strong>Chat</strong> is your central command center. Use it for quick queries,
          iterative debugging, translation, and orchestrating other tools.
        </p>
      </div>

      {/* 2. Model Highlights */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-primary-500/10 border border-primary-500/20 p-4 rounded-xl">
          <div className="text-xs font-bold text-primary-600 dark:text-primary-400 mb-1 flex items-center gap-1"><Sparkles size={12} /> General Use</div>
          <div className="text-lg font-bold text-primary">Llama 3.1 8B</div>
          <div className="text-xs text-secondary opacity-80">Balanced & Stable</div>
        </div>
        <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-xl">
          <div className="text-xs font-bold text-blue-600 dark:text-blue-400 mb-1 flex items-center gap-1"><Cpu size={12} /> Reasoning</div>
          <div className="text-lg font-bold text-primary">DeepSeek R1</div>
          <div className="text-xs text-secondary opacity-80">Logic & Math</div>
        </div>
        <div className="bg-orange-500/10 border border-orange-500/20 p-4 rounded-xl opacity-60">
          <div className="text-xs font-bold text-orange-600 dark:text-orange-400 mb-1 flex items-center gap-1"><Terminal size={12} /> Coding</div>
          <div className="text-lg font-bold text-primary">Qwen 3 (Reasoning)</div>
          <div className="text-xs text-secondary opacity-80">Better in Code Tab</div>
        </div>
        <div className="bg-purple-500/10 border border-purple-500/20 p-4 rounded-xl opacity-60">
          <div className="text-xs font-bold text-purple-600 dark:text-purple-400 mb-1 flex items-center gap-1"><Wand2 size={12} /> Creative</div>
          <div className="text-lg font-bold text-primary">Mistral Nemo</div>
          <div className="text-xs text-secondary opacity-80">Better in Article Tab</div>
        </div>
      </div>

      <SectionTitle icon={<Cpu size={20} />}>Platform Defaults</SectionTitle>
      <Table
        headers={['Platform', 'Default Precision', 'Default Framework', 'Notes']}
        rows={[
          [<span className="font-bold text-primary">CUDA (NVIDIA)</span>, <CodeBadge>int4</CodeBadge>, 'PyTorch (BnB)', 'Auto-quantized'],
          [<span className="font-bold text-primary">MPS (Mac)</span>, <CodeBadge>float16</CodeBadge>, 'PyTorch', 'Stable, no quantization'],
          [<span className="font-bold text-primary">MLX (Mac)</span>, <CodeBadge>int4</CodeBadge>, 'MLX (Native)', <span className="text-emerald-500 font-bold">Fastest Inference</span>]
        ]}
      />

      {/* 3. Model Table */}
      <SectionTitle icon={<Cpu size={20} />}>Model Comparison</SectionTitle>
      <Table
        headers={['Model', 'VRAM', 'RAM', 'Speed', 'Platform', 'Best For']}
        rows={[
          [
            <span className="font-medium">Llama 3.1 8B</span>,
            '~16GB', '24GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'General Assistant'
          ],
          [
            <span className="font-medium">Mistral Nemo 12B</span>,
            '~24GB', '32GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Creative Writing'
          ],
          [
            <span className="font-medium">Qwen 2.5 32B</span>,
            '~24GB ⚠️', '120GB', <span className="text-orange-500 font-bold">🐢 Slow</span>, 'CUDA Only', 'SOTA Coding'
          ],
          [
            <span className="font-medium">Qwen 3 8B (Reasoning)</span>,
            '~16GB', '24GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Instruction Following'
          ],
          [
            <span className="font-medium">Qwen 3 14B (Reasoning)</span>,
            '~28GB', '48GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Coding & Logic'
          ],
          [
            <span className="font-medium text-blue-500">DeepSeek R1 Qwen 7B</span>,
            '~7GB', '16GB+', <span className="text-blue-500 font-bold">🧠 Reasoning</span>, 'Mac/CUDA', 'Logic & Math (Fast)'
          ],
          [
            <span className="font-medium text-blue-500">DeepSeek R1 Qwen 14B</span>,
            '~14GB', '32GB+', <span className="text-blue-500 font-bold">🧠 Reasoning</span>, 'Mac/CUDA', 'Hard Problems'
          ],
          [
            <span className="font-medium text-blue-500">DeepSeek R1 Llama 8B</span>,
            '~8GB', '16GB+', <span className="text-blue-500 font-bold">🧠 Reasoning</span>, 'Mac/CUDA', 'General Reasoning'
          ],
          [
            <span className="font-medium text-blue-500">DeepSeek R1 Llama 70B</span>,
            '~40GB', '128GB+', <span className="text-orange-500 font-bold">🐢 Slow</span>, 'CUDA Only', 'SOTA Intelligence'
          ],
          [
            <span className="font-medium text-purple-500">Qwen 3 Opus 4.5 8B</span>,
            '~8GB', '16GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Opus Reasoning'
          ],
          [
            <span className="font-medium text-purple-500">Qwen 3 Opus 4.5 14B</span>,
            '~14GB', '32GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Opus Reasoning'
          ],
          [
            <span className="font-medium text-indigo-500">Qwen 3 GPT-5.2 8B</span>,
            '~8GB', '16GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'GPT-5.2 Reasoning'
          ],
          [
            <span className="font-medium text-indigo-500">Qwen 3 GPT-5.2 14B</span>,
            '~14GB', '32GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'GPT-5.2 Reasoning'
          ]
        ]}
      />

      {/* 4. Core Capabilities */}
      <SectionTitle icon={<Terminal size={20} />}>Core Capabilities</SectionTitle>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Translation - High Priority in Chat */}
        <InfoCard title="Smart Translation" icon={<Globe size={16} className="text-blue-500" />}>
          <p className="mb-2">Real-time, zero-overhead translation integrated directly into chat.</p>
          <button
            onClick={() => onNavigate('translate')}
            className="text-xs text-brand-600 dark:text-brand-400 hover:text-brand-500 font-medium inline-flex items-center gap-1 transition-colors"
          >
            View Translation Models <ArrowRight size={12} />
          </button>
        </InfoCard>

        {/* Deep Research - Shared */}
        <InfoCard title="Deep Research" icon={<Search size={16} className="text-green-500" />}>
          <p>
            Autonomous agent that browses the web to answer complex questions.
          </p>
        </InfoCard>

        {/* Links to Other Tools */}
        <InfoCard title="Article Generator" icon={<FileText size={16} className="text-purple-500" />}>
          <p className="mb-2">Need to write a blog post or documentation?</p>
          <button
            onClick={() => onNavigate('article')}
            className="text-xs text-brand-600 dark:text-brand-400 hover:text-brand-500 font-medium inline-flex items-center gap-1 transition-colors"
          >
            Switch to Article Guide <ArrowRight size={12} />
          </button>
        </InfoCard>

        <InfoCard title="Code Generator" icon={<Code size={16} className="text-orange-500" />}>
          <p className="mb-2">Starting a new software project?</p>
          <button
            onClick={() => onNavigate('code')}
            className="text-xs text-brand-600 dark:text-brand-400 hover:text-brand-500 font-medium inline-flex items-center gap-1 transition-colors"
          >
            Switch to Code Guide <ArrowRight size={12} />
          </button>
        </InfoCard>
      </div>

      {/* 5. Commands */}
      <SectionTitle icon={<LayoutList size={20} />}>Command Reference</SectionTitle>
      <Table
        headers={['Command', 'Description']}
        rows={[
          [<CodeBadge>/search</CodeBadge>, 'Perform a live web search.'],
          [<CodeBadge>/read</CodeBadge>, 'Load a local file.'],
          [<CodeBadge>/clear</CodeBadge>, 'Reset history.'],
        ]}
      />
    </div>
  );
}

function HelpArticle({ onNavigate }: HelpSectionProps) {
  return (
    <div className="space-y-8 max-w-4xl text-secondary text-sm">
      <div className="bg-primary/5 p-6 rounded-xl border border-border">
        <h4 className="text-xl font-bold text-primary mb-2">Article Generator</h4>
        <p className="text-base leading-relaxed">
          The <strong>Article</strong> tool is designed for long-form content creation. It generates structured,
          SEO-optimized articles with automatic markdown formatting.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-purple-500/10 border border-purple-500/20 p-4 rounded-xl">
          <div className="text-xs font-bold text-purple-600 dark:text-purple-400 mb-1 flex items-center gap-1"><Wand2 size={12} /> Best For Writing</div>
          <div className="text-lg font-bold text-primary">Mistral Nemo</div>
          <div className="text-xs text-secondary opacity-80">Excellent Prose</div>
        </div>
        <div className="bg-primary-500/10 border border-primary-500/20 p-4 rounded-xl">
          <div className="text-xs font-bold text-primary-600 dark:text-primary-400 mb-1 flex items-center gap-1"><Sparkles size={12} /> Balanced</div>
          <div className="text-lg font-bold text-primary">Llama 3.1</div>
          <div className="text-xs text-secondary opacity-80">Good Structure</div>
        </div>
      </div>

      <SectionTitle icon={<FileText size={20} />}>Capabilities</SectionTitle>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Long-Form Generation" icon={<FileText size={16} className="text-purple-500" />}>
          <p className="mb-2">Generates content in sections. Options:</p>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>Quick</strong>: ~500 words</li>
            <li><strong>Standard</strong>: ~1500 words</li>
            <li><strong>Detailed</strong>: ~3000 words</li>
            <li><strong>Exhaustive</strong>: ~10,000 words</li>
          </ul>
        </InfoCard>
        <InfoCard title="SEO Optimization" icon={<TrendingUp size={16} className="text-green-500" />}>
          <p>Automatically structures headings, meta descriptions, and keywords for better readability and reach.</p>
        </InfoCard>
      </div>

      <SectionTitle icon={<Cpu size={20} />}>Model Comparison</SectionTitle>
      <Table
        headers={['Model', 'VRAM', 'RAM', 'Speed', 'Platform', 'Best For']}
        rows={[
          [
            <span className="font-medium">Llama 3.1 8B</span>,
            '~16GB', '24GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'General Assistant'
          ],
          [
            <span className="font-medium">Mistral Nemo 12B</span>,
            '~24GB', '32GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Creative Writing'
          ],
          [
            <span className="font-medium">Qwen 2.5 32B</span>,
            '~24GB ⚠️', '120GB', <span className="text-orange-500 font-bold">🐢 Slow</span>, 'CUDA Only', 'SOTA Coding'
          ],
          [
            <span className="font-medium">Qwen 3 8B (Reasoning - 16GB)</span>,
            '~16GB', '24GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Instruction Following'
          ],
          [
            <span className="font-medium">Qwen 3 14B (Reasoning - 28GB)</span>,
            '~28GB', '48GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Coding & Logic'
          ],
          [
            <span className="font-medium text-blue-500">DeepSeek R1 Qwen 7B</span>,
            '~7GB', '16GB+', <span className="text-blue-500 font-bold">🧠 Reasoning</span>, 'Mac/CUDA', 'Logic & Math (Fast)'
          ],
          [
            <span className="font-medium text-blue-500">DeepSeek R1 Qwen 14B</span>,
            '~14GB', '32GB+', <span className="text-blue-500 font-bold">🧠 Reasoning</span>, 'Mac/CUDA', 'Hard Problems'
          ],
          [
            <span className="font-medium text-blue-500">DeepSeek R1 Llama 8B</span>,
            '~8GB', '16GB+', <span className="text-blue-500 font-bold">🧠 Reasoning</span>, 'Mac/CUDA', 'General Reasoning'
          ],
          [
            <span className="font-medium text-blue-500">DeepSeek R1 Llama 70B</span>,
            '~40GB', '128GB+', <span className="text-orange-500 font-bold">🐢 Slow</span>, 'CUDA Only', 'SOTA Intelligence'
          ],
          [
            <span className="font-medium text-purple-500">Qwen 3 Opus 4.5 8B</span>,
            '~8GB', '16GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Opus Reasoning'
          ],
          [
            <span className="font-medium text-purple-500">Qwen 3 Opus 4.5 14B</span>,
            '~14GB', '32GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Opus Reasoning'
          ],
          [
            <span className="font-medium text-indigo-500">Qwen 3 GPT-5.2 8B</span>,
            '~8GB', '16GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'GPT-5.2 Reasoning'
          ],
          [
            <span className="font-medium text-indigo-500">Qwen 3 GPT-5.2 14B</span>,
            '~14GB', '32GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'GPT-5.2 Reasoning'
          ]
        ]}
      />

      <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-xl flex items-center justify-between">
        <div>
          <h4 className="font-bold text-primary text-sm mb-1">Need to Chat?</h4>
          <p className="text-xs text-secondary">Switch to Chat for brainstorming or quick edits.</p>
        </div>
        <button
          onClick={() => onNavigate('chat')}
          className="text-xs text-brand-600 dark:text-brand-400 hover:text-brand-500 font-medium inline-flex items-center gap-1"
        >
          Chat Guide <ArrowRight size={12} />
        </button>
      </div>
    </div>
  );
}

function HelpCode({ onNavigate }: HelpSectionProps) {
  return (
    <div className="space-y-8 max-w-4xl text-secondary text-sm">
      <div className="bg-primary/5 p-6 rounded-xl border border-border">
        <h4 className="text-xl font-bold text-primary mb-2">Code Generator</h4>
        <p className="text-base leading-relaxed">
          The <strong>Code</strong> tool is a specialized environment for scaffolding complex software projects.
          Unlike conversational coding in Chat, this tool focuses on structure, file separation, and iterative development.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-8">
        <div className="bg-orange-500/10 border border-orange-500/20 p-4 rounded-xl">
          <div className="text-xs font-bold text-orange-600 dark:text-orange-400 mb-1 flex items-center gap-1"><Terminal size={12} /> Best For Coding</div>
          <div className="text-lg font-bold text-primary">Qwen 3 (Reasoning)</div>
          <div className="text-xs text-secondary opacity-80">SOTA Code Gen</div>
        </div>
        <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-xl">
          <div className="text-xs font-bold text-blue-600 dark:text-blue-400 mb-1 flex items-center gap-1"><Cpu size={12} /> Hard Logic</div>
          <div className="text-lg font-bold text-primary">DeepSeek R1</div>
          <div className="text-xs text-secondary opacity-80">Reasoning & Math</div>
        </div>
      </div>

      <SectionTitle icon={<Cpu size={20} />}>Model Comparison</SectionTitle>
      <Table
        headers={['Model', 'VRAM', 'RAM', 'Speed', 'Platform', 'Best For']}
        rows={[
          [
            <span className="font-medium">Llama 3.1 8B</span>,
            '~16GB', '24GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'General Assistant'
          ],
          [
            <span className="font-medium">Mistral Nemo 12B</span>,
            '~24GB', '32GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Creative Writing'
          ],
          [
            <span className="font-medium">Qwen 2.5 32B</span>,
            '~24GB ⚠️', '120GB', <span className="text-orange-500 font-bold">🐢 Slow</span>, 'CUDA Only', 'SOTA Coding'
          ],
          [
            <span className="font-medium">Qwen 3 8B (Reasoning - 16GB)</span>,
            '~16GB', '24GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Instruction Following'
          ],
          [
            <span className="font-medium">Qwen 3 14B (Reasoning - 28GB)</span>,
            '~28GB', '48GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Coding & Logic'
          ],
          [
            <span className="font-medium text-blue-500">DeepSeek R1 Qwen 7B</span>,
            '~7GB', '16GB+', <span className="text-blue-500 font-bold">🧠 Reasoning</span>, 'Mac/CUDA', 'Logic & Math (Fast)'
          ],
          [
            <span className="font-medium text-blue-500">DeepSeek R1 Qwen 14B</span>,
            '~14GB', '32GB+', <span className="text-blue-500 font-bold">🧠 Reasoning</span>, 'Mac/CUDA', 'Hard Problems'
          ],
          [
            <span className="font-medium text-blue-500">DeepSeek R1 Llama 8B</span>,
            '~8GB', '16GB+', <span className="text-blue-500 font-bold">🧠 Reasoning</span>, 'Mac/CUDA', 'General Reasoning'
          ],
          [
            <span className="font-medium text-blue-500">DeepSeek R1 Llama 70B</span>,
            '~40GB', '128GB+', <span className="text-orange-500 font-bold">🐢 Slow</span>, 'CUDA Only', 'SOTA Intelligence'
          ],
          [
            <span className="font-medium text-purple-500">Qwen 3 Opus 4.5 8B</span>,
            '~8GB', '16GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Opus Reasoning'
          ],
          [
            <span className="font-medium text-purple-500">Qwen 3 Opus 4.5 14B</span>,
            '~14GB', '32GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'Opus Reasoning'
          ],
          [
            <span className="font-medium text-indigo-500">Qwen 3 GPT-5.2 8B</span>,
            '~8GB', '16GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'GPT-5.2 Reasoning'
          ],
          [
            <span className="font-medium text-indigo-500">Qwen 3 GPT-5.2 14B</span>,
            '~14GB', '32GB+', <span className="text-emerald-500 font-bold">⚡ Fast</span>, 'Mac/CUDA', 'GPT-5.2 Reasoning'
          ]
        ]}
      />

      <SectionTitle icon={<Terminal size={20} />}>Project Scaffolding</SectionTitle>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Multi-File Generation" icon={<FileText size={16} className="text-blue-500" />}>
          <p>Automatically creates folder structures, configuration files, and source code in one pass.</p>
        </InfoCard>
        <InfoCard title="Framework Ready" icon={<Code size={16} className="text-green-500" />}>
          <p>Presets for React, Python, Node.js, and more. Or define custom stacks.</p>
        </InfoCard>
      </div>

      <div className="bg-orange-500/10 border border-orange-500/20 p-4 rounded-xl flex items-start gap-4">
        <div className="p-2 bg-orange-500/20 rounded shrink-0 text-orange-600 dark:text-orange-400">
          <TrendingUp size={20} />
        </div>
        <div>
          <h4 className="font-bold text-primary text-sm mb-1">When to use Code Tool vs Chat?</h4>
          <p className="text-xs text-secondary leading-normal mb-2">
            Use <strong>Chat</strong> for quick snippets, debugging, or explaining concepts. <br />
            Use <strong>Code Tool</strong> when starting a new project or adding a substantial feature that spans multiple files.
          </p>
          <button
            onClick={() => onNavigate('chat')}
            className="text-xs text-brand-600 dark:text-brand-400 hover:text-brand-500 font-medium inline-flex items-center gap-1 transition-colors"
          >
            Back to Chat Guide <ArrowRight size={12} />
          </button>
        </div>
      </div>
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
            <li><strong>Z-Image Turbo</strong>: Fast, high-quality edits (MLX/Mac Optimized).</li>
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

function HelpAnalysis() {
  return (
    <div className="space-y-6 max-w-4xl text-secondary text-sm">
      <p className="text-base">
        Use <strong>Computer Vision</strong> and <strong>Speech Recognition</strong> to analyze media content.
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

        <InfoCard title="Speech Recognition" icon={<MessageSquare size={16} className="text-orange-600 dark:text-orange-400" />}>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>Auto Subtitles</strong>: Generate <code>.srt</code> files for videos.</li>
            <li><strong>Transcription</strong>: Convert speech to text (Markdown/JSON) with timestamps.</li>
          </ul>
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

function HelpPrecision() {
  return (
    <div className="space-y-6 max-w-4xl text-secondary text-sm">
      <p className="text-base font-medium text-primary">
        Control model precision and ML framework for optimal performance on your hardware.
      </p>

      {/* Quick Pick Recommendations */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        <div className="bg-emerald-500/10 border border-emerald-500/20 p-3 rounded-lg">
          <div className="text-emerald-600 dark:text-emerald-400 font-medium text-xs mb-1 text-center">🚀 Speed King</div>
          <div className="text-primary font-bold text-sm text-center">int4 (4-bit)</div>
          <p className="text-secondary text-xs mt-1">Fastest, lowest memory. ~95% quality. Best for quick iterations and limited RAM.</p>
        </div>
        <div className="bg-purple-500/10 border border-purple-500/20 p-3 rounded-lg">
          <div className="text-purple-600 dark:text-purple-400 font-medium text-xs mb-1 text-center">⚖️ Balanced</div>
          <div className="text-primary font-bold text-sm text-center">bfloat16</div>
          <p className="text-secondary text-xs mt-1">Recommended for LLMs. Full quality with 2x memory savings. Default on MPS.</p>
        </div>
        <div className="bg-blue-500/10 border border-blue-500/20 p-3 rounded-lg">
          <div className="text-blue-600 dark:text-blue-400 font-medium text-xs mb-1 text-center">🎯 Reference</div>
          <div className="text-primary font-bold text-sm text-center">float32</div>
          <p className="text-secondary text-xs mt-1">Maximum quality. 4x memory. Use for benchmarking or when quality is critical.</p>
        </div>
      </div>

      <SectionTitle icon={<Cpu size={20} />}>Precision Types</SectionTitle>
      <Table
        headers={['Precision', 'Description', 'Quality', 'Memory']}
        rows={[
          [<span className="font-bold text-primary">float32</span>, 'Full precision (reference)', '100%', '4x'],
          [<span className="font-bold text-primary">bfloat16</span>, 'Brain float (recommended)', '~100%', '2x'],
          [<span className="font-bold text-primary">float16</span>, 'Half precision (standard)', '~99%', '2x'],
          [<span className="font-bold text-primary">int8</span>, '8-bit quantization', '~98%', '1x'],
          [<span className="font-bold text-primary">int6</span>, '6-bit quantization', '~97%', '0.75x'],
          [<span className="font-bold text-primary">int4</span>, '4-bit quantization (fastest)', '~95%', '0.5x'],
        ]}
      />

      <SectionTitle icon={<Monitor size={20} />}>Platform Support</SectionTitle>
      <Table
        headers={['Precision', 'CUDA (NVIDIA)', 'MPS (PyTorch Mac)', 'MLX (Native Mac)']}
        rows={[
          ['float32', <span className="text-emerald-500">✅</span>, <span className="text-emerald-500">✅</span>, <span className="text-emerald-500">✅</span>],
          ['bfloat16', <span className="text-emerald-500">✅ Ampere+</span>, <span className="text-emerald-500">✅</span>, <span className="text-emerald-500">✅</span>],
          ['float16', <span className="text-emerald-500">✅</span>, <span className="text-emerald-500">✅</span>, <span className="text-emerald-500">✅</span>],
          ['int8', <span className="text-emerald-500">✅ bitsandbytes</span>, <span className="text-red-500">❌ Use MLX</span>, <span className="text-emerald-500">✅</span>],
          ['int6', <span className="text-red-500">❌ Not Supported</span>, <span className="text-red-500">❌ Use MLX</span>, <span className="text-emerald-500">✅</span>],
          ['int4', <span className="text-emerald-500">✅ bitsandbytes</span>, <span className="text-red-500">❌ Use MLX</span>, <span className="text-emerald-500">✅</span>],
        ]}
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="CLI Options" icon={<Terminal size={16} className="text-blue-600 dark:text-blue-400" />}>
          <ul className="list-disc pl-4 space-y-1">
            <li><code className="text-xs bg-tertiary px-1 rounded">--precision-force</code> / <code className="text-xs bg-tertiary px-1 rounded">-pf</code>: Force precision</li>
            <li><code className="text-xs bg-tertiary px-1 rounded">--ml-framework</code> / <code className="text-xs bg-tertiary px-1 rounded">-mf</code>: Force framework (Mac)</li>
          </ul>
          <p className="mt-2 text-xs text-tertiary">Example: <code className="bg-tertiary px-1 rounded">-mf mlx -pf int4</code></p>
        </InfoCard>

        <InfoCard title="Inference Server" icon={<Globe size={16} className="text-emerald-600 dark:text-emerald-400" />}>
          <p className="mb-2">Use <code className="text-xs bg-tertiary px-1 rounded">model:precision</code> syntax:</p>
          <div className="bg-tertiary p-2 rounded text-xs font-mono">
            llama-3.1-8b:int4<br />
            qwen-coder-14b:bfloat16
          </div>
        </InfoCard>
      </div>

      <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-xl flex items-start gap-4">
        <div className="p-2 bg-blue-500/20 rounded shrink-0 text-blue-600 dark:text-blue-400">
          <Info size={20} />
        </div>
        <div>
          <h4 className="font-bold text-primary text-sm mb-1">Mac Users: MLX Recommended</h4>
          <p className="text-xs text-secondary leading-normal">
            For <strong>Text, Image, and Video</strong> generation on Apple Silicon, use <strong>MLX with int4</strong> for fastest speeds and lowest memory.
            MLX is optimized for Apple's Neural Engine and provides significantly better performance than PyTorch/MPS.
            <br /><br />
            <em>Note: If MLX fails to load for any reason, the system will automatically fallback to PyTorch (MPS) and upgrade precision to float16 to ensure generation succeeds.</em>
          </p>
        </div>
      </div>
    </div>
  );
}

function HelpInference() {
  return (
    <div className="space-y-6 max-w-4xl text-secondary text-sm">
      <SectionTitle icon={<Globe size={20} />}>OpenAI-Compatible Server</SectionTitle>
      <p className="text-base">
        AI-Media includes a built-in inference server that exposes an OpenAI-compatible API (`/v1`).
        This allows you to use your local AI-Media models with third-party tools like <strong>Continue</strong>, <strong>LM Studio</strong>, <strong>Cursor</strong>, or any application that supports the OpenAI API specification.
      </p>

      <InfoCard title="How to Start" icon={<Terminal size={16} className="text-blue-600 dark:text-blue-400" />}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <div className="font-bold text-primary mb-2 flex items-center gap-2">
              <Terminal size={14} /> CLI Mode (Recommended)
            </div>
            <p className="mb-2 text-xs">Run directly from your terminal:</p>
            <div className="bg-tertiary p-2 rounded font-mono text-xs mb-3 text-primary">
              python ai-media.py --inference-server
            </div>

            <p className="text-xs mb-2"><strong>Custom Port:</strong></p>
            <div className="bg-tertiary p-2 rounded font-mono text-xs mb-3 text-primary">
              python ai-media.py --inference-server --port 8090
            </div>

            <div className="font-bold text-primary mb-1 text-xs">Verbose Mode (Debug)</div>
            <p className="text-xs mb-2">
              To see detailed reasoning logs and token streaming in your terminal:
            </p>
            <div className="bg-tertiary p-2 rounded font-mono text-xs text-primary">
              python ai-media.py --inference-server-verbose
            </div>
          </div>

          <div>
            <div className="font-bold text-primary mb-2 flex items-center gap-2">
              <LayoutList size={14} /> Interactive Mode
            </div>
            <p className="mb-3">
              Select <strong>"Web Server Mode"</strong> &rarr; <strong>"Start Inference Server"</strong> from the interactive menu.
            </p>

            <div className="bg-blue-500/10 border border-blue-500/20 p-3 rounded text-xs text-blue-600 dark:text-blue-400">
              <strong>Tip:</strong> The server will stay running until you press <code>Ctrl+C</code> or send a "stop server" message in chat.
            </div>
          </div>
        </div>
      </InfoCard>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Connection Details" icon={<Globe size={16} className="text-emerald-600 dark:text-emerald-400" />}>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>Base URL</strong>: <code className="bg-tertiary px-1 rounded">http://localhost:8000/v1</code></li>
            <li><strong>API Key</strong>: Any string (e.g. <code className="bg-tertiary px-1 rounded">local</code>)</li>
            <li><strong>Chat Endpoint</strong>: <code className="bg-tertiary px-1 rounded">/v1/chat/completions</code></li>
            <li><strong>Models Endpoint</strong>: <code className="bg-tertiary px-1 rounded">/v1/models</code></li>
            <li><strong>Responses Endpoint</strong>: <code className="bg-tertiary px-1 rounded">/v1/responses</code> (Agent Mode)</li>
          </ul>
        </InfoCard>

        <InfoCard title="Memory Management" icon={<Cpu size={16} className="text-purple-600 dark:text-purple-400" />}>
          <p className="mb-2">The server enforces a <strong>Single Active Model</strong> policy to prevent OOM errors.</p>
          <ul className="list-disc pl-4 space-y-1">
            <li>Switching from Text to Image automatically unloads the Text model.</li>
            <li>Use <code className="bg-tertiary px-1 rounded">unload model</code> in chat to free RAM manually.</li>
            <li>Use <code className="bg-tertiary px-1 rounded">flush memory</code> to force garbage collection.</li>
          </ul>
        </InfoCard>
      </div>

      <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Stopping the Server" icon={<X size={16} className="text-red-500" />}>
          <ul className="list-disc pl-4 space-y-1 text-sm text-secondary">
            <li><strong>Terminal</strong>: Press <code className="bg-tertiary px-1 rounded">Ctrl+C</code> in the terminal.</li>
            <li><strong>Chat Command</strong>: Send <code className="bg-tertiary px-1 rounded">stop inference server</code> in chat.</li>
            <li><strong>Kill Command</strong>: Run <code className="bg-tertiary px-1 rounded">kill &lt;pid&gt;</code>.</li>
          </ul>
        </InfoCard>

        <InfoCard title="Random Prompts" icon={<Sparkles size={16} className="text-yellow-500" />}>
          <p className="text-sm text-secondary mb-2">
            Send specific keywords in chat to get creative prompts:
          </p>
          <ul className="list-disc pl-4 space-y-1 text-sm text-secondary">
            <li>Send <code className="bg-tertiary px-1 rounded">rndPr</code> or <code className="bg-tertiary px-1 rounded">random prompt</code>.</li>
            <li><strong>Image Models</strong>: Returns creative scene descriptions.</li>
            <li><strong>Text Models</strong>: Returns coding tasks or article ideas.</li>
          </ul>
        </InfoCard>
      </div>

      <div className="mt-4">
        <InfoCard title="Using with Continue (VS Code / JetBrains)" icon={<Code size={16} className="text-blue-500" />}>
          <p className="text-sm text-secondary mb-3">
            <a href="https://continue.dev" target="_blank" rel="noreferrer" className="text-primary hover:underline">Continue</a> is an open-source AI code assistant. You can configure it to use AI-Media as its backend.
          </p>
          <div className="bg-tertiary p-3 rounded text-xs border border-border/50">
            <strong className="text-primary block mb-1">🚀 Quick Setup:</strong>
            Use the included sample config file found in this repository:
            <code className="block my-2 p-2 bg-black/10 dark:bg-black/30 rounded font-mono text-primary select-all">extras/continue-dev-example-config/config.sample.yaml</code>
            <p className="text-secondary">
              Simply copy its contents to your <code className="bg-black/10 dark:bg-black/30 px-1 rounded">~/.continue/config.yaml</code> to get all AI-Media models pre-configured instantly.
            </p>
          </div>
        </InfoCard>
      </div>

      <div className="mt-4">
        <InfoCard title="Adding Context (Files, Folders, Workspace)" icon={<FolderOpen size={16} className="text-orange-500" />}>
          <p className="text-sm text-secondary mb-2">
            The server supports full context awareness. Clients like Continue resolve references into text <em>before</em> sending to the server.
          </p>
          <ul className="list-disc pl-4 space-y-1 text-xs text-secondary mb-3">
            <li><strong>@File / @Open Files</strong>: Attaches specific file contents.</li>
            <li><strong>@Folder</strong>: Attaches an entire directory.</li>
            <li><strong>@Codebase</strong>: Uses embeddings to find relevant code snippets.</li>
            <li><strong>@Docs</strong>: Attaches external documentation.</li>
          </ul>
          <div className="bg-yellow-500/10 border border-yellow-500/20 p-2 rounded text-xs text-yellow-600 dark:text-yellow-400">
            <strong>Important:</strong> You must <strong>select</strong> the context item from the dropdown menu (so it becomes a "chip") for it to work. Typing <code>@file.py</code> without selecting it sends raw text, not the file content.
          </div>
        </InfoCard>
      </div>

      <div className="mt-4">
        <InfoCard title="Supported Models" icon={<Book size={16} className="text-primary" />}>
          <p className="text-sm text-secondary mb-2">
            The server automatically exposes all available models. See their respective sections for details:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
            <div className="bg-tertiary p-2 rounded border border-border">
              <strong className="block text-primary mb-1">Text Models (Chat, Code, Articles)</strong>
              <p className="text-secondary">See <strong>"Chat Interface"</strong> or <strong>"Code Generator"</strong> sections.</p>
            </div>
            <div className="bg-tertiary p-2 rounded border border-border">
              <strong className="block text-primary mb-1">Image Models</strong>
              <p className="text-secondary">See <strong>"Image Generation"</strong> section.</p>
            </div>
          </div>
        </InfoCard>
      </div>

      <div className="bg-secondary/50 p-4 rounded-lg border border-border mt-4">
        <h4 className="text-primary-600 dark:text-primary-400 font-medium mb-2">Compatible Clients</h4>
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="bg-secondary px-2 py-1 rounded border border-border">VS Code (Continue)</span>
          <span className="bg-secondary px-2 py-1 rounded border border-border">JetBrains (Continue)</span>
          <span className="bg-secondary px-2 py-1 rounded border border-border">LM Studio</span>
          <span className="bg-secondary px-2 py-1 rounded border border-border">Cursor</span>
          <span className="bg-secondary px-2 py-1 rounded border border-border">Open WebUI</span>
        </div>
      </div>
    </div>
  );
}

function HelpCleanup() {
  return (
    <div className="space-y-6 max-w-4xl text-secondary text-sm">
      <p className="text-base font-medium text-primary">
        Manage disk space by clearing generated outputs and cached AI models.
      </p>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Output Folders" icon={<FolderOpen size={16} className="text-blue-600 dark:text-blue-400" />}>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>Test Outputs</strong>: Temporary files generated during testing. Safe to clear anytime.</li>
            <li><strong>Media Output</strong>: Final generated images, videos, and audio. Back these up before clearing!</li>
          </ul>
        </InfoCard>

        <InfoCard title="Model Cache" icon={<HardDrive size={16} className="text-orange-600 dark:text-orange-400" />}>
          <p className="mb-2">AI models are downloaded from Hugging Face and stored in <code>HF_HOME</code>.</p>
          <ul className="list-disc pl-4 space-y-1">
            <li>Cached models can take up significant space (10GB-100GB+).</li>
            <li>Deleting a model removes it from disk to free up space.</li>
            <li>Models will be <strong>re-downloaded automatically</strong> if you use them again.</li>
          </ul>
        </InfoCard>
      </div>

      <div className="bg-primary/50 p-4 rounded-lg border border-border">
        <h4 className="text-primary font-bold mb-3 flex items-center gap-2">
          <Trash2 size={16} className="text-red-500" />
          Cleanup Options
        </h4>
        <Table
          headers={['Action', 'Description', 'Risk']}
          rows={[
            [
              <span className="font-bold text-primary">Clear Test Outputs</span>,
              'Removes all files in testing/data/outputs',
              <span className="text-emerald-500">Safe</span>
            ],
            [
              <span className="font-bold text-primary">Clear Media Output</span>,
              'Removes all generated media files',
              <span className="text-yellow-500">Data Loss</span>
            ],
            [
              <span className="font-bold text-primary">Clear All Outputs</span>,
              'Clears BOTH test and media folders',
              <span className="text-yellow-500">Data Loss</span>
            ],
            [
              <span className="font-bold text-primary">Delete Model</span>,
              'Removes specific model from cache',
              <span className="text-blue-500">Re-download needed</span>
            ]
          ]}
        />
      </div>

      <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-xl flex items-start gap-4">
        <div className="p-2 bg-blue-500/20 rounded shrink-0 text-blue-600 dark:text-blue-400">
          <Info size={20} />
        </div>
        <div>
          <h4 className="font-bold text-primary text-sm mb-1">Tip: Auto-Redownload</h4>
          <p className="text-xs text-secondary leading-normal">
            Don't worry about deleting models to save space. If you delete a model (e.g., <code>flux.1</code>) and then try to generate an image with it later,
            AI-Media will simply download it again automatically.
          </p>
        </div>
      </div>
    </div>
  );
}





function LanguageTable({ languages, type }: {
  languages: { label: string; value: string; audioOut?: boolean; code_alias?: string }[],
  type: 'nllb' | 'seamless' | 'llm' | 'alma'
}) {
  const [searchTerm, setSearchTerm] = useState('');

  const filtered = languages.filter(l =>
    l.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
    l.value.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-4">
      <div className="relative">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-tertiary" />
        <input
          type="text"
          placeholder={`Search ${languages.length} languages...`}
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-9 pr-3 py-1.5 rounded-lg bg-secondary border border-border text-xs focus:ring-2 focus:ring-brand-500 outline-none w-full md:w-64"
        />
      </div>

      <div className="border border-border rounded-lg overflow-hidden bg-secondary/30">
        <div className="overflow-y-auto max-h-96 custom-scrollbar">
          <table className="w-full text-left text-xs">
            <thead className="bg-tertiary dark:bg-primary text-primary font-semibold sticky top-0 z-10 shadow-sm">
              <tr>
                <th className="p-3 border-b border-border">Language</th>
                <th className="p-3 border-b border-border">Code</th>
                {type === 'seamless' && (
                  <>
                    <th className="p-3 border-b border-border w-24 text-center">Audio In</th>
                    <th className="p-3 border-b border-border w-24 text-center">Audio Out</th>
                  </>
                )}
                <th className="p-3 border-b border-border w-full"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {filtered.map((lang) => (
                <tr key={lang.value} className="hover:bg-primary/50 transition-colors">
                  <td className="p-3 font-medium text-secondary">
                    {lang.label.split('(')[0].trim()}
                    <span className="text-[10px] text-tertiary ml-1 opacity-70">
                      {lang.label.includes('(') ? `(${lang.label.split('(')[1]}` : ''}
                    </span>
                  </td>
                  <td className="p-3 font-mono text-tertiary select-all">
                    {lang.code_alias || lang.value}
                  </td>
                  {type === 'seamless' && (
                    <>
                      <td className="p-3 text-center">
                        <div className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                          <Check size={12} />
                        </div>
                      </td>
                      <td className="p-3 text-center">
                        {lang.audioOut ? (
                          <div className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400">
                            <Check size={12} />
                          </div>
                        ) : (
                          <span className="text-tertiary/20">-</span>
                        )}
                      </td>
                    </>
                  )}
                  <td></td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={4} className="p-8 text-center text-tertiary">
                    No languages found matching "{searchTerm}"
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function HelpTranslate(_props: HelpSectionProps) {
  // Tabs: 'nllb', 'seamless', 'llm', 'alma'
  const [activeTab, setActiveTab] = useState<'nllb' | 'seamless' | 'llm' | 'alma'>('nllb');

  return (
    <div className="space-y-6 max-w-4xl text-secondary text-sm">
      <p className="text-base">
        Translate text, documents, and audio using advanced local models.
      </p>

      {/* Quick Pick Recommendations */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-emerald-500/10 border border-emerald-500/20 p-3 rounded-lg">
          <div className="text-emerald-600 dark:text-emerald-400 font-medium text-xs mb-1">⚡ Speed & Coverage</div>
          <div className="text-primary font-bold text-sm">NLLB-200</div>
          <div className="text-secondary text-xs">Fast, 200+ languages, auto-detect</div>
        </div>
        <div className="bg-blue-500/10 border border-blue-500/20 p-3 rounded-lg">
          <div className="text-blue-600 dark:text-blue-400 font-medium text-xs mb-1">🎯 Professional Quality</div>
          <div className="text-primary font-bold text-sm">ALMA (13B)</div>
          <div className="text-secondary text-xs">Best nuance & professional tone</div>
        </div>
        <div className="bg-purple-500/10 border border-purple-500/20 p-3 rounded-lg">
          <div className="text-purple-600 dark:text-purple-400 font-medium text-xs mb-1">💬 Natural Conversation</div>
          <div className="text-primary font-bold text-sm">Qwen / Llama</div>
          <div className="text-secondary text-xs">Context-aware, handles idioms</div>
        </div>
        <div className="bg-amber-500/10 border border-amber-500/20 p-3 rounded-lg">
          <div className="text-amber-600 dark:text-amber-400 font-medium text-xs mb-1">🎤 Speech Translation</div>
          <div className="text-primary font-bold text-sm">Seamless M4T</div>
          <div className="text-secondary text-xs">Audio in/out, dubbing</div>
        </div>
      </div>

      {/* When to Use Each Model - Detailed Guide */}
      <div className="bg-primary/50 p-4 rounded-lg border border-border">
        <h4 className="text-primary font-bold mb-3 flex items-center gap-2">
          <Info size={16} className="text-blue-500" />
          Choosing the Right Model
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div>
            <div className="font-bold text-emerald-600 dark:text-emerald-400 mb-1">NLLB-200 High Quality (3.3B)</div>
            <ul className="list-disc pl-4 space-y-1 text-secondary mb-3">
              <li>✅ Maximum accuracy & nuance</li>
              <li>✅ Supports 200+ languages</li>
              <li>⚠️ ~8GB VRAM Required</li>
              <li>⚠️ Slower (5-10 seconds)</li>
            </ul>

            <div className="font-bold text-teal-600 dark:text-teal-400 mb-1">NLLB-200 Fast (Distilled 600M)</div>
            <ul className="list-disc pl-4 space-y-1 text-secondary">
              <li>✅ Lightning fast (0.5-2 seconds)</li>
              <li>✅ Low Memory (~4GB VRAM)</li>
              <li>✅ Good for quick chats/drafts</li>
              <li>⚠️ Slightly less precise than 3.3B</li>
            </ul>
          </div>
          <div>
            <div className="font-bold text-purple-600 dark:text-purple-400 mb-1">LLM Models (ALMA, Qwen, Llama)</div>
            <ul className="list-disc pl-4 space-y-1 text-secondary">
              <li>✅ Better preservation of tone & nuance</li>
              <li>✅ Understands idioms & cultural context</li>
              <li>✅ More natural-sounding output</li>
              <li>✅ Ideal for professional/creative content</li>
              <li>⚠️ Slower (10-30 seconds)</li>
              <li>⚠️ Requires explicit source language</li>
              <li>⚠️ Higher memory (8-26GB)</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Translation Modes" icon={<Globe size={16} className="text-blue-600 dark:text-blue-400" />}>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>Text & Documents</strong>: High-accuracy translation for text input and documents (.pdf, .docx, .txt).</li>
            <li><strong>Images (OCR)</strong>: Extract text from images and translate it automatically.</li>
            <li><strong>Audio</strong>: Speech-to-Speech and Speech-to-Text translation (Seamless M4T).</li>
          </ul>
        </InfoCard>

        <InfoCard title="Language Support" icon={<Languages size={16} className="text-emerald-600 dark:text-emerald-400" />}>
          <p>
            We classify coverage into three tiers:
          </p>
          <ul className="list-disc pl-4 space-y-1 mt-1">
            <li><strong>NLLB-200</strong>: Massive support for 200+ global languages including low-resource ones.</li>
            <li><strong>Seamless M4T</strong>: 100+ input languages for speech translation, with 35 output languages.</li>
            <li><strong>Major (LLMs)</strong>: Context-aware translation for ~25 major languages using Qwen / Llama.</li>
          </ul>
        </InfoCard>
      </div>

      <div id="models">
        <SectionTitle icon={<Cpu size={20} />}>Model Capabilities</SectionTitle>
        <Table
          headers={['Model', 'Languages', 'Type', 'Best For']}
          rows={[
            [
              <span className="font-bold text-primary">NLLB-200-3.3B</span>,
              '200+ (FLORES)', 'Neural Machine', 'Accuracy & Coverage (Default)'
            ],
            [
              <span className="font-bold text-primary">Seamless M4T v2</span>,
              '100 (In) / 35 (Out)', 'Multimodal', 'Speech Translation & Dubbing'
            ],
            [
              <span className="font-bold text-primary">Qwen / Llama</span>,
              'Major Languages', 'LLM', 'Context, Nuance & Complex Docs'
            ],
            [
              <span className="font-bold text-primary">ALMA</span>,
              'Major Languages', 'LLM', 'Professional Translation'
            ]
          ]}
        />
      </div>

      <div id="languages" className="bg-primary/50 p-4 rounded-lg border border-border mt-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
          <h4 className="text-primary font-bold flex items-center gap-2">
            <Globe size={16} className="text-primary-600 dark:text-primary-400" />
            Supported Languages
          </h4>
        </div>

        {/* Tabs - Styled like TranslateView / TransformView */}
        <div className="flex bg-primary p-1 rounded-lg border border-border mb-4 w-full md:w-[480px]">
          <button
            className={`flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all flex items-center justify-center gap-2 ${activeTab === 'nllb' ? 'bg-indigo-100 dark:bg-secondary text-indigo-900 dark:text-primary shadow-lg' : 'text-slate-600 dark:text-slate-400 hover:text-indigo-700 dark:hover:text-slate-200 hover:bg-indigo-50 dark:hover:bg-tertiary'}`}
            onClick={() => setActiveTab('nllb')}
          >
            <LayoutList size={14} />
            NLLB-200
          </button>
          <button
            className={`flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all flex items-center justify-center gap-2 ${activeTab === 'seamless' ? 'bg-indigo-100 dark:bg-secondary text-indigo-900 dark:text-primary shadow-lg' : 'text-slate-600 dark:text-slate-400 hover:text-indigo-700 dark:hover:text-slate-200 hover:bg-indigo-50 dark:hover:bg-tertiary'}`}
            onClick={() => setActiveTab('seamless')}
          >
            <Mic size={14} />
            Seamless M4T
          </button>
          <button
            className={`flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all flex items-center justify-center gap-2 ${activeTab === 'alma' ? 'bg-indigo-100 dark:bg-secondary text-indigo-900 dark:text-primary shadow-lg' : 'text-slate-600 dark:text-slate-400 hover:text-indigo-700 dark:hover:text-slate-200 hover:bg-indigo-50 dark:hover:bg-tertiary'}`}
            onClick={() => setActiveTab('alma')}
          >
            <Sparkles size={14} />
            ALMA
          </button>
          <button
            className={`flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all flex items-center justify-center gap-2 ${activeTab === 'llm' ? 'bg-indigo-100 dark:bg-secondary text-indigo-900 dark:text-primary shadow-lg' : 'text-slate-600 dark:text-slate-400 hover:text-indigo-700 dark:hover:text-slate-200 hover:bg-indigo-50 dark:hover:bg-tertiary'}`}
            onClick={() => setActiveTab('llm')}
          >
            <MessageSquare size={14} />
            LLMs (Qwen)
          </button>
        </div>

        {activeTab === 'nllb' && (
          <LanguageTable
            languages={ALL_LANGUAGES}
            type="nllb"
          />
        )}

        {activeTab === 'seamless' && (
          <div className="space-y-4">
            <div className="p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg text-xs text-blue-300">
              <strong>Note:</strong> Seamless M4T supports <strong>~100 languages for Audio Input</strong>.
              For Audio Output (Speech-to-Speech), it supports a subset of 35 languages (marked below).
            </div>
            <LanguageTable
              languages={SEAMLESS_LANGUAGES}
              type="seamless"
            />
          </div>
        )}

        {activeTab === 'alma' && (
          <div className="space-y-4">
            <div className="p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg text-xs text-purple-300">
              <strong>ALMA (Advanced Language Model-based Translator)</strong> is explicitly optimized for professional-grade translation between these core languages.
            </div>
            <LanguageTable
              languages={ALMA_LANGUAGES}
              type="alma"
            />
          </div>
        )}

        {activeTab === 'llm' && (
          <LanguageTable
            languages={LLM_LANGUAGES}
            type="llm"
          />
        )}

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
    { id: 'analysis', title: 'Analysis (Vision & Audio)', icon: <ScanEye className="text-indigo-600 dark:text-indigo-400" size={24} />, desc: 'Image Description, Subtitles, Transcription' },
    { id: 'transform', title: 'Transformations', icon: <Wand2 className="text-pink-600 dark:text-pink-400" size={24} />, desc: 'Edit, BG Removal' },
    { id: 'multimedia', title: 'Converters', icon: <FileType className="text-slate-600 dark:text-slate-400" size={24} />, desc: 'Format conversion & OCR' },
    { id: 'translate', title: 'Translation', icon: <Globe className="text-teal-600 dark:text-teal-400" size={24} />, desc: 'NLLB & Seamless models' },
    { id: 'upscale', title: 'Upscaling', icon: <TrendingUp className="text-cyan-600 dark:text-cyan-400" size={24} />, desc: 'AI & Lanczos enlargement' },
    { id: 'cleanup', title: 'Cleanup', icon: <Trash2 className="text-red-500" size={24} />, desc: 'Manage storage & space' },
    { id: 'precision', title: 'Precision & Framework', icon: <Cpu className="text-slate-600 dark:text-slate-400" size={24} />, desc: 'Quantization & MLX/CUDA' },
    { id: 'inference', title: 'Inference Server', icon: <Globe className="text-green-600 dark:text-green-400" size={24} />, desc: 'OpenAI API & Clients' },
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
  { id: 'chat', label: 'Chat Interface', icon: <MessageSquare size={18} />, component: HelpChat },
  { id: 'article', label: 'Article Generator', icon: <FileText size={18} />, component: HelpArticle },
  { id: 'code', label: 'Code Generator', icon: <Code size={18} />, component: HelpCode },
  { id: 'analysis', label: 'Analysis', icon: <ScanEye size={18} />, component: HelpAnalysis },
  { id: 'transform', label: 'Transformations', icon: <Wand2 size={18} />, component: HelpTransform },
  { id: 'multimedia', label: 'Converters', icon: <FileType size={18} />, component: HelpMultimedia },
  { id: 'translate', label: 'Translation', icon: <Globe size={18} />, component: HelpTranslate },
  { id: 'upscale', label: 'Upscaling', icon: <TrendingUp size={18} />, component: HelpUpscale },
  { id: 'cleanup', label: 'Cleanup', icon: <Trash2 size={18} />, component: HelpCleanup },
  { id: 'precision', label: 'Precision & Framework', icon: <Cpu size={18} />, component: HelpPrecision },
  { id: 'inference', label: 'Inference Server', icon: <Globe size={18} />, component: HelpInference },
];

export function HelpModal() {
  const { isHelpOpen, helpSection, toggleHelp } = useAppStore();
  const [activeSection, setActiveSection] = useState(helpSection || 'general');

  // Sync activeSection when store's helpSection changes AND handle hash scrolling
  useEffect(() => {
    if (helpSection && isHelpOpen) {
      const parts = helpSection.split('#');
      const section = parts[0];
      const hash = parts[1];

      setActiveSection(section);

      if (hash) {
        // Wait for render cycle/animation
        setTimeout(() => {
          const element = document.getElementById(hash);
          if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
            // Optional: highlight effect
            element.classList.add('bg-brand-500/10', 'rounded-lg', 'transition-colors', 'duration-1000');
            setTimeout(() => element.classList.remove('bg-brand-500/10'), 3000);
          }
        }, 300);
      }
    }
  }, [helpSection, isHelpOpen]);

  if (!isHelpOpen) return null;

  // Fallback to general if activeSection not found (e.g. invalid URL param)
  const activeItem = SECTIONS.find(s => s.id === activeSection) || SECTIONS[0];
  const ActiveComponent = activeItem.component;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200" onClick={toggleHelp}>
      <div
        className="bg-secondary border border-border rounded-xl shadow-2xl w-full max-w-6xl h-[85vh] flex flex-col md:flex-row overflow-hidden animate-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Sidebar / Top Nav */}
        <div className="w-full md:w-64 bg-primary border-b md:border-b-0 md:border-r border-border flex flex-row md:flex-col shrink-0 overflow-x-auto md:overflow-visible custom-scrollbar">
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

          <div className="flex-1 overflow-y-auto p-4 md:p-8 custom-scrollbar scroll-smooth">
            <ActiveComponent onNavigate={setActiveSection} />
          </div>
        </div>
      </div>
    </div>
  );
}
