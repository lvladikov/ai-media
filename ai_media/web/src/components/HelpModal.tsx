import { X, Book, Lock, ExternalLink, Image, Film, Music, FileText, Wand2, ScanEye, Monitor, Cpu, Terminal, Palette, FileType, MessageSquare } from 'lucide-react';
import { useAppStore } from '../store';
import { useState } from 'react';

// --- Shared UI Components ---

function ExternalLinkBtn({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a 
      href={href} 
      target="_blank" 
      rel="noreferrer" 
      className="text-primary-400 hover:text-primary-300 inline-flex items-center gap-1 hover:underline"
    >
      {children} <ExternalLink size={12} />
    </a>
  );
}

function CodeBadge({ children }: { children: React.ReactNode }) {
  return (
    <code className="bg-slate-800 px-1.5 py-0.5 rounded text-amber-200 text-sm font-mono border border-slate-700">
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
    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2 border-b border-slate-700 pb-2 mt-8 first:mt-0">
      <span className="text-primary-400">{icon}</span>
      {children}
    </h3>
  );
}

function InfoCard({ title, children, icon }: { title: string; children: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <div className="bg-slate-950/50 p-4 rounded-lg border border-slate-800">
      <h4 className="text-md font-medium text-slate-200 mb-2 flex items-center gap-2">
        {icon || <Book size={16} className="text-slate-500"/>}
        {title}
      </h4>
      <div className="text-sm text-slate-400 space-y-2 pl-6">
        {children}
      </div>
    </div>
  );
}

function Table({ headers, rows }: { headers: string[], rows: (string | React.ReactNode)[][] }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-700 my-4">
      <table className="w-full text-left text-xs">
        <thead className="bg-slate-950 text-slate-300">
          <tr>
            {headers.map((h, i) => <th key={i} className="p-3">{h}</th>)}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800 bg-slate-900/50">
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => <td key={j} className="p-3 text-slate-300">{cell}</td>)}
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
    <div className="space-y-6 max-w-3xl text-slate-300">
      <p className="text-lg text-slate-200">
        Some state-of-the-art models (like <CodeBadge>FLUX.1</CodeBadge> and <CodeBadge>SD 3.5</CodeBadge>) require Hugging Face authentication, but correspond to free-to-use research licenses.
      </p>

      <div className="space-y-4">
        <div className="bg-slate-950/50 p-4 rounded-lg border border-slate-800">
           <h3 className="text-white font-semibold flex items-center gap-2 mb-2">
             <span className="flex items-center justify-center w-6 h-6 rounded-full bg-slate-800 text-xs border border-slate-700">1</span>
             Create Account
           </h3>
           <p className="text-sm ml-8">
             Sign up for a free <ExternalLinkBtn href="https://huggingface.co/join">Hugging Face Account</ExternalLinkBtn>.
           </p>
        </div>

        <div className="bg-slate-950/50 p-4 rounded-lg border border-slate-800">
          <h3 className="text-white font-semibold flex items-center gap-2 mb-2">
             <span className="flex items-center justify-center w-6 h-6 rounded-full bg-slate-800 text-xs border border-slate-700">2</span>
             Accept Licenses
           </h3>
          <p className="text-sm ml-8 mb-4">
            Visit each model page below and click <span className="text-white font-medium">"Agree and access repository"</span>:
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
              <div key={model.name} className="flex items-center justify-between bg-slate-800/40 px-3 py-2 rounded border border-slate-700/50 hover:bg-slate-800/60 transition-colors">
                <span className="font-mono text-slate-300">{model.name}</span>
                <ExternalLinkBtn href={model.url}>Accept License</ExternalLinkBtn>
              </div>
            ))}
          </div>
        </div>
        
         <div className="bg-slate-950/50 p-4 rounded-lg border border-slate-800">
          <h3 className="text-white font-semibold flex items-center gap-2 mb-2">
             <span className="flex items-center justify-center w-6 h-6 rounded-full bg-slate-800 text-xs border border-slate-700">3</span>
             Create Token
           </h3>
           <ul className="list-disc ml-8 space-y-1 text-sm">
            <li>Go to <ExternalLinkBtn href="https://huggingface.co/settings/tokens">Settings → Access Tokens</ExternalLinkBtn>.</li>
            <li>Create a new token with <strong>Read</strong> permissions.</li>
          </ul>
        </div>

        <div className="bg-slate-950/50 p-4 rounded-lg border border-slate-800">
          <h3 className="text-white font-semibold flex items-center gap-2 mb-2">
             <span className="flex items-center justify-center w-6 h-6 rounded-full bg-slate-800 text-xs border border-slate-700">4</span>
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
    <div className="space-y-6 max-w-4xl text-slate-300 text-sm">
      <p className="text-base">
        Generate high-quality images using state-of-the-art diffusion models running locally. 
        Supports SDXL, SD 1.5, SD 3.5, and Flux.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Key Features" icon={<Image size={16} className="text-blue-400"/>}>
          <ul className="list-disc pl-4 space-y-1">
            <li><strong>Text-to-Image</strong>: Detailed wallpapers, art, photos.</li>
            <li><strong>Format Control</strong>: Landscape (16:9), Portrait (9:16), Square.</li>
            <li><strong>Proactive Optimization</strong>: Auto-upscaling for high resolutions.</li>
            <li><strong>Negative Prompt</strong>: List items to remove (e.g., "blurry, text"). Avoid "no" or "without".</li>
            <li><strong>Safety</strong>: Optional NSFW checker.</li>
          </ul>
        </InfoCard>
        
        <InfoCard title="Supported Formats" icon={<Monitor size={16} className="text-green-400"/>}>
           <ul className="list-disc pl-4 space-y-1">
            <li><strong>Resolutions</strong>: 720p, 1080p, 4k, 8k, HD, UHD.</li>
            <li><strong>Custom</strong>: Width x Height (e.g. 1024x1024).</li>
            <li><strong>Files</strong>: PNG (lossless), JPG (compressed).</li>
           </ul>
        </InfoCard>
      </div>

      <SectionTitle icon={<Cpu size={20}/>}>Recommended Models</SectionTitle>
      <Table 
        headers={['Model', 'VRAM', 'Best For']}
        rows={[
          [<span className="font-bold text-white flex items-center">SD 3.5 Large Turbo <GatedLock onClick={() => onNavigate('gated-models')}/></span>, '~19GB', 'Default. Fast (4 steps) & High Qual.'],
          [<span className="font-bold text-white">SDXL Turbo</span>, '~8GB', 'Fast, no login. Good all-rounder.'],
          [<span className="font-bold text-white flex items-center">FLUX.1-schnell <GatedLock onClick={() => onNavigate('gated-models')}/></span>, '~12GB+', <span>State-of-the-art realism. <span className="text-yellow-400">⚠️ Very slow on Mac.</span></span>],
          [<span className="font-bold text-white">SD 1.5</span>, '~4GB', 'Low VRAM, artistic styles, faster.'],
          [<span className="font-bold text-white">Qwen-Image</span>, '~20GB', 'Best text rendering. CUDA only.'],
          [<span className="font-bold text-white">Qwen-Image (MPS)</span>, '~40GB', 'Text rendering on Mac (Float32).']
        ]}
      />

      <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-lg mt-4">
        <h4 className="text-blue-400 font-medium mb-2 flex items-center gap-2"><Monitor size={16}/> Smart Multi-Stage Strategy (Proactive Workflow)</h4>
        <p className="mb-2">
          Generating native 4K+ images can crash systems. We use a smart strategy for requests &gt; 6 Megapixels:
        </p>
        <ol className="list-decimal pl-5 space-y-1">
          <li><strong>Threshold Detection</strong>: Detects high-res request (e.g. 8K).</li>
          <li><strong>Step 1</strong>: Generates at a stable ~3K base resolution (fits in VRAM).</li>
          <li><strong>Step 2</strong>: Instantly <strong>AI Upscales</strong> to target using Real-ESRGAN.</li>
        </ol>
      </div>
    </div>
  );
}

function HelpVideo({ onNavigate }: HelpSectionProps) {
  return (
     <div className="space-y-6 max-w-4xl text-slate-300 text-sm">
      <p className="text-base">
        Create engaging short clips using models like Zeroscope, LTX-Video, and Wan 2.2.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Modes" icon={<Film size={16} className="text-purple-400"/>}>
           <ul className="list-disc pl-4 space-y-1">
            <li><strong>Text-to-Video</strong>: From scratch prompts ("A robot dancing").</li>
            <li><strong>Image-to-Video (I2V)</strong>: Animate a starting image (`-ii`).</li>
            <li><strong>Audio-Reactive</strong>: Generates matching audio track automatically.</li>
           </ul>
        </InfoCard>

         <InfoCard title="Mac M-Series Optimization" icon={<Cpu size={16} className="text-yellow-400"/>}>
           <ul className="list-disc pl-4 space-y-1">
            <li><strong>LTX-Video</strong>: ✅ Best native performance (~35s for 2s).</li>
            <li><strong>Zeroscope</strong>: ✅ Fast, efficient, auto-upscales.</li>
            <li><strong>Mochi 1</strong>: ⚠️ Works but slow (Sequential Offload).</li>
            <li><strong>Wan 2.2</strong>: ❌ Impractical (Too slow).</li>
            <li><strong>Hunyuan</strong>: ❌ Fails on &lt;64GB Macs.</li>
           </ul>
        </InfoCard>
      </div>

      <SectionTitle icon={<Cpu size={20}/>}>Video Models</SectionTitle>
      <Table
        headers={['Model', 'Type', 'VRAM', 'Notes']}
        rows={[
          [<span className="font-bold text-white">Zeroscope</span>, 'T2V', '~4GB', 'Default. Fast. Dynamic Upscaling.'],
          [<span className="font-bold text-white">LTX-Video</span>, 'T2V / I2V', '~12GB', 'Best balance for Mac/Consumer GPU.'],
          [<span className="font-bold text-white">SVD</span>, 'I2V Only', '~8GB', 'Stable Video Diffusion. Slow on Mac (CPU).'],
          [<span className="font-bold text-white flex items-center">Wan 2.2 <GatedLock onClick={() => onNavigate('gated-models')}/></span>, 'T2V / I2V', '~24GB', 'SOTA 2025. Very heavy (NVIDIA rec).'],
          [<span className="font-bold text-white">Mochi 1</span>, 'T2V', '~19GB', 'High motion fidelity. Slow on Mac.']
        ]}
      />

       <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800">
        <h4 className="text-primary-400 font-medium mb-2">Zeroscope Dynamic Upscaling Pipeline</h4>
        <p className="mb-2">When generating &gt; 576x320 with Zeroscope:</p>
        <div className="flex items-center gap-2 text-xs">
          <span className="bg-slate-800 px-2 py-1 rounded">1. Native 576x320</span>
          <span>→</span>
          <span className="bg-slate-800 px-2 py-1 rounded">2. Temporal Upscale (XL)</span>
          <span>→</span>
          <span className="bg-slate-800 px-2 py-1 rounded">3. Real-ESRGAN 2x-4x</span>
        </div>
      </div>
    </div>
  );
}

function HelpAudio({ onNavigate }: HelpSectionProps) {
  return (
    <div className="space-y-6 max-w-4xl text-slate-300 text-sm">
      <p className="text-base">
        Compose music, SFX, and speech entirely offline.
      </p>

       <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Models" icon={<Music size={16} className="text-pink-400"/>}>
           <ul className="list-disc pl-4 space-y-1">
             <li><strong>MusicGen</strong>: High-fidelity music composition.</li>
             <li><strong>AudioLDM 2</strong>: Best for Sound Effects (Foley, Rain).</li>
             <li><strong>Bark</strong>: Realistic Speech + Emotion (Laughter).</li>
             <li><strong>Stable Audio</strong>: <GatedLock onClick={() => onNavigate('gated-models')}/> Variable-length high-quality.</li>
           </ul>
        </InfoCard>

         <InfoCard title="Features" icon={<Wand2 size={16} className="text-violet-400"/>}>
           <ul className="list-disc pl-4 space-y-1">
             <li><strong>Auto-Chunking</strong>: Unlimited length generation (Bark/MusicGen).</li>
             <li><strong>Visual-to-Audio</strong>: Generate soundtrack from image/video.</li>
             <li><strong>Voice Presets</strong>: Multilingual support in Bark.</li>
           </ul>
        </InfoCard>
      </div>

       <SectionTitle icon={<MessageSquare size={20}/>}>Bark Special Tokens</SectionTitle>
       <p className="mb-2">Add these to your prompt to trigger sound effects:</p>
       <div className="flex flex-wrap gap-2 mb-4">
         <CodeBadge>[laughter]</CodeBadge>
         <CodeBadge>[cheers]</CodeBadge>
         <CodeBadge>[music]</CodeBadge>
         <CodeBadge>[sighs]</CodeBadge>
         <CodeBadge>[gasps]</CodeBadge>
         <CodeBadge>[clears throat]</CodeBadge>
         <CodeBadge>♪ lyrics ♪</CodeBadge>
       </div>
       <p className="text-xs text-slate-500 mb-4">Note: Bark ignores the 'Duration' setting; length depends on text amount. Use <CodeBadge>--voice-preset v2/en_speaker_6</CodeBadge> for best results.</p>
    </div>
  );
}

function HelpText() {
    return (
     <div className="space-y-6 max-w-4xl text-slate-300 text-sm">
       <p className="text-base">
         Unified hub for <strong>Articles</strong>, <strong>Code</strong>, and <strong>Deep Research</strong> using LLMs.
       </p>
 
       <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
         <InfoCard title="Capabilities" icon={<FileText size={16} className="text-emerald-400"/>}>
            <ul className="list-disc pl-4 space-y-1">
             <li><strong>Article</strong>: Context-aware offline writing.</li>
             <li><strong>Deep Research</strong>: Autonomous web search (DuckDuckGo).</li>
             <li><strong>Chat</strong>: Local ChatGPT style with `/read` commands.</li>
             <li><strong>Code</strong>: Project scaffolding and script generation.</li>
            </ul>
         </InfoCard>
         
          <InfoCard title="DeepSeek R1" icon={<Cpu size={16} className="text-blue-400"/>}>
            <p>
              We support <strong>DeepSeek R1 Distilled</strong> models (Qwen/Llama). 
              These are "Reasoning" models that output a <code className="text-amber-200">&lt;think&gt;</code> process before answering.
              Excellent for complex logic, math, and coding.
            </p>
         </InfoCard>
       </div>

       <SectionTitle icon={<Terminal size={20}/>}>Chat Commands</SectionTitle>
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
    <div className="space-y-6 max-w-4xl text-slate-300 text-sm">
      <p className="text-base">
         Modify existing images using AI-powered instructional commands or computer vision tasks.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
         <InfoCard title="Tools" icon={<Palette size={16} className="text-pink-400"/>}>
             <ul className="list-disc pl-4 space-y-1">
              <li><strong>InstructPix2Pix</strong>: Stylistic edits ("Make it anime").</li>
              <li><strong>Qwen-Image-Edit</strong>: Precision text/object edits.</li>
              <li><strong>Background Removal</strong>: RMBG-1.4 (Transparent PNG).</li>
            </ul>
         </InfoCard>

         <InfoCard title="Guidance Scale" icon={<Wand2 size={16} className="text-purple-400"/>}>
            <ul className="list-disc pl-4 space-y-1">
              <li><strong>&lt; 1.2</strong>: Creative, loose adherence to original.</li>
              <li><strong>&gt; 1.5</strong>: Strict, keeps original structure.</li>
              <li>Default is <strong>1.5</strong>.</li>
            </ul>
         </InfoCard>
      </div>

      <SectionTitle icon={<Book size={20}/>}>Recipe Book</SectionTitle>
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
    <div className="space-y-6 max-w-4xl text-slate-300 text-sm">
       <p className="text-base">
         Use Vision-Language Models (VLMs) to give eyes to your AI. Describes images or videos.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <InfoCard title="Image Captioning" icon={<Image size={16} className="text-blue-400"/>}>
           <p className="mb-2">Two models available:</p>
           <ul className="list-disc pl-4 space-y-1">
            <li><strong>Florence-2</strong> (Default): Rich detail, spatial awareness.</li>
            <li><strong>BLIP</strong>: Short, concise captions ("A dog on a bench").</li>
           </ul>
        </InfoCard>

        <InfoCard title="Video Analysis" icon={<Film size={16} className="text-purple-400"/>}>
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
    <div className="space-y-6 max-w-4xl text-slate-300 text-sm">
       <p className="text-base">
         Instantly convert images, videos, audio, and documents between formats (No AI required).
      </p>

       <SectionTitle icon={<FileType size={20}/>}>Document Conversion Matrix</SectionTitle>
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
       <p className="text-xs text-slate-500 mt-2">
          📷 = <strong>OCR (Optical Character Recognition)</strong>: Extract text from images/scans.
       </p>
       <div className="bg-slate-950/50 p-4 rounded-lg border border-slate-800 mt-2">
         <h4 className="text-xs font-bold text-slate-200 mb-2 uppercase tracking-tight">OCR Model Options</h4>
         <div className="space-y-3">
           <div>
             <div className="flex justify-between items-center mb-1">
               <span className="text-xs font-bold text-white">Qwen-VL (Default)</span>
               <span className="text-[10px] text-blue-400 bg-blue-400/10 px-1.5 py-0.5 rounded border border-blue-400/20">PRECISE</span>
             </div>
             <p className="text-[11px] text-slate-400">~30GB RAM usage. High-precision extraction of code, paths, and 🐍 emojis.</p>
           </div>
           <div className="pt-2 border-t border-slate-800">
             <div className="flex justify-between items-center mb-1">
               <span className="text-xs font-bold text-white">Florence-2 (Fast Choice)</span>
               <span className="text-[10px] text-green-400 bg-green-400/10 px-1.5 py-0.5 rounded border border-green-400/20">FAST</span>
             </div>
             <p className="text-[11px] text-slate-400 italic">Lightweight (~1.5GB RAM). Best for quick scans and general text drafts.</p>
           </div>
         </div>
       </div>
    </div>
  )
}


// --- Main Modal ---

const SECTIONS = [
  { id: 'gated-models', label: 'Gated Models', icon: <Lock size={18} />, component: HelpGatedModels },
  { id: 'general', label: 'General Info', icon: <Book size={18} />, component: () => (
      <div className="text-slate-300 space-y-4">
        <p>Select a topic from the sidebar to view detailed documentation.</p>
        <InfoCard title="About AI-Media" icon={<Terminal size={16}/>}>
          AI-Media is a comprehensive local generative AI studio. 
          All models run offline on your hardware (support for Mac MPS and NVIDIA CUDA).
        </InfoCard>
      </div>
  )},
  { id: 'image', label: 'Image Generation', icon: <Image size={18} />, component: HelpImage },
  { id: 'video', label: 'Video Generation', icon: <Film size={18} />, component: HelpVideo },
  { id: 'audio', label: 'Audio Generation', icon: <Music size={18} />, component: HelpAudio },
  { id: 'text', label: 'Text & Deep Research', icon: <FileText size={18} />, component: HelpText },
  { id: 'transform', label: 'Transformations', icon: <Wand2 size={18} />, component: HelpTransform },
  { id: 'description', label: 'Vision / Description', icon: <ScanEye size={18} />, component: HelpDescription },
  { id: 'multimedia', label: 'Converters', icon: <FileType size={18} />, component: HelpMultimedia },
];

export function HelpModal() {
  const { isHelpOpen, toggleHelp } = useAppStore();
  const [activeSection, setActiveSection] = useState('image'); 

  if (!isHelpOpen) return null;

  const activeItem = SECTIONS.find(s => s.id === activeSection) || SECTIONS[2];
  const ActiveComponent = activeItem.component;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm" onClick={toggleHelp}>
      <div 
        className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full max-w-6xl h-[85vh] flex flex-col md:flex-row overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Sidebar / Top Nav */}
        <div className="w-full md:w-64 bg-slate-950 border-b md:border-b-0 md:border-r border-slate-800 flex flex-row md:flex-col shrink-0 overflow-x-auto md:overflow-visible">
          <div className="p-4 border-r md:border-r-0 md:border-b border-slate-800 flex items-center gap-2 font-semibold text-slate-200 shrink-0 sticky left-0 bg-slate-950 z-10">
            <Book size={20} className="text-primary-400" />
            <span className="hidden md:inline">Help Guide</span>
            <span className="md:hidden">Help</span>
          </div>
          
          <div className="flex-1 md:overflow-y-auto p-2 flex md:block gap-2 md:gap-0 md:space-y-1">
            {SECTIONS.map((section) => (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`flex items-center gap-2 md:gap-3 px-3 py-2 rounded-lg text-sm transition-colors text-left shrink-0 md:w-full ${
                  activeSection === section.id 
                    ? 'bg-primary-500/10 text-primary-400 border border-primary-500/20' 
                    : 'text-slate-400 hover:bg-slate-900 hover:text-slate-200 border border-transparent'
                }`}
              >
                {section.icon}
                <span className="whitespace-nowrap md:truncate">{section.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 flex flex-col bg-slate-900 overflow-hidden min-w-0">
          <div className="flex items-center justify-between p-4 border-b border-slate-800 shrink-0">
            <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2 truncate">
              {activeItem.icon}
              <span className="truncate">{activeItem.label}</span>
            </h2>
            <button 
              onClick={toggleHelp}
              className="p-1 hover:bg-slate-800 rounded-lg text-slate-400 transition-colors shrink-0"
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
