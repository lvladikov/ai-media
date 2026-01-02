import { useState, useRef } from 'react';
import { useAppStore } from '../store';
import { TrendingUp, Zap, Loader2, Wand2, Image as ImageIcon, AlertTriangle, ArrowRight } from 'lucide-react';
import { API_BASE_URL } from '../config';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';

export function UpscaleView() {
  const { addJob } = useAppStore();
  const [file, setFile] = useState<File | null>(null);
  const [inputPreview, setInputPreview] = useState<string | null>(null);
  const [serverFilePath, setServerFilePath] = useState<string | null>(null);
  
  const [factor, setFactor] = useState(2.0); // 2 or 4
  const [method, setMethod] = useState('fast'); // fast, ai, simple
  const [strength, setStrength] = useState(0.3); // 0.0 - 1.0 (denoising)
  const [isUploading, setIsUploading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      
      // Preview if image
      if (selectedFile.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (ev) => setInputPreview(ev.target?.result as string);
        reader.readAsDataURL(selectedFile);
      } else {
        setInputPreview(null);
      }

      // Upload
      setIsUploading(true);
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      try {
        const response = await fetch(`${API_BASE_URL}/api/upload`, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) throw new Error('Upload failed');
        
        const data = await response.json();
        setServerFilePath(data.path);
      } catch (error) {
        console.error("Upload error:", error);
        alert("Failed to upload file");
      } finally {
        setIsUploading(false);
      }
    }
  };

  const handleUpscale = async () => {
    if (!serverFilePath) return;

    setIsSubmitting(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upscale`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_path: serverFilePath,
          factor: factor,
          method: method,
          strength: strength,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Upscale request failed');
      }

      const data = await response.json();
      
      setCurrentJobId(data.job_id);

      addJob({
        job_id: data.job_id,
        type: 'upscale',
        status: 'pending',
        progress: 0,
        phase: 'queued',
        message: 'Job queued',
        result_path: null,
        error: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });
      
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to start upscale");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Watch for job completion
  useState(() => {
    const unsubscribe = useAppStore.subscribe((state) => {
        if (!currentJobId) return;
        const job = state.jobs.find(j => j.job_id === currentJobId);
        if (job) {
            if (job.status === 'complete') {
                setResult(job.result_path);
            }
        }
    });
    return () => unsubscribe();
  });

  return (
    <div className="flex h-full bg-slate-900 text-slate-200">
       {/* Sidebar Params */}
       <div className="w-80 border-r border-slate-800 p-6 flex flex-col gap-6 overflow-y-auto">
         <div>
          <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
            <TrendingUp className="text-emerald-400" /> Upscale
          </h2>
          <p className="text-xs text-slate-500">Enhance resolution and details</p>
        </div>

        {/* Upload */}
        <div 
          className="border-2 border-dashed border-slate-700 rounded-lg p-6 flex flex-col items-center justify-center gap-2 hover:border-emerald-500/50 hover:bg-slate-800/50 transition-colors cursor-pointer relative"
          onClick={() => fileInputRef.current?.click()}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            className="hidden" 
            accept="image/*,video/*"
            onChange={handleFileSelect}
          />
          
          {inputPreview ? (
             <img src={inputPreview} alt="Preview" className="max-h-32 object-contain rounded" />
          ) : (
            <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center text-slate-400">
               <ImageIcon size={24}/>
            </div>
          )}
          
          <div className="text-center">
             <p className="text-sm font-medium">{file ? file.name : "Upload Media"}</p>
             <p className="text-xs text-slate-500">Image or Video</p>
          </div>
          
          {isUploading && (
             <div className="absolute inset-0 bg-black/60 flex items-center justify-center rounded-lg">
               <Loader2 className="animate-spin text-emerald-400" />
             </div>
          )}
        </div>

        {/* Controls */}
        <div className="space-y-6">
            
            {/* Factor */}
            <div className="space-y-2">
               <label className="text-sm font-medium text-slate-400">Scale Factor</label>
               <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800">
                 {[2.0, 4.0].map(f => (
                   <button 
                     key={f}
                     className={`flex-1 py-1.5 text-sm font-medium rounded ${factor === f ? 'bg-slate-800 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
                     onClick={() => setFactor(f)}
                   >
                     {f}x
                   </button>
                 ))}
               </div>
            </div>

            {/* Method */}
            <div className="space-y-2">
               <label className="text-sm font-medium text-slate-400">Method</label>
               <div className="grid grid-cols-1 gap-2">
                  <button 
                    onClick={() => setMethod('fast')}
                    className={`flex items-center gap-3 p-3 rounded-lg border text-left transition-all ${
                       method === 'fast' 
                         ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400' 
                         : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-900'
                    }`}
                  >
                    <Zap size={18} />
                    <div>
                      <div className="font-medium text-sm">Fast</div>
                      <div className="text-[10px] opacity-70">Real-ESRGAN (Best for clean styles)</div>
                    </div>
                  </button>

history
                  <button 
                    onClick={() => setMethod('ai')}
                    className={`flex items-center gap-3 p-3 rounded-lg border text-left transition-all ${
                       method === 'ai' 
                         ? 'bg-purple-500/10 border-purple-500/50 text-purple-400' 
                         : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-900'
                    }`}
                  >
                    <Wand2 size={18} />
                    <div>
                      <div className="font-medium text-sm">Creative (AI)</div>
                      <div className="text-[10px] opacity-70">Latent Upscale (Adds details/texture)</div>
                    </div>
                  </button>
               </div>
            </div>

            {/* Denoise Strength (Only for AI) */}
            {method === 'ai' && (
              <div className="space-y-2 animate-in fade-in slide-in-from-top-2 duration-200">
                 <label className="text-sm font-medium text-slate-400 flex justify-between">
                    Creativity / Denoise
                    <span className="text-white">{strength}</span>
                 </label>
                 <input 
                    type="range"
                    min="0.0"
                    max="1.0"
                    step="0.05"
                    value={strength}
                    onChange={(e) => setStrength(parseFloat(e.target.value))}
                    className="w-full accent-emerald-500 h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer"
                  />
                  <p className="text-xs text-slate-500">Higher = more imagined details (modifies original)</p>
              </div>
            )}
        </div>

        <button
          onClick={handleUpscale}
          disabled={!serverFilePath || isUploading || isSubmitting}
           className="mt-auto w-full bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold py-3 rounded-lg shadow-lg shadow-emerald-900/20 disabled:opacity-50 disabled:grayscale flex items-center justify-center gap-2 transition-all"
        >
          {isSubmitting ? (
             <><Loader2 className="animate-spin" size={18} /> Upscaling...</>
          ) : (
             <><TrendingUp size={18} /> Start Upscale</>
          )}
        </button>

        {error && (
          <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-200 flex items-start gap-2">
             <AlertTriangle className="shrink-0 mt-0.5" size={18} />
             <div>
               <p className="font-semibold">Request Failed</p>
               <p className="text-sm opacity-90">{error}</p>
             </div>
          </div>
        )}
       </div>

       <div className="flex-1 p-8 flex items-center justify-center bg-slate-950/30">
          {inputPreview ? (
             <div className="flex flex-col md:flex-row items-center gap-6 max-w-full">
                {/* Original */}
                <div className="relative border-4 border-slate-800 rounded-xl overflow-hidden shadow-2xl max-w-full">
                  <img src={inputPreview} className="max-w-full max-h-[70vh] object-contain" />
                  <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-md px-3 py-1.5 rounded-lg text-xs font-mono text-white border border-white/10">
                    Original
                  </div>
                </div>

                {/* Arrow */}
                {result && <ArrowRight className="hidden md:block text-slate-600" size={32} />}

                {/* Upscaled Result */}
                {result && (
                  <div className="relative border-4 border-brand-500/30 rounded-xl overflow-hidden shadow-2xl max-w-full cursor-pointer group" onClick={() => setIsPreviewOpen(true)}>
                    <img src={`http://localhost:8000/api/files/${result}`} className="max-w-full max-h-[70vh] object-contain" />
                    <div className="absolute top-4 left-4 bg-brand-600 backdrop-blur-md px-3 py-1.5 rounded-lg text-xs font-mono text-white shadow-lg">
                      Upscaled {factor}x
                    </div>
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                       <span className="bg-white/90 text-black px-4 py-2 rounded-lg font-bold text-sm">Open Full Preview</span>
                    </div>
                  </div>
                )}
             </div>
          ) : (
            <div className="text-center text-slate-600">
              <TrendingUp size={64} className="mx-auto mb-4 opacity-20" />
              <h3 className="text-xl font-medium mb-1">Select Media to Upscale</h3>
              <p>Supports Image (PNG, JPG) and Video (MP4)</p>
            </div>
          )}
       </div>

      {currentJobId && (
        <JobProgressModal 
          jobId={currentJobId} 
          onClose={() => {
            setCurrentJobId(null);
            if (result) setIsPreviewOpen(true);
          }} 
        />
      )}

      {result && (
        <PreviewModal 
          isOpen={isPreviewOpen}
          onClose={() => setIsPreviewOpen(false)}
          filePath={result}
          fileName={result.split('/').pop() || 'upscaled.png'}
        />
      )}
    </div>
  );
}
