import { useState, useRef } from 'react';
import { useAppStore } from '../store';
import { Upload, Wand2, Loader2, Image as ImageIcon, AlertTriangle, ArrowRight } from 'lucide-react';
import { API_BASE_URL } from "../config";
import { ValidationTooltip } from './common/ValidationTooltip';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';

export function TransformView() {
  const { addJob } = useAppStore();
  const [instruction, setInstruction] = useState('');
  const [model, setModel] = useState('instruct-pix2pix');
  const [removeBg, setRemoveBg] = useState(false);
  const [guidanceScale, setGuidanceScale] = useState(1.5);
  
  const [inputImage, setInputImage] = useState<string | null>(null);
  const [serverFilePath, setServerFilePath] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      
      // Local preview
      const reader = new FileReader();
      reader.onload = (ev) => {
        if (ev.target?.result) {
          setInputImage(ev.target.result as string);
        }
      };
      reader.readAsDataURL(file);

      // Upload to server
      setIsUploading(true);
      const formData = new FormData();
      formData.append('file', file);
      
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
        alert("Failed to upload image");
      } finally {
        setIsUploading(false);
      }
    }
  };

  const handleTransform = async () => {
    if (!serverFilePath) {
      alert("Please upload an image first");
      return;
    }
    
    if (!instruction && !removeBg) {
      alert("Please enter an instruction or select 'Remove Background'");
      return;
    }

    setIsSubmitting(true);
    setError(null);
    
    // Determine effective model/instruction
    let effectiveModel = model;
    let effectiveInstruction = instruction;
    
    if (removeBg) {
      effectiveModel = 'remove-bg';
      effectiveInstruction = 'remove-bg';
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/transform`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_path: serverFilePath,
          instruction: effectiveInstruction,
          model: effectiveModel,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Transform request failed');
      }

      const data = await response.json();
      
      setCurrentJobId(data.job_id);

      addJob({
        job_id: data.job_id,
        type: 'transform',
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
      setError(err.message || "Failed to start transformation");
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
      {/* Parameters Sidebar */}
      <div className="w-80 border-r border-slate-800 p-6 flex flex-col gap-6 overflow-y-auto">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
            <Wand2 className="text-pink-400" /> Transform
          </h2>
          <p className="text-xs text-slate-500">Edit images with AI instructions</p>
        </div>
        
        {/* Upload Area */}
        <div 
          className="border-2 border-dashed border-slate-700 rounded-lg p-6 flex flex-col items-center justify-center gap-2 hover:border-primary-500/50 hover:bg-slate-800/50 transition-colors cursor-pointer relative group"
          onClick={() => fileInputRef.current?.click()}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            className="hidden" 
            accept="image/*"
            onChange={handleFileSelect}
          />
          
          {inputImage ? (
            <div className="relative w-full aspect-square bg-slate-950 rounded overflow-hidden">
               <img src={inputImage} alt="Input" className="w-full h-full object-contain" />
               <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                 <span className="text-white font-medium flex items-center gap-2"><Upload size={16}/> Change</span>
               </div>
            </div>
          ) : (
            <>
              <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center text-slate-400">
                <ImageIcon size={24} />
              </div>
              <p className="text-sm font-medium">Click to Upload Image</p>
              <p className="text-xs text-slate-500">JPG, PNG, WEBP</p>
            </>
          )}
          
          {isUploading && (
             <div className="absolute inset-0 bg-black/60 flex items-center justify-center rounded-lg">
               <Loader2 className="animate-spin text-primary-400" />
             </div>
          )}
        </div>

        {/* Options */}
        <div className="space-y-4">
           {/* Mode Toggle */}
           <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800">
             <button 
               className={`flex-1 py-1.5 text-xs font-medium rounded ${!removeBg ? 'bg-slate-800 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
               onClick={() => setRemoveBg(false)}
             >
               With Instruction
             </button>
             <button 
               className={`flex-1 py-1.5 text-xs font-medium rounded ${removeBg ? 'bg-slate-800 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
               onClick={() => setRemoveBg(true)}
             >
                Remove BG
             </button>
           </div>
           
           {!removeBg && (
             <>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-400">Instruction</label>
                  <textarea
                    value={instruction}
                    onChange={(e) => setInstruction(e.target.value)}
                    placeholder="E.g. Make it look like a van gogh painting, add fireworks, turn day into night..."
                    rows={3}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm focus:outline-none focus:border-primary-500 resize-none"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-400">Model</label>
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-sm focus:outline-none focus:border-primary-500"
                  >
                    <option value="instruct-pix2pix">InstructPix2Pix (Creative)</option>
                    <option value="qwen-image-edit">Qwen-Image-Edit (Precise)</option>
                    <option value="magic-mix">MagicMix</option>
                  </select>
                </div>
                
                 {/* 
                    Note: Guidance Scale is not explicitly passed to /api/transform yet in Python server.
                    For now UI is here but it's cosmetic until server supports it.
                 */}
                  <div className="space-y-2 opacity-50 pointer-events-none" title="Guidance Scale support coming soon to server">
                      <label className="text-sm font-medium text-slate-400 flex justify-between">
                         Guidance Scale (CFG)
                         <span className="text-primary-400">{guidanceScale}</span>
                      </label>
                       <input 
                        type="range"
                        min="1.0"
                        max="5.0"
                        step="0.1"
                        value={guidanceScale}
                        onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                        className="w-full accent-primary-500 h-1 bg-slate-800 rounded-lg appearance-none cursor-not-allowed"
                      />
                  </div>
             </>
           )}
        </div>


        <ValidationTooltip 
          error={!serverFilePath ? "Please upload an image first" : (!instruction && !removeBg ? "Please enter an instruction or select Remove BG" : null)} 
          className="w-full mt-auto"
        >
          <button
            onClick={handleTransform}
            disabled={!serverFilePath || (!instruction && !removeBg) || isSubmitting || isUploading}
             className="w-full bg-gradient-to-r from-primary-600 to-indigo-600 hover:from-primary-500 hover:to-indigo-500 text-white font-bold py-3 rounded-lg shadow-lg shadow-primary-900/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all"
          >
            {isSubmitting ? (
               <><Loader2 className="animate-spin" size={18} /> Processing...</>
            ) : (
               <><Wand2 size={18} /> {removeBg ? 'Remove Background' : 'Transform'}</>
            )}
          </button>
        </ValidationTooltip>

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

      {/* Main Preview Area */}
      <div className="flex-1 p-6 flex items-center justify-center bg-slate-950/30">
        {inputImage ? (
           <div className="flex flex-col items-center justify-center max-w-full h-full gap-4">
             <div className="flex flex-col md:flex-row items-center gap-4">
                {/* Input */}
                <div className="relative group rounded-lg overflow-hidden border border-slate-800 shadow-xl max-h-[70vh]">
                  <img src={inputImage} alt="Original" className="max-h-[70vh] object-contain" />
                  <div className="absolute top-2 left-2 bg-black/60 px-2 py-1 rounded text-xs backdrop-blur-sm">Original</div>
                </div>

                {/* Arrow if result available */}
                {result && <ArrowRight className="hidden md:block text-slate-600" size={32} />}

                {/* Result Preview */}
                {result && (
                  <div className="relative group rounded-lg overflow-hidden border border-brand-500/30 shadow-2xl max-h-[70vh] cursor-pointer" onClick={() => setIsPreviewOpen(true)}>
                    <img src={`http://localhost:8000/api/files/${result}`} alt="Transformed" className="max-h-[70vh] object-contain" />
                    <div className="absolute top-2 left-2 bg-brand-600 px-2 py-1 rounded text-xs text-white shadow-lg">Result</div>
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                       <span className="bg-white/90 text-black px-4 py-2 rounded-lg font-bold text-sm">Open Full Preview</span>
                    </div>
                  </div>
                )}
             </div>
             {!result && (
              <p className="text-sm text-slate-500 text-center max-w-md">
                The transformed image will appear here when complete.
              </p>
             )}
           </div>
        ) : (
          <div className="text-center text-slate-500">
            <Wand2 size={48} className="mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-medium mb-2">Ready to transform</h3>
            <p className="max-w-sm mx-auto">Upload an image from the sidebar to start editing with AI instructions.</p>
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
          fileName={result.split('/').pop() || 'transformed.png'}
        />
      )}
    </div>
  );
}
