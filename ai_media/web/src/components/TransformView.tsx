import { useState, useRef, useEffect } from 'react';
import { useAppStore } from '../store';
import { Upload, Wand2, Loader2, Image as ImageIcon, ArrowRight, BookOpen, X, Dices } from 'lucide-react';
import { API_BASE_URL } from "../config";
import { ValidationTooltip } from './common/ValidationTooltip';
import { JobProgressModal } from './common/JobProgressModal';
import { ComparisonPreviewModal } from './common/ComparisonPreviewModal';
import { ErrorAlert } from './common/ErrorAlert';
import { Tooltip } from './common/Tooltip';

// Preset transformation recipes
const TRANSFORM_RECIPES = [
  { name: "Black & White", instruction: "convert to black and white photograph" },
  { name: "Oil Painting", instruction: "make it look like an oil painting with visible brush strokes" },
  { name: "Watercolor", instruction: "turn into a soft watercolor painting" },
  { name: "Van Gogh Style", instruction: "make it look like a Van Gogh painting with swirling brushstrokes" },
  { name: "Anime Style", instruction: "convert to anime illustration style" },
  { name: "Pencil Sketch", instruction: "turn into a detailed pencil sketch drawing" },
  { name: "Pop Art", instruction: "convert to vibrant pop art style like Andy Warhol" },
  { name: "Day to Night", instruction: "turn day into night with moonlight and stars" },
  { name: "Night to Day", instruction: "turn night into bright daylight" },
  { name: "Add Snow", instruction: "add snow falling and snow covering the ground" },
  { name: "Add Rain", instruction: "add rain and wet reflections" },
  { name: "Vintage Photo", instruction: "make it look like an old vintage photograph with sepia tones" },
  { name: "Cyberpunk", instruction: "add neon lights and cyberpunk aesthetic" },
  { name: "Make Younger", instruction: "make the person look 20 years younger" },
  { name: "Make Older", instruction: "make the person look 30 years older with wrinkles and gray hair" },
  { name: "Add Smile", instruction: "make the person smile naturally" },
  { name: "Add Sunglasses", instruction: "add stylish sunglasses" },
  { name: "Fantasy Scene", instruction: "transform into a magical fantasy scene with glowing elements" },
];

export function TransformView() {
  const { addJob } = useAppStore();
  const [instruction, setInstruction] = useState('');
  const [model, setModel] = useState('instruct-pix2pix');
  const [removeBg, setRemoveBg] = useState(false);
  const [guidanceScale, setGuidanceScale] = useState(1.5);
  const [showRecipes, setShowRecipes] = useState(false);
  
  const [inputImage, setInputImage] = useState<string | null>(null);
  const [serverFilePath, setServerFilePath] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const recipeRef = useRef<HTMLDivElement>(null);

  // Close recipe dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (recipeRef.current && !recipeRef.current.contains(e.target as Node)) {
        setShowRecipes(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setResult(null);
      setError(null);
      setCurrentJobId(null);
      
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
        model: effectiveModel,
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
  useEffect(() => {
    if (!currentJobId) return;
    
    const unsubscribe = useAppStore.subscribe((state) => {
        const job = state.jobs.find(j => j.job_id === currentJobId);
        if (job) {
            if (job.status === 'complete') {
                setResult(job.result_path);
                setIsSubmitting(false);
            } else if (job.status === 'failed') {
                setIsSubmitting(false);
                setError(job.error || job.message || "Transformation failed");
            } else if (job.status === 'cancelled') {
                setIsSubmitting(false);
                setError("Job cancelled.");
                setTimeout(() => setError(null), 6000);
            }
        } else {
            // Job not found in store (removed on cancellation)
            setIsSubmitting(false);
            setCurrentJobId(null);
        }
    });
    return () => unsubscribe();
  }, [currentJobId]);

  return (
    <div className="flex h-full bg-slate-900 text-slate-200">
      {/* Parameters Sidebar */}
      <div className="w-96 border-r border-slate-800 p-6 flex flex-col gap-6 overflow-y-auto">
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
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-slate-400">Instruction</label>
                    {/* Recipe Book Button */}
                    <div className="relative" ref={recipeRef}>
                      <button
                        onClick={() => setShowRecipes(!showRecipes)}
                        className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${showRecipes ? 'bg-amber-500/20 text-amber-400' : 'bg-slate-800 text-slate-400 hover:text-amber-400 hover:bg-slate-700'}`}
                        title="Recipe Book - Click for preset transformations"
                      >
                        <BookOpen size={14} />
                        <span>Recipes</span>
                      </button>
                      
                      {/* Recipe Dropdown */}
                      {showRecipes && (
                        <div className="absolute right-0 top-full mt-2 w-72 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-50 max-h-80 overflow-y-auto scrollbar-themed">
                          <div className="sticky top-0 z-10 bg-slate-800 px-3 py-2 border-b border-slate-700 flex items-center justify-between">
                            <span className="text-xs font-semibold text-amber-400 flex items-center gap-1.5">
                              <BookOpen size={12} /> Transform Recipes
                            </span>
                            <div className="flex items-center gap-1">
                              <Tooltip content="Click here to use a random recipe" align="left">
                                <button 
                                  onClick={() => {
                                    const randomRecipe = TRANSFORM_RECIPES[Math.floor(Math.random() * TRANSFORM_RECIPES.length)];
                                    setInstruction(randomRecipe.instruction);
                                    setShowRecipes(false);
                                  }}
                                  className="p-1 text-primary-400 hover:text-primary-300 hover:bg-primary-500/10 rounded transition-colors"
                                  title="Random Recipe"
                                >
                                  <Dices size={14} />
                                </button>
                              </Tooltip>
                              <button onClick={() => setShowRecipes(false)} className="p-1 text-slate-500 hover:text-white transition-colors">
                                <X size={14} />
                              </button>
                            </div>
                          </div>
                          <div className="p-1">
                            {TRANSFORM_RECIPES.map((recipe, idx) => (
                              <Tooltip key={idx} content={recipe.instruction} align="left">
                                <button
                                  onClick={() => {
                                    setInstruction(recipe.instruction);
                                    setShowRecipes(false);
                                  }}
                                  className="w-full text-left px-3 py-2 text-xs rounded hover:bg-slate-700 transition-colors group"
                                >
                                  <span className="font-medium text-slate-200 group-hover:text-amber-400">{recipe.name}</span>
                                  <p className="text-slate-500 mt-0.5 line-clamp-1">{recipe.instruction}</p>
                                </button>
                              </Tooltip>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
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


        <ErrorAlert error={error} onDismiss={() => setError(null)} />

        <ValidationTooltip 
          error={!serverFilePath ? "Please upload an image first" : (!instruction && !removeBg ? "Please enter an instruction or select Remove BG" : null)} 
          className="w-full mt-auto"
        >
          <button
            onClick={handleTransform}
            disabled={!serverFilePath || (!instruction && !removeBg) || isSubmitting || isUploading}
             className="w-full bg-gradient-to-r from-primary-600 via-indigo-500 to-purple-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-white font-bold py-3 rounded-lg shadow-lg shadow-primary-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all"
          >
            {isSubmitting ? (
               <><Loader2 className="animate-spin" size={18} /> Processing...</>
            ) : (
               <><Wand2 size={18} /> {removeBg ? 'Remove Background' : 'Transform'}</>
            )}
          </button>
        </ValidationTooltip>

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

      {result && inputImage && (
        <ComparisonPreviewModal 
          isOpen={isPreviewOpen}
          onClose={() => setIsPreviewOpen(false)}
          originalPath={inputImage}
          resultPath={result}
          fileName={result.split('/').pop() || 'transformed.png'}
          resultLabel="Transformed"
        />
      )}
    </div>
  );
}
