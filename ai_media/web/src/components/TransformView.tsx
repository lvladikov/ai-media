import { useState, useRef, useEffect } from 'react';
import { useAppStore } from '../store';
import { Upload, Wand2, Loader2, Image as ImageIcon, ArrowRight, BookOpen, X, Dices } from 'lucide-react';
import { API_BASE_URL } from "../config";
import { ValidationTooltip } from './common/ValidationTooltip';
import { JobProgressModal } from './common/JobProgressModal';
import { ComparisonPreviewModal } from './common/ComparisonPreviewModal';
import { ErrorAlert } from './common/ErrorAlert';
import { Tooltip } from './common/Tooltip';
import { DragDropZone } from './common/DragDropZone';
import { formatDuration } from '../utils/formatTime';
import { ModelHelpLink } from './common/ModelHelpLink';
import { getDynamicRam } from '../utils/modelResources';
import { HelpCircle } from 'lucide-react';

// Check if a file extension can't be previewed in browsers
const isNonPreviewableFormat = (filename: string): boolean => {
  const ext = filename.split('.').pop()?.toLowerCase();
  return ['tiff', 'tif', 'psd', 'raw'].includes(ext || '');
};

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

const REMOVE_RECIPES = [
  { name: "Remove Text", instruction: "remove the text" },
  { name: "Remove Objects", instruction: "remove the objects" },
  { name: "Remove People", instruction: "remove the people" },
  { name: "Remove Background", instruction: "remove the background" },
  { name: "Remove Wires", instruction: "remove the power lines and wires" },
  { name: "Remove Watermark", instruction: "remove the watermark" },
];

export function TransformView() {
  const { addJob } = useAppStore();
  const [instruction, setInstruction] = useState('');
  const [model, setModel] = useState('instruct-pix2pix');
  const [activeTab, setActiveTab] = useState<'instruction' | 'rembg' | 'remove-object'>('instruction');
  const [silhouette, setSilhouette] = useState(false);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [imageGuidanceScale, setImageGuidanceScale] = useState(1.5);
  const [showRecipes, setShowRecipes] = useState(false);

  const [framework, setFramework] = useState(navigator.userAgent.toLowerCase().includes('mac') ? 'mlx' : 'auto');
  const [precision, setPrecision] = useState("auto");

  const [inputImage, setInputImage] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [serverFilePath, setServerFilePath] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [genDuration, setGenDuration] = useState<number | null>(null);
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

  // Update guidance defaults when model changes to match CLI/Backend best practices
  useEffect(() => {
    if (model === 'instruct-pix2pix') {
      setGuidanceScale(7.5);
      setImageGuidanceScale(1.5);
    } else if (model === 'qwen-image-edit') {
      setGuidanceScale(4.0);
      setImageGuidanceScale(1.5);
    } else if (model === 'qwen-image-edit-lightning') {
      setGuidanceScale(2.0);
      setImageGuidanceScale(1.5);
    } else if (model === 'z-image') {
      setGuidanceScale(1.0); // Not used by backend, but set safe default
      setImageGuidanceScale(1.5);
    }
  }, [model]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setResult(null);
      setError(null);
      setCurrentJobId(null);

      // Local preview
      if (selectedFile.type.startsWith('image/') && !isNonPreviewableFormat(selectedFile.name)) {
        const reader = new FileReader();
        reader.onload = (ev) => {
          if (ev.target?.result) {
            setInputImage(ev.target.result as string);
          }
        };
        reader.readAsDataURL(selectedFile);
      } else {
        setInputImage(null);
      }

      // Upload to server
      setIsUploading(true);
      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const response = await fetch(`${API_BASE_URL()}/api/upload`, {
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

  const handleFileDrop = async (droppedFile: File) => {
    setFile(droppedFile);
    setResult(null);
    setError(null);
    setCurrentJobId(null);

    // Local preview
    if (droppedFile.type.startsWith('image/') && !isNonPreviewableFormat(droppedFile.name)) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        if (ev.target?.result) {
          setInputImage(ev.target.result as string);
        }
      };
      reader.readAsDataURL(droppedFile);
    } else {
      setInputImage(null);
    }

    // Upload
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', droppedFile);

    try {
      const response = await fetch(`${API_BASE_URL()}/api/upload`, {
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
  };

  const handleTransform = async () => {
    if (!serverFilePath) {
      alert("Please upload an image first");
      return;
    }

    if (!instruction && activeTab !== 'rembg') {
      alert("Please enter an instruction or select a preset");
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setGenDuration(null);

    // Determine effective model/instruction
    const isRembg = activeTab === 'rembg';
    const effectiveModel = isRembg ? "remove-bg" : model;
    const effectiveInstruction = isRembg ? "remove-bg" : instruction;

    try {
      const response = await fetch(`${API_BASE_URL()}/api/transform`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_path: serverFilePath,
          instruction: effectiveInstruction,
          model: effectiveModel,
          guidance_scale: guidanceScale,
          image_guidance_scale: imageGuidanceScale,
          silhouette: silhouette,
          framework: framework !== 'auto' ? framework : undefined,
          precision: precision !== 'auto' ? precision : undefined,
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

    let hasSeenJob = false;

    const unsubscribe = useAppStore.subscribe((state) => {
      const job = state.jobs.find(j => j.job_id === currentJobId);
      if (job) {
        hasSeenJob = true;

        if (job.status === 'complete') {
          setResult(job.result_path);

          // Calculate duration
          if (job.generation_started_at && job.updated_at) {
            const start = new Date(job.generation_started_at).getTime();
            const end = new Date(job.updated_at).getTime();
            const seconds = Math.round((end - start) / 1000);
            setGenDuration(seconds > 0 ? seconds : 1);
          } else if (job.created_at && job.updated_at) {
            const start = new Date(job.created_at).getTime();
            const end = new Date(job.updated_at).getTime();
            setGenDuration(Math.round((end - start) / 1000));
          }

          setIsSubmitting(false);
        } else if (job.status === 'failed') {
          setIsSubmitting(false);
          setError(job.error || job.message || "Transformation failed");
        } else if (job.status === 'cancelled') {
          setIsSubmitting(false);
          setError("Job cancelled.");
          setTimeout(() => setError(null), 6000);
        }
      } else if (hasSeenJob) {
        // Job was in store but now removed (e.g., cancelled)
        setIsSubmitting(false);
        setCurrentJobId(null);
      }
    });
    return () => unsubscribe();
  }, [currentJobId]);

  return (
    <div className="flex flex-col lg:flex-row h-full bg-primary text-primary">
      {/* Parameters Sidebar */}
      <div className="w-full lg:w-[500px] border-b lg:border-b-0 lg:border-r border-border p-4 lg:py-6 lg:pr-[27px] lg:pl-1 flex flex-col gap-6 overflow-y-auto shrink-0 h-auto lg:h-full">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
            <Wand2 className="text-pink-400" /> Transform
          </h2>
          <p className="text-xs text-tertiary">Edit images with AI instructions</p>
        </div>

        {/* Upload Area */}
        <DragDropZone
          className="border-2 border-dashed border-border rounded-lg p-6 flex flex-col items-center justify-center gap-2 hover:border-primary-500/50 hover:bg-secondary/50 transition-colors cursor-pointer relative group"
          draggingClassName="border-primary-500 bg-primary-500/10"
          onFileDrop={handleFileDrop}
          accept="image/*,.tiff,.tif"
        >
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept="image/*,.tiff,.tif"
            onChange={handleFileSelect}
          />

          {file && isNonPreviewableFormat(file.name) ? (
            <div className="flex flex-col items-center gap-2 py-4">
              <div className="w-20 h-20 bg-primary rounded overflow-hidden flex items-center justify-center border border-border">
                <ImageIcon size={32} className="text-tertiary" />
              </div>
              <div className="text-center">
                <p className="text-sm font-medium text-secondary">{file.name.split('.').pop()?.toUpperCase()}</p>
                <p className="text-[10px] text-tertiary uppercase">No Preview</p>
              </div>
            </div>
          ) : inputImage ? (
            <div className="flex flex-col gap-2 w-full">
              <div className="relative w-full aspect-square bg-primary rounded overflow-hidden border border-border">
                <img src={inputImage} alt="Input" className="w-full h-full object-contain" />
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                  <span className="text-white font-medium flex items-center gap-2"><Upload size={16} /> Change</span>
                </div>
              </div>
              <div className="text-center w-full px-1">
                <p className="font-medium text-primary text-xs truncate" title={file?.name}>{file?.name}</p>
                <p className="text-xs text-secondary">{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : ''}</p>
              </div>
            </div>
          ) : (
            <>
              <div className="w-12 h-12 rounded-full bg-secondary flex items-center justify-center text-secondary">
                <ImageIcon size={24} />
              </div>
              <p className="text-sm font-medium">Click or Drag & Drop</p>
              <p className="text-xs text-tertiary">JPG, PNG, WEBP, TIFF</p>
            </>
          )}

          {isUploading && (
            <div className="absolute inset-0 bg-black/60 flex items-center justify-center rounded-lg">
              <Loader2 className="animate-spin text-primary-400" />
            </div>
          )}
        </DragDropZone>

        {/* Options */}
        {/* Mode Toggle */}
        <div className="space-y-4">
          <div className="flex bg-primary p-1 rounded-lg border border-border">
            <button
              className={`flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${activeTab === 'instruction' ? 'bg-indigo-100 dark:bg-secondary text-indigo-900 dark:text-primary shadow-lg' : 'text-slate-600 dark:text-slate-400 hover:text-indigo-700 dark:hover:text-slate-200 hover:bg-indigo-50 dark:hover:bg-tertiary'}`}
              onClick={() => setActiveTab('instruction')}
            >
              Edit
            </button>
            <button
              className={`flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${activeTab === 'rembg' ? 'bg-indigo-100 dark:bg-secondary text-indigo-900 dark:text-primary shadow-lg' : 'text-slate-600 dark:text-slate-400 hover:text-indigo-700 dark:hover:text-slate-200 hover:bg-indigo-50 dark:hover:bg-tertiary'}`}
              onClick={() => setActiveTab('rembg')}
            >
              Remove BG
            </button>
            <button
              className={`flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded transition-all ${activeTab === 'remove-object' ? 'bg-indigo-100 dark:bg-secondary text-indigo-900 dark:text-primary shadow-lg' : 'text-slate-600 dark:text-slate-400 hover:text-indigo-700 dark:hover:text-slate-200 hover:bg-indigo-50 dark:hover:bg-tertiary'}`}
              onClick={() => setActiveTab('remove-object')}
            >
              Remove Object
            </button>
          </div>

          <div className="space-y-4">
            {activeTab === 'instruction' ? (
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="label">Instruction</label>
                    <div className="relative" ref={recipeRef}>
                      <button
                        onClick={() => setShowRecipes(!showRecipes)}
                        className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${showRecipes ? 'bg-indigo-100 dark:bg-amber-500/20 text-indigo-700 dark:text-amber-400' : 'bg-secondary text-secondary hover:text-indigo-700 dark:hover:text-amber-400 hover:bg-indigo-50 dark:hover:bg-tertiary'}`}
                      >
                        <BookOpen size={14} />
                        <span>Recipes</span>
                      </button>

                      {showRecipes && (
                        <div className="absolute right-0 top-full mt-2 w-72 bg-secondary border border-border rounded-lg shadow-xl z-50 max-h-80 overflow-y-auto scrollbar-themed">
                          <div className="sticky top-0 z-10 bg-secondary px-3 py-2 border-b border-border flex items-center justify-between">
                            <span className="text-xs font-semibold text-indigo-700 dark:text-amber-400 flex items-center gap-1.5">
                              <BookOpen size={12} /> Edit Recipes
                            </span>
                            <div className="flex items-center gap-1">
                              <Tooltip content="Random Recipe" align="right">
                                <button
                                  onClick={() => {
                                    const r = TRANSFORM_RECIPES[Math.floor(Math.random() * TRANSFORM_RECIPES.length)];
                                    setInstruction(r.instruction);
                                    setShowRecipes(false);
                                  }}
                                  className="p-1 text-primary-400 hover:text-primary-300 hover:bg-primary-500/10 rounded"
                                >
                                  <Dices size={14} />
                                </button>
                              </Tooltip>
                              <button onClick={() => setShowRecipes(false)} className="p-1 text-tertiary hover:text-primary transition-colors">
                                <X size={14} />
                              </button>
                            </div>
                          </div>
                          <div className="p-1">
                            {TRANSFORM_RECIPES.map((recipe, idx) => (
                              <button
                                key={idx}
                                onClick={() => {
                                  setInstruction(recipe.instruction);
                                  setShowRecipes(false);
                                }}
                                className="w-full text-left px-3 py-2 text-xs rounded hover:bg-tertiary transition-colors group"
                              >
                                <span className="font-medium text-primary group-hover:text-indigo-700 dark:group-hover:text-amber-400">{recipe.name}</span>
                                <p className="text-tertiary mt-0.5 line-clamp-1">{recipe.instruction}</p>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                  <textarea
                    value={instruction}
                    onChange={(e) => setInstruction(e.target.value)}
                    placeholder="Enter instructions or open Recipes (includes Random tool)..."
                    rows={3}
                    className="w-full bg-primary border border-border rounded-lg p-3 text-sm focus:outline-none focus:border-primary-500 resize-none"
                  />
                </div>

                <div className="space-y-4">
                  {/* Framework Selector matched to ImageGenerator logic */}
                  <div className={`space-y-1 ${!navigator.userAgent.toLowerCase().includes('mac') ? 'hidden' : ''}`}>
                    <label className="label">Platform</label>
                    <select
                      className="select w-auto bg-primary border-border text-sm focus:border-brand-500 max-w-full"
                      value={framework}
                      onChange={(e) => setFramework(e.target.value)}
                      title="Inference Framework - Use MLX for best performance on Mac"
                    >
                      <option value="mlx">MLX (Native Mac)</option>
                      <option value="torch">PyTorch (MPS)</option>
                    </select>
                  </div>

                  {/* Precision Selector */}
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <label className="label">Precision</label>
                      <button
                        onClick={() => useAppStore.getState().openHelpSection('precision')}
                        className="text-tertiary hover:text-brand-500 transition-colors"
                        title="Learn about precision options"
                      >
                        <HelpCircle size={14} />
                      </button>
                    </div>
                    <select
                      className="select w-auto bg-primary border-border text-sm focus:border-brand-500 max-w-full"
                      value={precision}
                      onChange={(e) => setPrecision(e.target.value)}
                      title="Model precision - affects speed and memory usage"
                    >
                      <option value="auto">
                        {(() => {
                          const isMac = navigator.userAgent.toLowerCase().includes('mac');
                          const isMlx = framework === 'mlx' || (framework === 'auto' && isMac);
                          return `Auto (${isMlx ? 'int4 - MLX Default' : 'bfloat16 - Default'})`;
                        })()}
                      </option>
                      <option value="int4">int4 (4-bit, Fast)</option>
                      <option value="int6">int6 (6-bit, Balanced Speed)</option>
                      <option value="int8">int8 (8-bit, Balanced Quality)</option>
                      <option value="float16">float16 (Standard)</option>
                      <option value="bfloat16">bfloat16 (Brain Float)</option>
                      <option value="float32">float32 (Slow, Max Quality)</option>
                    </select>
                  </div>

                  <div className="space-y-2">
                    <label className="label">Model</label>
                    <select
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      className="select w-full bg-primary border-border text-sm focus:border-primary-500"
                    >
                      {[
                        { id: 'instruct-pix2pix', label: 'InstructPix2Pix (Creative)' },
                        { id: 'qwen-image-edit', label: 'Qwen-Image-Edit (Precise)' },
                        { id: 'qwen-image-edit-lightning', label: 'Qwen-Edit-Lightning (Fast)' },
                        { id: 'z-image', label: 'Z-Image Turbo (Mac/MLX Fast)' }
                      ].map(m => {
                        const vram = getDynamicRam(m.id, precision, framework);
                        const isHighRam = parseInt(vram.replace('~', '').replace('GB', '')) > 32;
                        return (
                          <option key={m.id} value={m.id}>
                            {`${isHighRam ? '⚠️ ' : ''}${m.label} (${vram})`}
                          </option>
                        );
                      })}
                    </select>
                  </div>
                </div>
              </div>
            ) : activeTab === 'remove-object' ? (
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="label">What to remove?</label>
                    <div className="relative" ref={recipeRef}>
                      <button
                        onClick={() => setShowRecipes(!showRecipes)}
                        className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors ${showRecipes ? 'bg-red-500/20 text-red-600 dark:text-red-400' : 'bg-secondary text-secondary hover:text-red-600 dark:hover:text-red-400 hover:bg-tertiary'}`}
                      >
                        <Wand2 size={12} />
                        <span>Removal Presets</span>
                      </button>

                      {showRecipes && (
                        <div className="absolute right-0 top-full mt-2 w-72 bg-secondary border border-border rounded-lg shadow-xl z-50 max-h-80 overflow-y-auto scrollbar-themed">
                          <div className="sticky top-0 z-10 bg-secondary px-3 py-2 border-b border-border flex items-center justify-between">
                            <span className="text-xs font-semibold text-red-600 dark:text-red-400 flex items-center gap-1.5">
                              <X size={12} /> Common Removals
                            </span>
                            <div className="flex items-center gap-1">
                              <Tooltip content="Random Removal" align="right">
                                <button
                                  onClick={() => {
                                    const r = REMOVE_RECIPES[Math.floor(Math.random() * REMOVE_RECIPES.length)];
                                    setInstruction(r.instruction);
                                    setShowRecipes(false);
                                  }}
                                  className="p-1 text-primary-400 hover:text-primary-300 hover:bg-primary-500/10 rounded"
                                >
                                  <Dices size={14} />
                                </button>
                              </Tooltip>
                              <button onClick={() => setShowRecipes(false)} className="p-1 text-tertiary hover:text-primary transition-colors">
                                <X size={14} />
                              </button>
                            </div>
                          </div>
                          <div className="p-1">
                            {REMOVE_RECIPES.map((recipe, idx) => (
                              <button
                                key={idx}
                                onClick={() => {
                                  setInstruction(recipe.instruction);
                                  setShowRecipes(false);
                                }}
                                className="w-full text-left px-3 py-2 text-xs rounded hover:bg-tertiary transition-colors group"
                              >
                                <span className="font-medium text-primary group-hover:text-red-600 dark:group-hover:text-red-400">{recipe.name}</span>
                                <p className="text-tertiary mt-0.5 line-clamp-1">Instruction: "{recipe.instruction}"</p>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                  <textarea
                    value={instruction}
                    onChange={(e) => setInstruction(e.target.value)}
                    placeholder="Enter instructions or open Removal Presets (includes Random tool)..."
                    rows={2}
                    className="w-full bg-primary border border-border rounded-lg p-3 text-sm focus:outline-none focus:border-red-500/50 resize-none"
                  />
                  <p className="text-[10px] text-tertiary italic flex items-center gap-1">
                    <span className="text-amber-500">⚠️</span> Experimental feature using instruction-based editing.
                  </p>
                </div>

                <div className="space-y-2">
                  <label className="label flex items-center">
                    Model
                    <ModelHelpLink section="transform" />
                  </label>
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="select w-full bg-primary border-border text-sm focus:border-red-500/50"
                  >
                    <option value="qwen-image-edit">Qwen-Image-Edit (Recommended)</option>
                    <option value="instruct-pix2pix">InstructPix2Pix (Fast)</option>
                  </select>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="label">Remove Background Mode</label>
                  <div className="flex p-1 bg-primary rounded-lg border border-border">
                    <button
                      onClick={() => setSilhouette(false)}
                      className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-all ${!silhouette ? 'bg-indigo-100 dark:bg-secondary text-indigo-900 dark:text-primary shadow-sm' : 'text-slate-600 dark:text-slate-400 hover:text-indigo-700 dark:hover:text-slate-200 hover:bg-indigo-50 dark:hover:bg-tertiary'
                        }`}
                    >
                      Transparent
                    </button>
                    <button
                      onClick={() => setSilhouette(true)}
                      className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-all ${silhouette ? 'bg-indigo-100 dark:bg-secondary text-indigo-900 dark:text-primary shadow-sm' : 'text-slate-600 dark:text-slate-400 hover:text-indigo-700 dark:hover:text-slate-200 hover:bg-indigo-50 dark:hover:bg-tertiary'
                        }`}
                    >
                      Silhouette
                    </button>
                  </div>
                  <p className="text-xs text-tertiary italic">
                    {silhouette ? "Creates a black shape on transparent background" : "Removes background, keeping the subject transparent"}
                  </p>
                </div>

                <div className="p-4 bg-primary/50 border border-border rounded-lg">
                  <p className="text-xs text-secondary leading-relaxed">
                    <span className="text-primary-400 font-medium italic block mb-1">Automatic Model</span>
                    High-speed precise background removal using RMBG-1.4.
                  </p>
                </div>
              </div>
            )}

            {activeTab !== 'rembg' && (
              <div className={`space-y-4 pt-2 border-t border-border/50 ${model === 'z-image' ? 'opacity-50 pointer-events-none' : ''}`}>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-secondary flex justify-between items-center">
                    <span className="flex items-center gap-1.5">
                      Text Guidance (CFG)
                      <Tooltip content="Controls adherence. High = strict, Low = subtle." align="left" />
                    </span>
                    <span className="text-primary-400">{model === 'z-image' ? 'Auto' : guidanceScale.toFixed(1)}</span>
                  </label>
                  <input
                    type="range"
                    min="1.0"
                    max="15.0"
                    step="0.5"
                    value={guidanceScale}
                    onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
                    className="w-full accent-primary-500 h-1 bg-secondary rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-secondary flex justify-between items-center">
                    <span className="flex items-center gap-1.5">
                      Image Preservation
                      <Tooltip content="Fidelity to original structure." align="left" />
                    </span>
                    <span className="text-primary-400">{model === 'z-image' ? 'Auto' : imageGuidanceScale.toFixed(1)}</span>
                  </label>
                  <input
                    type="range"
                    min="1.0"
                    max="3.0"
                    step="0.1"
                    value={imageGuidanceScale}
                    onChange={(e) => setImageGuidanceScale(parseFloat(e.target.value))}
                    className="w-full accent-primary-500 h-1 bg-secondary rounded-lg appearance-none cursor-pointer"
                  />
                </div>
                {model === 'z-image' && (
                  <p className="text-[10px] text-tertiary text-center">Z-Image Turbo uses fixed internal guidance parameters.</p>
                )}
              </div>
            )}
          </div>
        </div>


        <ErrorAlert error={error} onDismiss={() => setError(null)} />

        <ValidationTooltip
          error={!serverFilePath ? "Please upload an image first" : (!instruction && activeTab !== 'rembg' ? "Please enter an instruction or select a preset" : null)}
          className="w-full mt-auto"
        >
          <button
            onClick={handleTransform}
            disabled={!serverFilePath || (!instruction && activeTab !== 'rembg') || isSubmitting || isUploading}
            className="w-full bg-gradient-to-r from-primary-600 via-indigo-500 to-purple-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-primary font-bold py-3 rounded-lg shadow-lg shadow-primary-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all"
          >
            {isSubmitting ? (
              <><Loader2 className="animate-spin" size={18} /> Processing...</>
            ) : (
              <><Wand2 size={18} /> {activeTab === 'rembg' ? 'Remove Background' : (activeTab === 'remove-object' ? 'Remove Object' : 'Transform')}</>
            )}
          </button>
        </ValidationTooltip>

      </div>

      {/* Main Preview Area */}
      <div className="flex-1 p-6 flex items-center justify-center bg-primary/30">
        {file && isNonPreviewableFormat(file.name) ? (
          <div className="flex flex-col items-center justify-center max-w-full h-full gap-4">
            <div className="flex flex-col md:flex-row items-stretch gap-4">
              {/* TIFF/Non-Previewable Placeholder */}
              <div className="relative border-4 border-border rounded-xl overflow-hidden shadow-2xl flex flex-col items-center justify-center p-8 bg-primary/50 min-w-[280px] min-h-[320px]">
                <ImageIcon size={64} className="text-tertiary mb-4" />
                <div className="flex flex-col items-center text-center">
                  <span className="text-lg font-bold text-secondary">{file.name.split('.').pop()?.toUpperCase()} File</span>
                  <span className="text-xs text-tertiary mt-1 uppercase tracking-widest font-mono">No Browser Preview</span>
                </div>
                <div className="absolute top-2 left-2 bg-black/70 backdrop-blur-md px-3 py-1.5 rounded-lg text-[10px] font-mono text-primary border border-white/10 uppercase tracking-tighter">
                  Original
                </div>
              </div>

              {/* Arrow if result available */}
              {result && <ArrowRight className="hidden md:block text-tertiary" size={32} />}

              {/* Result Preview */}
              {result && (
                <div className="relative border-4 border-brand-500/30 rounded-xl overflow-hidden shadow-2xl flex flex-col items-center justify-center bg-primary/50 min-w-[280px] min-h-[320px] cursor-pointer group" onClick={() => setIsPreviewOpen(true)}>
                  {isNonPreviewableFormat(result) ? (
                    <>
                      <ImageIcon size={64} className="text-tertiary mb-4" />
                      <div className="flex flex-col items-center text-center">
                        <span className="text-lg font-bold text-secondary">{result.split('.').pop()?.toUpperCase()} File</span>
                        <span className="text-xs text-tertiary mt-1 uppercase tracking-widest font-mono">No Browser Preview</span>
                      </div>
                    </>
                  ) : (
                    <img src={`${API_BASE_URL()}/api/files/${result}`} alt="Transformed" className="max-h-[70vh] object-contain" />
                  )}
                  <div className="absolute top-2 left-2 bg-brand-600 px-2 py-1 rounded text-[10px] sm:text-xs text-primary shadow-lg flex flex-col items-start leading-none gap-0.5">
                    <span className="font-bold uppercase tracking-wider">Result</span>
                    {genDuration && <span className="opacity-80 font-medium">in {formatDuration(genDuration * 1000)}</span>}
                  </div>
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                    <span className="bg-white/90 text-black px-4 py-2 rounded-lg font-bold text-sm">Open Full Preview</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : inputImage ? (
          <div className="flex flex-col items-center justify-center max-w-full h-full gap-4">
            <div className="flex flex-col md:flex-row items-center gap-4">
              {/* Input */}
              <div className="relative border-4 border-border rounded-xl overflow-hidden shadow-xl max-h-[70vh]">
                <img src={inputImage} alt="Original" className="max-h-[70vh] object-contain" />
                <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-md px-3 py-1.5 rounded-lg text-[10px] font-mono text-white border border-white/10 uppercase tracking-tighter">
                  Original
                </div>
              </div>

              {/* Arrow if result available */}
              {result && <ArrowRight className="hidden md:block text-tertiary" size={32} />}

              {/* Result Preview */}
              {result && (
                <div className="relative border-4 border-brand-500/30 rounded-xl overflow-hidden shadow-2xl max-h-[70vh] cursor-pointer group" onClick={() => setIsPreviewOpen(true)}>
                  <img src={`${API_BASE_URL()}/api/files/${result}`} alt="Transformed" className="max-h-[70vh] object-contain" />
                  <div className="absolute top-4 left-4 bg-brand-600 backdrop-blur-md px-3 py-1.5 rounded-lg border border-brand-400/20 shadow-lg flex flex-col items-start leading-none gap-0.5">
                    <span className="text-[10px] font-mono text-white uppercase tracking-tighter">Result</span>
                    {genDuration && <span className="text-[9px] opacity-80 font-medium uppercase tracking-widest">in {formatDuration(genDuration * 1000)}</span>}
                  </div>
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                    <span className="bg-white/90 text-black px-4 py-2 rounded-lg font-bold text-sm">Open Full Preview</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="text-center text-tertiary">
            <Wand2 size={48} className="mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-medium mb-2">Ready to transform</h3>
            <p className="text-secondary max-w-sm">
              Upload an image from the <span className="lg:hidden">controls above</span><span className="hidden lg:inline">sidebar</span> to start editing with AI instructions.
            </p>
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

      {result && (inputImage || serverFilePath) && (
        <ComparisonPreviewModal
          isOpen={isPreviewOpen}
          onClose={() => setIsPreviewOpen(false)}
          originalPath={inputImage || serverFilePath || ''}
          resultPath={result}
          fileName={result.split('/').pop() || 'transformed.png'}
          resultLabel="Transformed"
          originalFormat={file?.name.split('.').pop()}
          resultFormat={result.split('.').pop()}
        />
      )}
    </div>
  );
}
