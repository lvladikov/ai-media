import { useState, useRef, useMemo, useEffect } from 'react';
import { useAppStore } from '../store';
import { Upload, RefreshCw, FileType, Loader2, Zap } from 'lucide-react';
import { API_BASE_URL } from '../config';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { ComparisonPreviewModal } from './common/ComparisonPreviewModal';
import { ErrorAlert } from './common/ErrorAlert';
import { formatDuration } from '../utils/formatTime';

export function ConvertView() {
  const { addJob } = useAppStore();
  const [file, setFile] = useState<File | null>(null);
  const [serverFilePath, setServerFilePath] = useState<string | null>(null);
  const [targetFormat, setTargetFormat] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [genDuration, setGenDuration] = useState<number | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [inputPreviewUrl, setInputPreviewUrl] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  const availableFormats = useMemo(() => {
    if (!file) return [];

    const ext = file.name.split('.').pop()?.toLowerCase();

    // Logic mirroring server capabilities
    if (['jpg', 'jpeg', 'png', 'webp', 'bmp'].includes(ext || '')) {
      return ['png', 'jpg', 'webp'];
    }
    if (['mp4', 'mov', 'avi', 'mkv', 'webm'].includes(ext || '')) {
      return ['mp4', 'gif', 'mov', 'webm'];
    }
    if (['mp3', 'wav', 'aac', 'flac', 'ogg', 'm4a'].includes(ext || '')) {
      return ['mp3', 'wav', 'aac', 'flac'];
    }
    if (ext === 'md') {
      return ['html', 'pdf', 'docx', 'txt'];
    }
    if (ext === 'html') {
      return ['md', 'pdf', 'docx', 'txt'];
    }
    if (ext === 'docx') {
      return ['md', 'html', 'pdf', 'txt'];
    }
    if (ext === 'pdf') {
      return ['md', 'txt']; // Extraction only usually
    }

    return [];
  }, [file]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setTargetFormat(''); // Reset format selection
      setResult(null); // Reset previous result

      // Create local preview if it's an image
      if (selectedFile.type.startsWith('image/')) {
        if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
        setInputPreviewUrl(URL.createObjectURL(selectedFile));
      } else {
        if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
        setInputPreviewUrl(null);
      }

      // Upload immediately
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

  const handleConvert = async () => {
    if (!serverFilePath || !targetFormat) return;

    setIsSubmitting(true);
    setError(null);
    setResult(null);
    setGenDuration(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/convert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_path: serverFilePath,
          target_format: targetFormat,
        }),
      });

      if (!response.ok) throw new Error('Convert request failed');

      const data = await response.json();

      setCurrentJobId(data.job_id);

      addJob({
        job_id: data.job_id,
        type: 'convert',
        status: 'pending',
        progress: 0,
        phase: 'queued',
        message: 'Job queued',
        result_path: null,
        error: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });

    } catch (error) {
      console.error(error);
      setError("Failed to start conversion");
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
          setError(job.error || job.message || "Conversion failed");
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

  const handleCloseModal = () => {
    setCurrentJobId(null);
    if (result) setIsPreviewOpen(true);
  };
  
  const handleViewResult = () => {
    setTimeout(() => {
        resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  };

  return (
    <div className="flex flex-col lg:flex-row h-full bg-slate-900 text-slate-200">
      {/* Parameters Sidebar */}
      <div className="w-full lg:w-[500px] border-b lg:border-b-0 lg:border-r border-slate-800 p-4 lg:py-6 lg:pr-[27px] lg:pl-1 flex flex-col gap-6 overflow-y-auto shrink-0 h-auto lg:h-full">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2 mb-1">
            <RefreshCw className="text-blue-400" /> Media Converter
          </h2>
          <p className="text-xs text-slate-500">Convert images, video, audio, and documents</p>
        </div>

        {/* File Upload */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-400">Input File</label>
          <div
            className={`border border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer relative overflow-hidden ${file ? 'border-primary-500/50 bg-primary-500/5' : 'border-slate-700 hover:border-slate-600 hover:bg-slate-950'
              }`}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              onChange={handleFileSelect}
            />

            {file ? (
              <div className="flex flex-col items-center justify-center gap-2">
                 <div className="w-10 h-10 bg-slate-800 rounded flex items-center justify-center text-primary-400 mb-1">
                  <FileType size={20} />
                </div>
                <div className="text-center w-full">
                  <p className="font-semibold text-white text-sm truncate max-w-[250px] mx-auto">{file.name}</p>
                  <p className="text-xs text-slate-400">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
                <div className="text-xs text-primary-400 font-medium">Click to change</div>
                {isUploading && <div className="absolute top-2 right-2"><Loader2 className="animate-spin text-slate-500" size={16} /></div>}
              </div>
            ) : (
              <>
                <Upload size={24} className="mx-auto mb-2 text-slate-500" />
                <p className="font-medium text-sm">Click to select file</p>
                <p className="text-[10px] text-slate-500 mt-1">Supports img, vid, audio, docs</p>
              </>
            )}
          </div>
        </div>

        {/* Target Format */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-400">Target Format</label>
          {file && availableFormats.length > 0 ? (
            <div className="grid grid-cols-3 gap-2">
              {availableFormats.map(fmt => (
                <button
                  key={fmt}
                  onClick={() => setTargetFormat(fmt)}
                  className={`py-2 px-2 rounded-md font-medium border text-xs uppercase transition-all ${targetFormat === fmt
                    ? 'bg-blue-600 border-blue-500 text-white shadow-md shadow-blue-900/20'
                    : 'bg-slate-950 border-slate-700 text-slate-400 hover:border-slate-600 hover:text-white'
                    }`}
                >
                  {fmt}
                </button>
              ))}
            </div>
          ) : (
            <div className="p-4 bg-slate-950 border border-slate-800 rounded-lg text-center text-slate-500 text-xs italic">
                {file ? "No compatible formats found." : "Upload a file to see options."}
            </div>
          )}
        </div>

        <ErrorAlert error={error} onDismiss={() => setError(null)} />

        {/* Action Button */}
        <div className="mt-auto pt-4">
            <button
            onClick={handleConvert}
            disabled={!targetFormat || isUploading || isSubmitting}
            className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 bg-[length:200%_100%] animate-gradient-x hover:brightness-110 text-white font-bold py-3 rounded-lg shadow-lg shadow-blue-900/20 disabled:opacity-50 disabled:cursor-not-allowed disabled:animate-none flex items-center justify-center gap-2 transition-all"
            >
            {isSubmitting ? (
                <><Loader2 className="animate-spin" size={18} /> Converting...</>
            ) : (
                <><RefreshCw size={18} /> Convert Now</>
            )}
            </button>
        </div>
      </div>

      {/* Main Preview Area */}
      <div ref={resultRef} className="flex-1 p-6 flex items-center justify-center bg-slate-950/30 min-h-[500px] lg:min-h-0">
        {!result && !file && (
          <div className="text-center text-slate-500">
            <RefreshCw size={48} className="mx-auto mb-4 opacity-20" />
            <h3 className="text-lg font-medium mb-2">Ready to Convert</h3>
            <p className="text-slate-400 max-w-sm">
              Upload a file from the <span className="lg:hidden">controls above</span><span className="hidden lg:inline">sidebar</span> to see conversion options.
            </p>
          </div>
        )}

        {!result && file && inputPreviewUrl && (
             <div className="flex flex-col items-center justify-center max-w-full h-full gap-4 opacity-50 grayscale hover:grayscale-0 hover:opacity-100 transition-all duration-500">
                <div className="relative rounded-lg overflow-hidden border border-slate-700 shadow-xl max-h-[60vh]">
                    <img src={inputPreviewUrl} alt="Input Preview" className="max-h-[60vh] object-contain" />
                    <div className="absolute top-2 left-2 bg-slate-800/80 px-2 py-1 rounded text-xs text-white">Input Preview</div>
                </div>
                 <p className="text-sm text-slate-400">Select a format and click convert to process this file.</p>
            </div>
        )}
        
        {!result && file && !inputPreviewUrl && (
            <div className="text-center text-slate-500">
                <FileType size={64} className="mx-auto mb-4 opacity-20" />
                <h3 className="text-lg font-medium mb-1">{file.name}</h3>
                <p className="text-slate-400">Preview not available for this file type.</p>
            </div>
        )}

        {result && (
           <div className="flex flex-col items-center justify-center max-w-full h-full gap-6 animate-in fade-in zoom-in duration-300">
             <div className="relative group rounded-lg overflow-hidden border border-blue-500/30 shadow-2xl max-h-[70vh] cursor-pointer bg-slate-900" onClick={() => setIsPreviewOpen(true)}>
               {/* Result Preview Logic */}
               {['jpg', 'jpeg', 'png', 'webp', 'gif'].includes(result.split('.').pop()?.toLowerCase() || '') ? (
                 <img src={`${API_BASE_URL}/api/files/${result}`} alt="Converted Result" className="max-h-[70vh] object-contain" />
               ) : ['mp4', 'webm', 'mov'].includes(result.split('.').pop()?.toLowerCase() || '') ? (
                 <video src={`${API_BASE_URL}/api/files/${result}`} controls className="max-h-[70vh]" />
               ) : (
                 <div className="w-64 h-64 flex flex-col items-center justify-center bg-slate-800 text-slate-400">
                   <FileType size={64} className="mb-4" />
                   <span className="font-mono text-lg">{result.split('.').pop()?.toUpperCase()} CONTENT</span>
                 </div>
               )}

               <div className="absolute top-2 left-2 bg-blue-600 px-2 py-1 rounded text-xs text-white shadow-lg flex flex-col items-start leading-none gap-0.5">
                   <span className="font-bold uppercase tracking-wider">Converted</span>
                   {genDuration && <span className="opacity-80 font-medium">in {formatDuration(genDuration * 1000)}</span>}
               </div>

                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                    <span className="bg-white/90 text-black px-4 py-2 rounded-lg font-bold text-sm">Open Full Preview</span>
                </div>
             </div>

             <div className="flex gap-3">
               <button className="btn-secondary text-sm" onClick={() => setIsPreviewOpen(true)}>Full Preview</button>
               <a href={`${API_BASE_URL}/api/files/${result}`} target="_blank" rel="noreferrer" className="btn-secondary text-sm">Download File</a>
             </div>
           </div>
        )}
      </div>

      {currentJobId && (
        <JobProgressModal
          jobId={currentJobId}
          onClose={handleCloseModal}
          onViewResult={handleViewResult}
        />
      )}

      {result && (isPreviewOpen || isSubmitting) && (
        ['jpg', 'jpeg', 'png', 'webp'].includes(result.split('.').pop()?.toLowerCase() || '') && inputPreviewUrl ? (
          <ComparisonPreviewModal
            isOpen={isPreviewOpen}
            onClose={() => setIsPreviewOpen(false)}
            originalPath={inputPreviewUrl}
            resultPath={result}
            fileName={result.split('/').pop() || 'converted-file'}
            resultLabel="Converted"
          />
        ) : (
          <PreviewModal
            isOpen={isPreviewOpen}
            onClose={() => setIsPreviewOpen(false)}
            filePath={result}
            fileName={result.split('/').pop() || 'converted-file'}
          />
        )
      )}
    </div>
  );
}
