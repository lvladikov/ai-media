import { useState, useRef, useMemo, useEffect } from 'react';
import { useAppStore } from '../store';
import { Upload, RefreshCw, FileType, Loader2, ArrowRight } from 'lucide-react';
import { API_BASE_URL } from '../config';
import { JobProgressModal } from './common/JobProgressModal';
import { PreviewModal } from './PreviewModal';
import { ComparisonPreviewModal } from './common/ComparisonPreviewModal';
import { ErrorAlert } from './common/ErrorAlert';

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
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [inputPreviewUrl, setInputPreviewUrl] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  return (
    <div className="flex h-full bg-slate-900 text-slate-200 items-center justify-center p-8">
      <div className="w-full max-w-2xl bg-slate-950/50 border border-slate-800 rounded-xl p-8 shadow-2xl">
         <div className="text-center mb-8">
           <div className="w-16 h-16 bg-blue-500/10 text-blue-400 rounded-full flex items-center justify-center mx-auto mb-4 border border-blue-500/20">
             <RefreshCw size={32} />
           </div>
           <h2 className="text-2xl font-bold text-white mb-2">Media Converter</h2>
           <p className="text-slate-400">Convert images, video, audio, and documents instantly.</p>
         </div>

         <div className="space-y-6">
            {/* Input Phase */}
            <div 
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
                file ? 'border-primary-500/50 bg-primary-500/5' : 'border-slate-700 hover:border-slate-600 hover:bg-slate-900'
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
                <div className="flex items-center justify-center gap-4">
                  <div className="w-12 h-12 bg-slate-800 rounded flex items-center justify-center text-primary-400">
                     <FileType size={24}/>
                  </div>
                  <div className="text-left">
                    <p className="font-semibold text-white">{file.name}</p>
                    <p className="text-xs text-slate-400">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                  {isUploading && <Loader2 className="animate-spin text-slate-500 ml-4"/>}
                </div>
              ) : (
                <>
                  <Upload size={32} className="mx-auto mb-2 text-slate-500"/>
                  <p className="font-medium">Click to select file</p>
                  <p className="text-sm text-slate-500">Supports images, video, audio, markdown, pdf, docx</p>
                </>
              )}
            </div>

            {/* Arrow */}
            {file && (
               <div className="flex justify-center">
                 <ArrowRight className="text-slate-600" />
               </div>
            )}

            {/* Target Options */}
            {file && availableFormats.length > 0 && (
               <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                 {availableFormats.map(fmt => (
                    <button
                      key={fmt}
                      onClick={() => setTargetFormat(fmt)}
                      className={`py-3 px-4 rounded-lg font-medium border text-sm transition-all ${
                         targetFormat === fmt 
                           ? 'bg-primary-600 border-primary-500 text-white shadow-lg shadow-primary-900/30' 
                           : 'bg-slate-900 border-slate-700 text-slate-400 hover:border-slate-600 hover:text-white'
                      }`}
                    >
                      {fmt.toUpperCase()}
                    </button>
                 ))}
               </div>
            )}
            
            {file && availableFormats.length === 0 && (
               <div className="text-center text-yellow-500 text-sm">
                 Format not supported for conversion options yet.
               </div>
            )}

            {/* Action */}
            <button
               onClick={handleConvert}
               disabled={!targetFormat || isUploading || isSubmitting}
               className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold py-4 rounded-lg shadow-xl shadow-blue-900/20 disabled:opacity-50 disabled:grayscale transition-all flex items-center justify-center gap-2 mt-4"
            >
               {isSubmitting ? (
                 <><Loader2 className="animate-spin"/> Converting...</>
               ) : (
                 <><RefreshCw size={20}/> Convert Now</>
               )}
            </button>

             <ErrorAlert error={error} onDismiss={() => setError(null)} />
            
            {result && (
              <div className="mt-6 p-4 bg-slate-900 border border-slate-800 rounded-lg flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-green-500/10 text-green-400 rounded flex items-center justify-center">
                    <FileType size={20} />
                  </div>
                  <div className="text-left overflow-hidden">
                    <p className="text-sm font-semibold truncate text-white">{result.split('/').pop()}</p>
                    <p className="text-xs text-slate-500">Conversion Complete</p>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button 
                    onClick={() => setIsPreviewOpen(true)}
                    className="btn-primary text-xs !py-1.5"
                  >
                    Preview
                  </button>
                  <a href={`http://localhost:8000/api/files/${result}`} target="_blank" rel="noreferrer" className="btn-secondary text-xs !py-1.5">
                    Download
                  </a>
                </div>
              </div>
            )}
         </div>
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
