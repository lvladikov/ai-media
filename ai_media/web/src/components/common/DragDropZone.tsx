import React, { useState, useRef } from 'react';

interface DragDropZoneProps {
  onFileDrop: (file: File) => void;
  className?: string;
  draggingClassName?: string;
  rejectClassName?: string;
  children: React.ReactNode | ((dragState: { isDragging: boolean; isDragReject: boolean }) => React.ReactNode);
  onClick?: () => void;
  accept?: string; // e.g. "image/*,.pdf"
}

export function DragDropZone({ 
  onFileDrop, 
  className = "", 
  draggingClassName = "", 
  rejectClassName = "",
  children,
  onClick,
  accept
}: DragDropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isDragReject, setIsDragReject] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateItems = (items: DataTransferItemList): boolean => {
    if (!accept || accept === '*' || accept === '*/*') return true;
    
    // Quick check based on MIME type if available in items
    // Note: Extensions are often not available in dragover, only types.
    // If strict validation is needed, we usually have to wait for drop, 
    // but we can catch obvious MIME mismatches here.
    const acceptedTypes = accept.split(',').map(t => t.trim());
    
    for (let i = 0; i < items.length; i++) {
      if (items[i].kind === 'file') {
        const type = items[i].type;
        // If system doesn't provide type during drag (common), we assume valid to not block valid files.
        if (!type) continue; 
        
        const isValid = acceptedTypes.some(acc => {
          if (acc.endsWith('/*')) {
             return type.startsWith(acc.replace('/*', ''));
          }
          // Exact mime type match
          if (!acc.startsWith('.')) return type === acc;
          
          // If we have a type, we rely on MIME rules (if provided) and skip extension rules for validation
          // This prevents "everything is valid" just because an extension is in the list.
          if (type) return false;
          
          // Only fall back to allowing based on extension if we have NO type info
          return true; 
        });
        
        if (!isValid) return false;
      }
    }
    return true;
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const valid = validateItems(e.dataTransfer.items);
    setIsDragging(true);
    setIsDragReject(!valid);
    
    if (valid) {
      e.dataTransfer.dropEffect = 'copy';
    } else {
      e.dataTransfer.dropEffect = 'none';
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    setIsDragReject(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    setIsDragReject(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      
      // Strict validation on actual file object
      if (validateFile(file)) {
        onFileDrop(file);
      } else {
         // Could trigger onFileReject here if added to props
         console.warn("File rejected based on accept prop:", file.name);
      }
    }
  };

  const validateFile = (file: File): boolean => {
    if (!accept || accept === '*' || accept === '*/*') return true;
    const acceptedTypes = accept.split(',').map(t => t.trim().toLowerCase());
    const fileName = file.name.toLowerCase();
    const fileType = file.type.toLowerCase();

    return acceptedTypes.some(acc => {
       if (acc.endsWith('/*')) {
           const base = acc.replace('/*', '');
           return fileType.startsWith(base);
       }
       if (acc.startsWith('.')) {
           return fileName.endsWith(acc);
       }
       return fileType === acc;
    });
  };

  const handleClick = () => {
    fileInputRef.current?.click();
    if (onClick) onClick();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
        onFileDrop(e.target.files[0]);
    }
  };

  // Combine classes logic
  let finalClass = className;
  if (isDragging) {
      finalClass = `${className} ${isDragReject ? rejectClassName : draggingClassName}`;
  }

  return (
    <div
      className={finalClass}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <input 
        type="file" 
        ref={fileInputRef} 
        className="hidden" 
        onChange={handleFileChange}
        accept={accept}
      />
      
      {/* Function-as-child pattern */}
      {typeof children === 'function' 
        ? (children as (dragState: { isDragging: boolean; isDragReject: boolean }) => React.ReactNode)({ isDragging, isDragReject }) 
        : children}
    </div>
  );
}
