import { useState, useEffect } from 'react';
import { NumberInput } from './NumberInput';
import { Tooltip } from './Tooltip';
import { API_BASE_URL } from '../../config';

interface ResolutionSelectorProps {
  width: number;
  height: number;
  onChange: (width: number, height: number) => void;
  disabled?: boolean;
}

interface ResolutionPreset {
  label: string; // e.g., "720p"
  width: number;
  height: number;
}

export function ResolutionSelector({ width, height, onChange, disabled }: ResolutionSelectorProps) {
  const [presets, setPresets] = useState<ResolutionPreset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>('custom');

  // Fetch constants on mount
  useEffect(() => {
    fetch(`${API_BASE_URL}/api/constants`)
      .then(res => res.json())
      .then(data => {
        if (data.resolutions) {
          // Convert dict to array and sort by width (ascending)
          const loadedPresets: ResolutionPreset[] = Object.entries(data.resolutions).map(([key, val]: [string, any]) => ({
            label: key,
            width: val[0],
            height: val[1]
          }));
          
          // Sort by total pixels, then width
          loadedPresets.sort((a, b) => (a.width * a.height) - (b.width * b.height));
          
          setPresets(loadedPresets);
        }
      })
      .catch(err => console.error("Failed to fetch resolution constants:", err));
  }, []);

  // Sync dropdown with current width/height
  useEffect(() => {
    const match = presets.find(p => p.width === width && p.height === height);
    if (match) {
      setSelectedPreset(match.label);
    } else {
      setSelectedPreset('custom');
    }
  }, [width, height, presets]);

  const handlePresetChange = (label: string) => {
    if (label === 'custom') {
      setSelectedPreset('custom');
      return;
    }
    
    const preset = presets.find(p => p.label === label);
    if (preset) {
      onChange(preset.width, preset.height);
      setSelectedPreset(preset.label);
    }
  };

  return (
    <div className="space-y-4">
      {/* Dropdown */}
      <div>
        <label className="label flex items-center mb-1">
           Resolution Template
           <Tooltip content="Select a standard resolution or choose Custom to enter specific values." />
        </label>
        <select 
          className="select w-full" 
          value={selectedPreset} 
          onChange={(e) => handlePresetChange(e.target.value)}
          disabled={disabled}
        >
          <option value="custom">Custom...</option>
           {/* Group presets logically if needed, for now flat list */}
           {presets.map(p => (
             <option key={p.label} value={p.label}>
               {p.label.toUpperCase()} ({p.width}x{p.height})
             </option>
           ))}
        </select>
      </div>
      
      {/* Inputs */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="label flex items-center">
             Width
             <Tooltip content="Image/Video width in pixels." />
          </label>
          <NumberInput 
             value={width} 
             onChange={(val) => onChange(val, height)} 
             step={64} 
             min={64} 
             max={2048} 
             // If we want to strictly lock inputs when not custom, we could, 
             // but user request implies typing here switches to custom, so keep specific valid.
             // "If the user changes the values... dropwodn would automatically change to Custom"
          />
        </div>
        <div>
          <label className="label flex items-center">
             Height
             <Tooltip content="Image/Video height in pixels." />
          </label>
          <NumberInput 
             value={height} 
             onChange={(val) => onChange(width, val)} 
             step={64} 
             min={64} 
             max={2048} 
          />
        </div>
      </div>
    </div>
  );
}
