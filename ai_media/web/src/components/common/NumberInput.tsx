import { useState, useEffect, type ChangeEvent } from 'react';

interface NumberInputProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  allowFloat?: boolean;
  disabled?: boolean;
  className?: string;
  placeholder?: string;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  title?: string;
}

export function NumberInput({
  value,
  onChange,
  min,
  max,
  step,
  allowFloat = false,
  disabled = false,
  className = "input",
  placeholder,
  onKeyDown,
  title
}: NumberInputProps) {
  const [inputValue, setInputValue] = useState(value.toString());

  // Sync internal state with external value changes
  useEffect(() => {
    setInputValue(value.toString());
  }, [value]);

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const newVal = e.target.value;

    // Allow empty string to let user clear input
    if (newVal === '') {
      setInputValue('');
      return;
    }

    // Regex validation
    // Integer: integers only
    // Float: numbers with optional single decimal point
    const regex = allowFloat 
      ? /^-?\d*\.?\d*$/ 
      : /^-?\d*$/;

    if (regex.test(newVal)) {
      setInputValue(newVal);
      
      const parsed = allowFloat ? parseFloat(newVal) : parseInt(newVal, 10);
      
      if (!isNaN(parsed)) {
        // Only trigger onChange if valid, but enforce min/max logic might be postponed to blur
        // usually standard React inputs fire change immediately.
        // We trigger it but parent can choose to clamp or not.
        onChange(parsed);
      }
    }
  };

  const handleBlur = () => {
    let parsed = allowFloat ? parseFloat(inputValue) : parseInt(inputValue, 10);
    
    if (isNaN(parsed)) {
       // Revert to last valid prop value if empty/invalid on blur
       setInputValue(value.toString());
       return;
    }

    // Clamp on blur
    if (min !== undefined && parsed < min) parsed = min;
    if (max !== undefined && parsed > max) parsed = max;

    setInputValue(parsed.toString());
    onChange(parsed);
  };

  return (
    <input
      type="text"
      inputMode={allowFloat ? "decimal" : "numeric"}
      className={className}
      value={inputValue}
      onChange={handleChange}
      onBlur={handleBlur}
      step={step}
      disabled={disabled}
      placeholder={placeholder}
      onKeyDown={onKeyDown}
      title={title}
    />
  );
}
