import React from 'react';

interface InputProps {
  label?: string;
  placeholder?: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  type?: string;
  className?: string;
  disabled?: boolean;
}

const Input: React.FC<InputProps> = ({ 
  label, 
  placeholder, 
  value, 
  onChange, 
  type = 'text', 
  className = '', 
  disabled = false 
}) => {
  return (
    <div className="space-y-2 w-full">
      {label && (
        <label className="text-sm font-bold text-slate-300 uppercase tracking-wider">
          {label}
        </label>
      )}
      <input
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        disabled={disabled}
        className={`
          w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-slate-200 
          focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
          transition-all duration-300
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          ${className}
        `}
      />
    </div>
  );
};

export default Input;