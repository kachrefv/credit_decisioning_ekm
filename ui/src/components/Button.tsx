import React from 'react';
import { motion } from 'framer-motion';

interface ButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  onClick?: () => void;
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
}

const Button: React.FC<ButtonProps> = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  className = '', 
  onClick, 
  disabled = false,
  type = 'button'
}) => {
  const baseClasses = "inline-flex items-center justify-center font-bold rounded-xl transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900";
  
  const sizeClasses = {
    sm: "text-xs px-3 py-1.5",
    md: "text-sm px-4 py-2",
    lg: "text-base px-6 py-3"
  };
  
  const variantClasses = {
    primary: "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/20 hover:shadow-blue-500/30 hover:from-blue-500 hover:to-indigo-500",
    secondary: "bg-gradient-to-r from-slate-700 to-slate-600 text-slate-200 shadow hover:from-slate-600 hover:to-slate-500",
    ghost: "text-slate-300 hover:bg-slate-800/50",
    outline: "border border-slate-600 text-slate-300 hover:bg-slate-800/50"
  };
  
  const disabledClasses = "opacity-50 cursor-not-allowed";
  
  const classes = `${baseClasses} ${sizeClasses[size]} ${variantClasses[variant]} ${disabled ? disabledClasses : ''} ${className}`;
  
  return (
    <motion.button
      className={classes}
      onClick={onClick}
      disabled={disabled}
      type={type}
      whileHover={!disabled ? { scale: 1.03 } : {}}
      whileTap={!disabled ? { scale: 0.98 } : {}}
    >
      {children}
    </motion.button>
  );
};

export default Button;