import React from 'react';
import { motion } from 'framer-motion';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  elevated?: boolean;
  gradient?: boolean;
}

const Card: React.FC<CardProps> = ({ children, className = '', elevated = true, gradient = false }) => {
  return (
    <motion.div
      className={`
        rounded-2xl border
        ${gradient 
          ? 'bg-gradient-to-br from-slate-800/50 to-slate-900/50 border-slate-700/50' 
          : 'bg-slate-800/30 border-slate-700/30'}
        ${elevated ? 'shadow-xl shadow-slate-900/20' : ''}
        backdrop-blur-sm
        overflow-hidden
        ${className}
      `}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      {children}
    </motion.div>
  );
};

interface CardHeaderProps {
  children: React.ReactNode;
  className?: string;
}

const CardHeader: React.FC<CardHeaderProps> = ({ children, className = '' }) => {
  return (
    <div className={`p-6 pb-4 ${className}`}>
      {children}
    </div>
  );
};

interface CardBodyProps {
  children: React.ReactNode;
  className?: string;
}

const CardBody: React.FC<CardBodyProps> = ({ children, className = '' }) => {
  return (
    <div className={`p-6 pt-0 ${className}`}>
      {children}
    </div>
  );
};

Card.Header = CardHeader;
Card.Body = CardBody;

export default Card;