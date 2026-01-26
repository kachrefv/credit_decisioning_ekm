import React from 'react';
import { motion } from 'framer-motion';
import { User } from '@heroui/react';
import { SettingsIcon, LogoutIcon } from './Icons';

interface HeaderProps {
  currentView: string;
  onNavigate: (view: string) => void;
  onLogout: () => void;
}

const menuItems = [
  { label: "Dashboard", view: 'dashboard' },
  { label: "Evaluate", view: 'evaluate' },
  { label: "Decisions", view: 'decisions' },
  { label: "Training", view: 'train' },
  { label: "Borrowers", view: 'borrowers' },
  { label: "Risk Analysis", view: 'graph' },
];

const Header: React.FC<HeaderProps> = ({ currentView, onNavigate, onLogout }) => {
  return (
    <header className="h-16 flex items-center justify-between px-6 bg-gradient-to-r from-slate-800/80 to-slate-900/80 backdrop-blur-xl border-b border-slate-700/50 sticky top-0 z-40">
      <div className="flex items-center">
        <div className="md:hidden flex items-center mr-3">
          <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-1.5 rounded-lg mr-2">
            <div className="w-4 h-4 bg-white rounded-sm rotate-45"></div>
          </div>
          <p className="font-bold text-lg tracking-tight uppercase leading-none text-white">Credithos</p>
        </div>
        <motion.h1 
          className="hidden md:block text-base font-bold text-slate-200 uppercase tracking-widest"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          {menuItems.find(i => i.view === currentView)?.label || 'Dashboard'}
        </motion.h1>
      </div>

      <div className="flex items-center gap-4">
        <motion.button
          onClick={onLogout}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-300 text-sm font-medium transition-colors"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <LogoutIcon size="sm" />
          <span>Logout</span>
        </motion.button>
        
        <motion.div
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <User
            as="button"
            avatarProps={{
              isBordered: true,
              src: "https://i.pravatar.cc/150?u=a042581f4e29026704d",
              size: "sm",
              className: "border-slate-600"
            }}
            className="transition-transform bg-slate-700/50 p-1 rounded-full"
            description="Administrator"
            name="Admin User"
            classNames={{
              name: "text-white font-bold",
              description: "text-slate-400 text-xs"
            }}
          />
        </motion.div>
      </div>
    </header>
  );
};

export default Header;