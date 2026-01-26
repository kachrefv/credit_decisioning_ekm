import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  DashboardIcon, 
  EvaluateIcon, 
  DecisionsIcon, 
  TrainingIcon, 
  BorrowersIcon, 
  RiskIcon, 
  SettingsIcon, 
  LogoutIcon 
} from './Icons';

interface SidebarProps {
  currentView: string;
  onNavigate: (view: string) => void;
  onLogout: () => void;
}

const menuItems = [
  { label: "Dashboard", icon: DashboardIcon, view: 'dashboard' },
  { label: "Evaluate", icon: EvaluateIcon, view: 'evaluate' },
  { label: "Decisions", icon: DecisionsIcon, view: 'decisions' },
  { label: "Training", icon: TrainingIcon, view: 'train' },
  { label: "Borrowers", icon: BorrowersIcon, view: 'borrowers' },
  { label: "Risk Analysis", icon: RiskIcon, view: 'graph' },
];

const Sidebar: React.FC<SidebarProps> = ({ currentView, onNavigate, onLogout }) => {
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);

  return (
    <aside className="hidden md:flex flex-col w-64 bg-gradient-to-b from-slate-900 to-slate-800 border-r border-slate-700 fixed inset-y-0 left-0 z-50">
      {/* Brand */}
      <div className="h-16 flex items-center px-6 border-b border-slate-700 cursor-pointer group" onClick={() => onNavigate('dashboard')}>
        <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-2 rounded-xl mr-3 group-hover:scale-105 transition-transform duration-300">
          <div className="w-5 h-5 bg-white rounded-sm rotate-45"></div>
        </div>
        <div className="flex flex-col">
          <p className="font-bold text-white text-lg tracking-tight uppercase leading-tight">Credithos</p>
          <span className="text-[9px] text-slate-400 font-bold tracking-widest leading-none">POWERED BY EKM</span>
        </div>
      </div>

      {/* Navigation Items */}
      <nav className="flex-1 px-3 py-6 space-y-1">
        {menuItems.map((item) => {
          const IconComponent = item.icon;
          const isActive = currentView === item.view;
          
          return (
            <motion.button
              key={item.view}
              onClick={() => onNavigate(item.view)}
              onMouseEnter={() => setHoveredItem(item.view)}
              onMouseLeave={() => setHoveredItem(null)}
              className={`
                w-full flex items-center px-4 py-3 rounded-xl transition-all duration-300 group relative overflow-hidden
                ${isActive 
                  ? 'bg-gradient-to-r from-blue-600/20 to-purple-600/20 text-blue-300 shadow-lg shadow-blue-500/10' 
                  : 'text-slate-300 hover:bg-slate-700/50 hover:text-white'}
              `}
              whileHover={{ x: 4 }}
              whileTap={{ scale: 0.98 }}
            >
              <motion.div
                animate={{ 
                  scale: hoveredItem === item.view ? 1.2 : 1,
                  rotate: hoveredItem === item.view ? 5 : 0
                }}
                transition={{ type: "spring", stiffness: 400, damping: 17 }}
              >
                <IconComponent 
                  className={`${isActive ? 'text-blue-300' : 'text-slate-400 group-hover:text-white'}`} 
                  size="md" 
                />
              </motion.div>
              
              <motion.span 
                className={`ml-3 font-semibold text-sm ${isActive ? 'text-blue-300' : 'text-slate-300 group-hover:text-white'}`}
                animate={{ x: hoveredItem === item.view ? 4 : 0 }}
              >
                {item.label}
              </motion.span>
              
              {isActive && (
                <motion.div 
                  className="absolute right-3 w-2 h-2 rounded-full bg-blue-400 shadow-[0_0_8px_rgba(96,165,250,0.6)]"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.2 }}
                ></motion.div>
              )}
              
              <AnimatePresence>
                {hoveredItem === item.view && !isActive && (
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-purple-500/10"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.2 }}
                  />
                )}
              </AnimatePresence>
            </motion.button>
          );
        })}
      </nav>

      {/* User Menu */}
      <div className="p-4 border-t border-slate-700">
        <div className="bg-slate-800/50 rounded-2xl p-4 border border-slate-700/50">
          <div className="flex items-center justify-between mb-3">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">System Status</span>
            <div className="flex items-center">
              <div className="w-2 h-2 rounded-full bg-green-500 mr-1.5 animate-pulse"></div>
              <span className="text-[10px] font-medium text-green-400">Online</span>
            </div>
          </div>
          <div className="h-2 w-full bg-slate-700 rounded-full overflow-hidden">
            <motion.div 
              className="h-full bg-gradient-to-r from-blue-500 to-purple-600 rounded-full"
              initial={{ width: "0%" }}
              animate={{ width: "85%" }}
              transition={{ duration: 1, ease: "easeOut" }}
            ></motion.div>
          </div>
          
          <motion.button
            onClick={onLogout}
            className="mt-4 w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-300 text-sm font-medium transition-colors"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <LogoutIcon size="sm" />
            <span>Logout</span>
          </motion.button>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;