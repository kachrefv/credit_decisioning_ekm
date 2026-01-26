import React from 'react';
import { motion } from 'framer-motion';
import {
    DashboardIcon,
    EvaluateIcon,
    DecisionsIcon,
    TrainingIcon,
    BorrowersIcon,
    RiskIcon
} from './Icons';

interface MobileNavProps {
    currentView: string;
    onNavigate: (view: string) => void;
}

const menuItems = [
    { label: "Home", icon: DashboardIcon, view: 'dashboard' },
    { label: "Evaluate", icon: EvaluateIcon, view: 'evaluate' },
    { label: "Decisions", icon: DecisionsIcon, view: 'decisions' },
    { label: "Training", icon: TrainingIcon, view: 'train' },
    { label: "Borrowers", icon: BorrowersIcon, view: 'borrowers' },
    { label: "Risk", icon: RiskIcon, view: 'graph' },
];

const MobileNav: React.FC<MobileNavProps> = ({ currentView, onNavigate }) => {
    return (
        <nav className="md:hidden fixed bottom-6 left-4 right-4 z-50">
            <div className="absolute inset-0 bg-slate-900/90 backdrop-blur-xl rounded-2xl border border-slate-700/50 shadow-2xl shadow-black/50" />
            <div className="relative flex items-center justify-between px-2 py-2">
                {menuItems.map((item) => {
                    const IconComponent = item.icon;
                    const isActive = currentView === item.view;

                    return (
                        <motion.button
                            key={item.view}
                            onClick={() => onNavigate(item.view)}
                            className="relative flex flex-col items-center justify-center flex-1 py-1"
                            whileTap={{ scale: 0.9 }}
                        >
                            <div className={`
                relative p-2 rounded-xl transition-all duration-300
                ${isActive ? 'bg-gradient-to-tr from-blue-600/20 to-purple-600/20 shadow-lg shadow-blue-500/20' : 'bg-transparent'}
              `}>
                                <IconComponent
                                    size="sm"
                                    className={`
                    transition-colors duration-300
                    ${isActive ? 'text-blue-400' : 'text-slate-400'}
                  `}
                                />

                                {isActive && (
                                    <motion.div
                                        layoutId="activeTab"
                                        className="absolute inset-0 rounded-xl bg-blue-400/10 border border-blue-400/20"
                                        initial={false}
                                        transition={{ type: "spring", stiffness: 500, damping: 30 }}
                                    />
                                )}
                            </div>

                            <span className={`
                text-[10px] font-medium mt-1 transition-colors duration-300
                ${isActive ? 'text-blue-300' : 'text-slate-500'}
              `}>
                                {item.label}
                            </span>

                            {isActive && (
                                <motion.div
                                    layoutId="activeIndicator"
                                    className="absolute -bottom-1 w-1 h-1 rounded-full bg-blue-400 shadow-[0_0_8px_rgba(96,165,250,0.8)]"
                                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                                />
                            )}
                        </motion.button>
                    );
                })}
            </div>
        </nav>
    );
};

export default MobileNav;
