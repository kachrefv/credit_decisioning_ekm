import React from 'react';
import Sidebar from '../components/Sidebar';
import Header from '../components/Header';
import MobileNav from '../components/MobileNav';

interface DashboardProps {
    onLogout: () => void;
    onNavigate: (view: any) => void;
    children?: React.ReactNode;
    currentView?: string;
}

export default function Dashboard({ onLogout, onNavigate, children, currentView }: DashboardProps) {
    return (
        <div className="flex min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-foreground font-sans overflow-hidden">
            {/* Desktop Sidebar */}
            <Sidebar
                currentView={currentView || 'dashboard'}
                onNavigate={onNavigate}
                onLogout={onLogout}
            />

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col min-w-0 md:pl-64">
                {/* Desktop Header */}
                <Header
                    currentView={currentView || 'dashboard'}
                    onNavigate={onNavigate}
                    onLogout={onLogout}
                />

                <main className="flex-1 overflow-y-auto p-4 md:p-8 pb-24 md:pb-8">
                    <div className="max-w-7xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">
                        {children}
                    </div>
                </main>
            </div>

            {/* Mobile Navigation */}
            <MobileNav
                currentView={currentView || 'dashboard'}
                onNavigate={onNavigate}
            />
        </div>
    );
}
