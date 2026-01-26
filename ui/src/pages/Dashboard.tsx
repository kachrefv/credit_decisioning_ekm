import React from 'react';
import {
    User,
    Dropdown,
    DropdownTrigger,
    DropdownMenu,
    DropdownItem,
} from "@heroui/react";

interface DashboardProps {
    onLogout: () => void;
    onNavigate: (view: any) => void;
    children?: React.ReactNode;
    currentView?: string;
}

export default function Dashboard({ onLogout, onNavigate, children, currentView }: DashboardProps) {
    const menuItems = [
        { label: "Home", icon: "📊", view: 'dashboard' },
        { label: "Evaluate", icon: "⚖️", view: 'evaluate' },
        { label: "Decisions", icon: "📜", view: 'decisions' },
        { label: "Training", icon: "🧠", view: 'train' },
        { label: "Borrowers", icon: "👥", view: 'borrowers' },
        { label: "Risk", icon: "🔮", view: 'graph' },
    ];

    return (
        <div className="flex min-h-screen bg-gray-50 dark:bg-black text-foreground font-sans overflow-hidden">
            {/* Sidebar - Hidden on mobile, visible on md+ */}
            <aside
                className="hidden md:flex flex-col w-64 bg-white dark:bg-zinc-950 border-r border-divider fixed inset-y-0 left-0 z-50"
            >
                {/* Brand */}
                <div
                    className="h-16 flex items-center px-6 border-b border-divider cursor-pointer"
                    onClick={() => onNavigate('dashboard')}
                >
                    <div className="bg-primary p-1.5 rounded-xl mr-3">
                        <div className="w-4 h-4 bg-white rounded-sm rotate-45"></div>
                    </div>
                    <div className="flex flex-col">
                        <p className="font-bold text-inherit text-lg tracking-tight uppercase leading-tight">Credithos</p>
                        <span className="text-[9px] text-zinc-500 font-bold tracking-widest leading-none">POWERED BY EKM</span>
                    </div>
                </div>

                {/* Navigation Items */}
                <nav className="flex-1 px-3 py-6 space-y-1">
                    {menuItems.map((item) => (
                        <button
                            key={item.view}
                            onClick={() => onNavigate(item.view)}
                            className={`
                                w-full flex items-center px-4 py-3 rounded-xl transition-all duration-200 group
                                ${currentView === item.view
                                    ? 'bg-primary/10 text-primary'
                                    : 'text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-900 hover:text-foreground'}
                            `}
                        >
                            <span className={`text-xl mr-3 transition-transform group-hover:scale-110 ${currentView === item.view ? 'grayscale-0' : 'grayscale opacity-70 group-hover:grayscale-0 group-hover:opacity-100'}`}>
                                {item.icon}
                            </span>
                            <span className={`font-semibold text-sm ${currentView === item.view ? 'translate-x-1' : ''} transition-transform`}>
                                {item.label}
                            </span>
                            {currentView === item.view && (
                                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-primary shadow-[0_0_8px_rgba(var(--heroui-primary-rgb),0.6)]"></div>
                            )}
                        </button>
                    ))}
                </nav>

                {/* Bottom Sidebar Info */}
                <div className="p-4 border-t border-divider">
                    <div className="bg-zinc-50 dark:bg-zinc-900/50 rounded-2xl p-4 border border-divider/50">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">System Status</span>
                            <div className="flex items-center">
                                <div className="w-2 h-2 rounded-full bg-green-500 mr-1.5 animate-pulse"></div>
                                <span className="text-[10px] font-medium text-green-600 dark:text-green-400">Online</span>
                            </div>
                        </div>
                        <div className="h-1.5 w-full bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden">
                            <div className="h-full bg-primary w-[85%] rounded-full"></div>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Main Content Area */}
            <div className="flex-1 flex flex-col min-w-0 md:pl-64">
                {/* Header */}
                <header className="h-16 flex items-center justify-between px-6 bg-white/80 dark:bg-black/80 backdrop-blur-md border-b border-divider sticky top-0 z-40">
                    <div className="flex items-center">
                        <div className="md:hidden flex items-center mr-3">
                            <div className="bg-primary p-1 rounded-lg mr-2">
                                <div className="w-3 h-3 bg-white rounded-sm rotate-45"></div>
                            </div>
                            <p className="font-bold text-lg tracking-tight uppercase leading-none">Credithos</p>
                        </div>
                        <h1 className="hidden md:block text-sm font-bold text-zinc-500 uppercase tracking-widest">
                            {menuItems.find(i => i.view === currentView)?.label || 'Dashboard'}
                        </h1>
                    </div>

                    <div className="flex items-center gap-4">
                        <Dropdown placement="bottom-end">
                            <DropdownTrigger>
                                <User
                                    as="button"
                                    avatarProps={{
                                        isBordered: true,
                                        src: "https://i.pravatar.cc/150?u=a042581f4e29026704d",
                                        size: "sm"
                                    }}
                                    className="transition-transform"
                                    description="Administrator"
                                    name="Admin User"
                                    aria-label="User menu"
                                />
                            </DropdownTrigger>
                            <DropdownMenu aria-label="User actions" variant="flat" onAction={(key) => {
                                if (key === 'logout') onLogout();
                                if (key === 'settings') onNavigate('profile');
                            }}>
                                <DropdownItem key="settings" startContent={<span>⚙️</span>}>Settings</DropdownItem>
                                <DropdownItem key="logout" color="danger" startContent={<span>🚪</span>}>Log Out</DropdownItem>
                            </DropdownMenu>
                        </Dropdown>
                    </div>
                </header>

                <main className="flex-1 overflow-y-auto p-4 md:p-8 pb-24 md:pb-8">
                    <div className="max-w-7xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">
                        {children}
                    </div>
                </main>
            </div>

            {/* Bottom Navigation for Mobile */}
            <nav className="md:hidden fixed bottom-6 left-1/2 -translate-x-1/2 z-50 w-[92%] max-w-lg bg-white/90 dark:bg-zinc-950/90 backdrop-blur-xl border border-divider/50 shadow-2xl rounded-3xl h-16 px-4 flex items-center justify-between">
                {menuItems.map((item) => (
                    <button
                        key={item.view}
                        onClick={() => onNavigate(item.view)}
                        className={`
                            flex flex-col items-center justify-center flex-1 h-full transition-all duration-300
                            ${currentView === item.view ? 'text-primary scale-110 -translate-y-1' : 'text-zinc-500 dark:text-zinc-400 opacity-70'}
                        `}
                    >
                        <span className={`text-xl mb-0.5 ${currentView === item.view ? 'grayscale-0' : 'grayscale'}`}>
                            {item.icon}
                        </span>
                        <span className={`text-[9px] font-bold uppercase tracking-tighter ${currentView === item.view ? 'opacity-100' : 'opacity-0 h-0 w-0'}`}>
                            {item.label}
                        </span>
                        {currentView === item.view && (
                            <div className="absolute bottom-1 w-1 h-1 rounded-full bg-primary"></div>
                        )}
                    </button>
                ))}
            </nav>
        </div>
    );
}
