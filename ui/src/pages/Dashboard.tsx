import React from 'react';
import {
    Navbar,
    NavbarBrand,
    NavbarContent,
    NavbarItem,
    Link,
    Chip,
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
}

export default function Dashboard({ onLogout, onNavigate, children }: DashboardProps) {
    const menuItems = [
        { label: "Home", icon: "📊", view: 'dashboard' },
        { label: "Evaluate", icon: "⚖️", view: 'evaluate' },
        { label: "Decisions", icon: "📜", view: 'decisions' },
        { label: "Training", icon: "🧠", view: 'train' },
        { label: "Borrowers", icon: "👥", view: 'borrowers' },
    ];

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-black text-foreground font-sans">
            <Navbar isBordered position="sticky" className="bg-white/80 dark:bg-black/80 backdrop-blur-md">
                <NavbarBrand className="cursor-pointer" onClick={() => onNavigate('dashboard')}>
                    <div className="bg-primary p-1.5 rounded-xl mr-2">
                        <div className="w-4 h-4 bg-white rounded-sm rotate-45"></div>
                    </div>
                    <p className="font-bold text-inherit text-xl tracking-tight uppercase">Credithos</p>
                    <Chip size="sm" variant="flat" color="primary" className="ml-2 text-[10px] h-5 font-bold">POWERED BY EKM</Chip>
                </NavbarBrand>

                <NavbarContent className="hidden sm:flex gap-8" justify="center">
                    {menuItems.map((item) => (
                        <NavbarItem key={item.view}>
                            <Link
                                color="foreground"
                                href="#"
                                className="text-sm font-semibold opacity-70 hover:opacity-100 transition-opacity"
                                onClick={() => onNavigate(item.view)}
                            >
                                {item.label}
                            </Link>
                        </NavbarItem>
                    ))}
                </NavbarContent>

                <NavbarContent justify="end">
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
                            <DropdownItem key="settings">Settings</DropdownItem>
                            <DropdownItem key="logout" color="danger">Log Out</DropdownItem>
                        </DropdownMenu>
                    </Dropdown>
                </NavbarContent>
            </Navbar>

            <main className="max-w-[1200px] mx-auto p-8 animate-in fade-in duration-500">
                <div className="w-full">
                    {children}
                </div>
            </main>
        </div>
    );
}
