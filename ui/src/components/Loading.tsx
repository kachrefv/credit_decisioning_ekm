import { Spinner } from "@heroui/react";

interface LoadingProps {
    label?: string;
    size?: "sm" | "md" | "lg";
    color?: "default" | "primary" | "secondary" | "success" | "warning" | "danger";
}

export const FullPageLoading = ({ label = "Initializing Credithos...", size = "lg", color = "primary" }: LoadingProps) => {
    return (
        <div className="fixed inset-0 z-[9999] flex flex-col items-center justify-center bg-white/80 dark:bg-black/80 backdrop-blur-md animate-in fade-in duration-300">
            <div className="relative mb-8">
                <div className="absolute inset-0 bg-primary/20 blur-3xl rounded-full scale-150 animate-pulse"></div>
                <div className="relative bg-primary p-4 rounded-3xl shadow-2xl shadow-primary/40 rotate-12 animate-in zoom-in duration-500 delay-150">
                    <div className="w-8 h-8 bg-white rounded-md rotate-45"></div>
                </div>
            </div>
            <Spinner
                size={size}
                color={color as any}
                label={label}
                labelColor={color as any}
                className="font-bold tracking-tighter"
            />
            <div className="mt-8 flex gap-4 text-[10px] font-black uppercase tracking-[0.2em] text-default-400">
                <span>Episodic Knowledge Mesh</span>
                <span className="opacity-50">•</span>
                <span>Active Neurons</span>
            </div>
        </div>
    );
};

export const LoadingOverlay = ({ label = "Processing...", size = "md", color = "primary" }: LoadingProps) => {
    return (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-white/60 dark:bg-black/60 backdrop-blur-[2px] rounded-inherit transition-all animate-in fade-in duration-300">
            <div className="bg-white/90 dark:bg-zinc-900/90 p-6 rounded-3xl shadow-xl flex flex-col items-center gap-3 border border-default-100">
                <Spinner size={size} color={color as any} />
                <span className="text-sm font-bold animate-pulse text-primary">{label}</span>
            </div>
        </div>
    );
};

export const ComponentLoading = ({ label = "Loading Data...", size = "md", color = "primary" }: LoadingProps) => {
    return (
        <div className="flex flex-col items-center justify-center p-12 gap-4">
            <Spinner size={size} color={color} />
            <span className="text-sm font-medium text-default-500">{label}</span>
        </div>
    );
};
