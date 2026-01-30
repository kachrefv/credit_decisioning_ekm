import React, { useState } from 'react';
import { Button, Card, CardBody, CardHeader, Divider } from '@heroui/react';

interface SetupProps {
    onFinish: () => void;
}

const Setup: React.FC<SetupProps> = ({ onFinish }) => {
    const [isCreating, setIsCreating] = useState(false);
    const [status, setStatus] = useState<{ type: 'success' | 'error' | null, message: string }>({ type: null, message: '' });

    const handleCreateShortcut = async () => {
        setIsCreating(true);
        setStatus({ type: null, message: '' });
        try {
            // @ts-ignore
            const result = await window.electron.ipcRenderer.invoke('create-desktop-shortcut');
            if (result.success) {
                setStatus({ type: 'success', message: 'Shortcut created successfully!' });
            } else {
                setStatus({ type: 'error', message: result.message || 'Failed to create shortcut.' });
            }
        } catch (err) {
            setStatus({ type: 'error', message: 'Error communicating with system.' });
        } finally {
            setIsCreating(false);
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-slate-950 p-4">
            <Card className="max-w-[500px] w-full bg-slate-900 border-none shadow-2xl">
                <CardHeader className="flex flex-col gap-1 p-8 text-center">
                    <h1 className="text-4xl font-extrabold tracking-tight text-white mb-2">
                        Welcome to <span className="text-primary-500">Credithos</span>
                    </h1>
                    <p className="text-white/60 text-lg">Let's get you set up for the best experience.</p>
                </CardHeader>
                <Divider className="bg-white/10" />
                <CardBody className="p-8 gap-8">
                    <div className="space-y-4">
                        <div className="p-4 bg-white/5 rounded-xl border border-white/10">
                            <h3 className="text-white font-semibold mb-2">Desktop Access</h3>
                            <p className="text-white/50 text-sm mb-4">
                                Access Credithos quickly from your desktop. We can create a shortcut for you right now.
                            </p>
                            <Button
                                onPress={handleCreateShortcut}
                                isLoading={isCreating}
                                variant="flat"
                                color="primary"
                                fullWidth
                            >
                                Create Desktop Shortcut
                            </Button>
                            {status.type && (
                                <p className={`mt-2 text-xs text-center ${status.type === 'success' ? 'text-emerald-400' : 'text-rose-400'}`}>
                                    {status.message}
                                </p>
                            )}
                        </div>

                        <div className="p-4 bg-white/5 rounded-xl border border-white/10">
                            <h3 className="text-white font-semibold mb-2">One-Time Setup</h3>
                            <p className="text-white/50 text-sm mb-4">
                                This setup wizard won't appear again. You're almost ready to start decisioning.
                            </p>
                            <Button
                                onPress={onFinish}
                                color="primary"
                                fullWidth
                                size="lg"
                                className="font-bold shadow-lg shadow-primary-500/20"
                            >
                                Finish Setup
                            </Button>
                        </div>
                    </div>
                </CardBody>
            </Card>
        </div>
    );
};

export default Setup;
