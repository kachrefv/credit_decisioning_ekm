import { useState } from 'react';
import { Button, Input, Card, CardBody } from "@heroui/react";
import client from '../api/client';

interface RegisterProps {
    onNavigateLogin: () => void;
}

export default function Register({ onNavigateLogin }: RegisterProps) {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleRegister = async () => {
        setLoading(true);
        setError('');
        try {
            await client.post('/auth/register', { name, email, password });
            onNavigateLogin();
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Registration failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex w-full h-screen bg-slate-950 overflow-hidden font-sans">
            {/* Left side - Branding/Image */}
            <div className="hidden md:flex w-1/2 p-8 relative items-center justify-center overflow-hidden border-r border-white/5">
                {/* Master Gradient Background */}
                <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900"></div>

                {/* Decorative Elements */}
                <div className="absolute top-0 left-0 w-full h-full">
                    <div className="absolute top-[-10%] left-[-10%] w-[60%] h-[60%] bg-blue-500/10 rounded-full blur-[120px] animate-pulse"></div>
                    <div className="absolute bottom-[-20%] right-[-10%] w-[70%] h-[70%] bg-blue-500/10 rounded-full blur-[100px]"></div>
                </div>

                {/* Content */}
                <div className="z-10 text-center text-white max-w-lg">
                    <div className="mb-8 inline-flex items-center justify-center p-4 bg-white/5 backdrop-blur-3xl rounded-[2rem] border border-white/10 shadow-2xl">
                        <svg className="w-12 h-12 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
                        </svg>
                    </div>
                    <h1 className="text-6xl font-black mb-6 tracking-tight bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">Credithos</h1>
                    <div className="inline-block px-5 py-2 bg-blue-500/10 backdrop-blur-xl border border-blue-500/20 rounded-full text-[11px] font-bold tracking-[0.3em] mb-8 uppercase text-blue-300 shadow-xl">
                        Powered by EKM
                    </div>
                    <p className="text-2xl font-medium text-white/90 mb-4">Join the Network</p>
                    <p className="text-lg text-slate-400 leading-relaxed font-light">
                        Create an account to access the most advanced credit decisioning platform in Europe.
                    </p>
                </div>
            </div>

            {/* Right side - Register Form */}
            <div className="flex w-full md:w-1/2 bg-slate-950 flex-col items-center justify-center p-6 relative">
                {/* Subtle Ambient Light */}
                <div className="absolute top-1/4 -right-20 w-80 h-80 bg-blue-600/5 rounded-full blur-[120px]"></div>
                <div className="absolute bottom-1/4 -left-20 w-80 h-80 bg-blue-600/5 rounded-full blur-[120px]"></div>

                <div className="w-full max-w-sm z-10">
                    <div className="text-center mb-10">
                        <h2 className="text-4xl font-bold text-white mb-2">Create Account</h2>
                        <p className="text-slate-400 text-medium">Start your journey with Credithos</p>
                    </div>

                    <Card className="border-none bg-slate-900/40 backdrop-blur-2xl shadow-2xl rounded-[2.5rem] p-4 border border-white/5">
                        <CardBody className="gap-6 py-8 px-6">
                            <Input
                                label="Full Name"
                                placeholder="Achref Riahi"
                                value={name}
                                onValueChange={setName}
                                variant="bordered"
                                labelPlacement="outside"
                                classNames={{
                                    label: "text-slate-200 font-medium pb-1.5",
                                    input: "text-white text-medium pt-1",
                                    inputWrapper: "h-14 bg-slate-800/50 border-white/5 hover:border-white/10 group-data-[focus=true]:border-blue-500 group-data-[focus=true]:bg-slate-800 transition-all rounded-2xl",
                                    innerWrapper: "pb-1"
                                }}
                            />

                            <Input
                                label="Email Address"
                                placeholder="name@company.com"
                                type="email"
                                value={email}
                                onValueChange={setEmail}
                                variant="bordered"
                                labelPlacement="outside"
                                classNames={{
                                    label: "text-slate-200 font-medium pb-1.5",
                                    input: "text-white text-medium pt-1",
                                    inputWrapper: "h-14 bg-slate-800/50 border-white/5 hover:border-white/10 group-data-[focus=true]:border-blue-500 group-data-[focus=true]:bg-slate-800 transition-all rounded-2xl",
                                    innerWrapper: "pb-1"
                                }}
                            />

                            <Input
                                label="Password"
                                placeholder="••••••••"
                                type="password"
                                value={password}
                                onValueChange={setPassword}
                                variant="bordered"
                                labelPlacement="outside"
                                description={<span className="text-[10px] text-slate-500">Minimum 8 characters with upper/lower case</span>}
                                classNames={{
                                    label: "text-slate-200 font-medium pb-1.5",
                                    input: "text-white text-medium pt-1",
                                    inputWrapper: "h-14 bg-slate-800/50 border-white/5 hover:border-white/10 group-data-[focus=true]:border-blue-500 group-data-[focus=true]:bg-slate-800 transition-all rounded-2xl",
                                    innerWrapper: "pb-1"
                                }}
                            />

                            {error && (
                                <div className="flex items-center gap-3 p-4 bg-danger-500/10 border border-danger-500/20 text-danger-400 text-sm rounded-2xl animate-appearance-in">
                                    <svg className="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    {error}
                                </div>
                            )}

                            <Button
                                onPress={handleRegister}
                                isLoading={loading}
                                className="h-14 w-full bg-gradient-to-r from-blue-600 to-blue-800 text-white font-bold text-lg rounded-2xl shadow-xl shadow-blue-600/20 hover:shadow-blue-600/40 hover:scale-[1.02] active:scale-[0.98] transition-all"
                            >
                                Sign Up
                            </Button>

                            <div className="flex flex-col items-center gap-4 mt-2">
                                <p className="text-slate-400 text-sm">
                                    Already have an account?{' '}
                                    <button
                                        onClick={onNavigateLogin}
                                        className="text-blue-400 font-bold hover:text-blue-300 transition-colors"
                                    >
                                        Log in here
                                    </button>
                                </p>
                            </div>
                        </CardBody>
                    </Card>
                </div>

                <div className="absolute bottom-8 text-slate-500/30 text-[10px] uppercase tracking-widest font-bold">
                    © 2026 Credithos — All Rights Reserved
                </div>
            </div>
        </div>
    );
}
