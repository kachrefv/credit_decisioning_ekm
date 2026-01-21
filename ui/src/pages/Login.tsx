import { useState } from 'react';
import { Button, Input, Card, CardBody, CardHeader } from "@heroui/react";
import client, { setAuthToken } from '../api/client';

interface LoginProps {
    onLoginSuccess: () => void;
    onNavigateRegister: () => void;
}

export default function Login({ onLoginSuccess, onNavigateRegister }: LoginProps) {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleLogin = async () => {
        setLoading(true);
        setError('');
        try {
            const res = await client.post('/auth/login', { email, password });
            // The backend returns { access_token, token_type, ... }
            const token = res.data.access_token;
            setAuthToken(token);
            // Optionally cache user info if returned, for now we will fetch profile later
            onLoginSuccess();
        } catch (err: any) {
            console.error(err);
            setError(err.response?.data?.detail || 'Login failed. Please check your credentials.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex w-full h-screen">
            {/* Left side - Branding/Image */}
            <div className="hidden md:flex w-1/2 bg-gradient-to-tr from-primary-500 to-secondary-500 items-center justify-center p-8 relative overflow-hidden">
                <div className="absolute inset-0 bg-black/20 backdrop-blur-sm z-0"></div>
                <div className="z-10 text-center text-white">
                    <h1 className="text-5xl font-bold mb-4">Credithos</h1>
                    <div className="inline-block px-3 py-1 bg-white/20 backdrop-blur-md rounded-full text-[12px] font-bold tracking-widest mb-6">POWERED BY EKM</div>
                    <p className="text-xl opacity-90">Decisions Redefined</p>
                    <p className="mt-4 opacity-75 max-w-md mx-auto">Advanced credit decisioning powered by episodic memory and AI.</p>
                </div>
            </div>

            {/* Right side - Login Form */}
            <div className="flex w-full md:w-1/2 bg-background flex-col items-center justify-center p-8">
                <Card className="w-full max-w-sm border-none shadow-none bg-transparent rounded-2xl">
                    <CardHeader className="flex flex-col gap-1 pb-4">
                        <h2 className="text-2xl font-bold">Welcome Back</h2>
                        <p className="text-small text-default-500">Sign in to your account to continue</p>
                    </CardHeader>
                    <CardBody>
                        <div className="flex flex-col gap-4">
                            <Input
                                autoFocus
                                label="Email"
                                placeholder="Enter your email"
                                value={email}
                                onValueChange={setEmail}
                                variant="bordered"
                                labelPlacement="outside"
                            />
                            <Input
                                label="Password"
                                placeholder="Enter your password"
                                type="password"
                                value={password}
                                onValueChange={setPassword}
                                variant="bordered"
                                labelPlacement="outside"
                            />
                            <div className="flex justify-between items-center text-tiny px-1">
                                <span className="text-default-500 cursor-pointer hover:underline">Forgot password?</span>
                            </div>

                            {error && <div className="p-2 bg-danger-50 text-danger text-tiny rounded-xl text-center">{error}</div>}

                            <Button color="primary" fullWidth size="lg" onPress={handleLogin} isLoading={loading} className="font-semibold shadow-lg shadow-primary/40 rounded-xl">
                                Log In
                            </Button>

                            <div className="flex items-center gap-2 text-small text-default-500 justify-center mt-4">
                                Need an account?
                                <span className="text-primary font-bold cursor-pointer hover:underline" onClick={onNavigateRegister}>
                                    Register
                                </span>
                            </div>
                        </div>
                    </CardBody>
                </Card>
            </div>
        </div>
    );
}
