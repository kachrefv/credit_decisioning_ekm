import { useState } from 'react';
import { Button, Input, Card, CardBody, CardHeader } from "@heroui/react";
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
        <div className="flex w-full h-screen">
            <div className="hidden md:flex w-1/2 bg-gradient-to-br from-secondary-500 to-primary-500 items-center justify-center p-8 relative overflow-hidden">
                <div className="absolute inset-0 bg-black/20 backdrop-blur-sm z-0"></div>
                <div className="z-10 text-center text-white">
                    <h1 className="text-5xl font-bold mb-4">Credithos</h1>
                    <div className="inline-block px-3 py-1 bg-white/20 backdrop-blur-md rounded-full text-[12px] font-bold tracking-widest mb-6 uppercase">Powered by EKM</div>
                    <p className="text-xl opacity-90">Join the Future of Credit</p>
                    <p className="mt-4 opacity-75 max-w-md mx-auto">Create an account to access the next generation of knowledge mesh technology.</p>
                </div>
            </div>

            <div className="flex w-full md:w-1/2 bg-background flex-col items-center justify-center p-8">
                <Card className="w-full max-w-sm border-none shadow-none bg-transparent rounded-2xl">
                    <CardHeader className="flex flex-col gap-1 pb-4">
                        <h2 className="text-2xl font-bold">Create Account</h2>
                        <p className="text-small text-default-500">Sign up specifically for Episodic Knowledge Mesh</p>
                    </CardHeader>
                    <CardBody>
                        <div className="flex flex-col gap-4">
                            <Input
                                label="Full Name"
                                placeholder="Enter your name"
                                value={name}
                                onValueChange={setName}
                                variant="bordered"
                                labelPlacement="outside"
                            />
                            <Input
                                label="Email"
                                placeholder="Enter your email"
                                type="email"
                                value={email}
                                onValueChange={setEmail}
                                variant="bordered"
                                labelPlacement="outside"
                            />
                            <Input
                                label="Password"
                                placeholder="Create a password"
                                type="password"
                                value={password}
                                onValueChange={setPassword}
                                variant="bordered"
                                labelPlacement="outside"
                                description="Must be at least 8 characters long"
                            />
                            {error && <div className="p-2 bg-danger-50 text-danger text-tiny rounded-xl text-center">{error}</div>}

                            <Button color="success" fullWidth size="lg" className="text-white font-semibold shadow-lg shadow-success/40 rounded-xl" onPress={handleRegister} isLoading={loading}>
                                Sign Up
                            </Button>

                            <div className="flex items-center gap-2 text-small text-default-500 justify-center mt-4">
                                Already have an account?
                                <span className="text-primary font-bold cursor-pointer hover:underline" onClick={onNavigateLogin}>
                                    Log In
                                </span>
                            </div>
                        </div>
                    </CardBody>
                </Card>
            </div>
        </div>
    );
}
