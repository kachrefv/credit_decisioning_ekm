import { useState, useRef } from 'react';
import {
    Button,
    Card,
    CardBody,
    CardHeader,
    Divider,
    Progress,
    Chip,
    Tooltip,
    Spinner
} from "@heroui/react";
import client from '../api/client';
import { LoadingOverlay } from '../components/Loading';

export default function Training() {
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState<any>(null);
    const [logs, setLogs] = useState<string[]>(["[SYSTEM] Kernel initialized.", "[SYSTEM] Awaiting ingestion signal..."]);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const addLog = (msg: string) => {
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`].slice(-8));
    };

    const handleDownloadTemplate = () => {
        const headers = "name,email,income,employment_years,credit_score,loan_amount,loan_purpose,term_months,interest_rate,decision,reason,expert_notes\n";
        const sample = "John Doe,john.doe@example.com,75000,5,720,25000,Personal,36,7.5,approved,Strong income and credit history,Client has stable employment.\n";
        const blob = new Blob([headers + sample], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'training_template.csv';
        a.click();
    };

    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setLoading(true);
        setStatus(null);
        setProgress(0);
        setLogs(["[SYSTEM] Connection established.", `[INGEST] Loading ${file.name}...`]);

        const formData = new FormData();
        formData.append('file', file);

        try {
            setProgress(30);
            addLog("Uploading dataset to MESH core...");

            const res = await client.post('/train/bulk', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 1));
                    setProgress(Math.min(percentCompleted * 0.5, 50));
                }
            });

            setProgress(80);
            addLog("Vectorizing nodes and rebuilding topology...");

            // Artificial smoothing delay for the processing phase
            setTimeout(() => {
                setProgress(100);
                setStatus(res.data);
                addLog(`Ingestion complete: ${res.data.trained_models} clusters updated.`);
                setLoading(false);
            }, 1200);

        } catch (err: any) {
            console.error(err);
            addLog(`!! TRACE ERROR: ${err.response?.data?.detail || 'General Connection Failure'}`);
            setLoading(false);
            setProgress(0);
        }
    };

    const triggerFilePicker = () => {
        fileInputRef.current?.click();
    };

    return (
        <div className="flex flex-col gap-10 max-w-6xl mx-auto p-4 py-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
            {/* Header Section */}
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
                <div className="space-y-2">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-primary rounded-2xl flex items-center justify-center shadow-lg shadow-primary/30">
                            <span className="text-white font-black">KM</span>
                        </div>
                        <h1 className="text-4xl font-black tracking-tight">Training Dashboard</h1>
                    </div>
                    <p className="text-default-500 font-medium text-lg">Re-topology and knowledge mesh ingestion engine.</p>
                </div>

                <Button
                    variant="flat"
                    color="primary"
                    className="font-black h-12 rounded-2xl px-8 uppercase tracking-widest text-xs"
                    onPress={handleDownloadTemplate}
                >
                    Download Template
                </Button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                {/* Main Action Area */}
                <Card className="lg:col-span-7 border-none shadow-2xl rounded-[3rem] bg-white dark:bg-zinc-950 overflow-hidden relative">
                    {loading && <LoadingOverlay label="Synchronizing Knowledge Mesh..." />}

                    <CardHeader className="p-10 pb-2 flex flex-col items-start gap-1">
                        <h3 className="text-2xl font-black tracking-tight">Bulk Ingestion</h3>
                        <p className="text-sm text-default-400">Drag and drop your historical datasets to train the agent.</p>
                    </CardHeader>

                    <CardBody className="p-10 flex flex-col gap-8">
                        <div
                            className={`
                                group relative h-64 border-2 border-dashed rounded-[2.5rem] 
                                flex flex-col items-center justify-center gap-4 transition-all duration-500
                                cursor-pointer
                                ${loading ? 'border-primary/20 bg-primary/5' : 'border-default-200 hover:border-primary hover:bg-primary/5'}
                            `}
                            onPress={triggerFilePicker}
                        >
                            <input
                                type="file"
                                ref={fileInputRef}
                                className="hidden"
                                accept=".csv"
                                onChange={handleFileUpload}
                                disabled={loading}
                            />

                            <div className={`
                                w-20 h-20 rounded-[2rem] bg-default-100 flex items-center justify-center 
                                group-hover:scale-110 group-hover:bg-primary transition-all duration-500
                            `}>
                                <span className={`text-3xl font-black group-hover:text-white transition-colors`}>+</span>
                            </div>

                            <div className="text-center">
                                <p className="font-black text-xl tracking-tight">Select CSV Dataset</p>
                                <p className="text-sm text-default-400 font-medium">Click here or drag files into this sector</p>
                            </div>

                            {loading && (
                                <div className="absolute inset-0 flex items-center justify-center rounded-[2.5rem] bg-zinc-950/20 backdrop-blur-[2px]">
                                    <Spinner size="lg" color="primary" />
                                </div>
                            )}
                        </div>

                        {loading && (
                            <div className="space-y-4 animate-in fade-in duration-500">
                                <div className="flex justify-between items-end">
                                    <div className="flex flex-col gap-1">
                                        <span className="text-[10px] font-black uppercase tracking-widest text-primary">Mesh Ingestion</span>
                                        <span className="text-lg font-black">{progress}% Syncing</span>
                                    </div>
                                    <span className="text-xs font-mono text-default-400">CORE01::ACTIVE</span>
                                </div>
                                <Progress
                                    value={progress}
                                    className="h-3"
                                    color="primary"
                                    isStriped
                                    classNames={{
                                        base: "max-w-full",
                                        indicator: "bg-gradient-to-r from-primary to-indigo-600",
                                        track: "bg-default-100"
                                    }}
                                />
                            </div>
                        )}

                        {status && (
                            <div className="p-8 bg-success-50/50 border border-success-200 rounded-[2.5rem] animate-in zoom-in-95 duration-500">
                                <div className="flex items-center gap-6">
                                    <div className="w-16 h-16 bg-success rounded-full flex items-center justify-center shadow-xl shadow-success/20">
                                        <span className="text-white text-2xl font-black">✓</span>
                                    </div>
                                    <div className="flex-1">
                                        <h4 className="text-xl font-black text-success-900 tracking-tight">Sync Complete</h4>
                                        <div className="flex flex-wrap gap-2 mt-2">
                                            <Chip variant="flat" color="success" className="font-bold border-none">{status.trained_models} Vectors</Chip>
                                            <Chip variant="flat" color="success" className="font-bold border-none">{status.training_duration.toFixed(2)}s Latency</Chip>
                                        </div>
                                    </div>
                                    <Button
                                        color="success"
                                        variant="shadow"
                                        className="font-black h-12 rounded-2xl px-6"
                                        onPress={() => { setStatus(null); setProgress(0); setLogs([]); }}
                                    >
                                        New Cycle
                                    </Button>
                                </div>
                            </div>
                        )}
                    </CardBody>
                </Card>

                {/* Info & Logs Sidebar */}
                <div className="lg:col-span-5 flex flex-col gap-8">
                    {/* System Monitor */}
                    <Card className="border-none shadow-xl rounded-[2.5rem] bg-zinc-900 text-white overflow-hidden">
                        <CardHeader className="p-8 pb-4 flex items-center gap-3">
                            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
                            <span className="text-[10px] font-black uppercase tracking-widest opacity-60">System Monitor</span>
                        </CardHeader>
                        <CardBody className="p-8 pt-0">
                            <div className="bg-black/40 rounded-2xl p-6 font-mono text-xs leading-relaxed overflow-hidden">
                                {logs.map((log, i) => (
                                    <div key={i} className={`mb-1 ${log.includes('!!') ? 'text-red-400' : 'text-emerald-400/80'}`}>
                                        {log}
                                    </div>
                                ))}
                                {loading && (
                                    <div className="text-white animate-pulse">_</div>
                                )}
                            </div>
                        </CardBody>
                    </Card>

                    {/* Architecture Stats */}
                    <div className="grid grid-cols-2 gap-4">
                        <Card className="border-none shadow-lg bg-indigo-600 text-white p-6 rounded-[2rem]">
                            <span className="text-[10px] font-black uppercase opacity-60">Knowledge Density</span>
                            <div className="text-3xl font-black mt-2">High</div>
                            <Progress value={88} color="secondary" size="sm" className="mt-4" />
                        </Card>
                        <Card className="border-none shadow-lg bg-zinc-900 text-white p-6 rounded-[2rem]">
                            <span className="text-[10px] font-black uppercase opacity-60">Mesh Mode</span>
                            <div className="text-3xl font-black mt-2 text-primary">Active</div>
                            <div className="mt-4 flex gap-1">
                                {[1, 2, 3, 4, 5].map(i => <div key={i} className="w-1.5 h-6 bg-primary/20 rounded-full flex items-end"><div className="w-full bg-primary rounded-full transition-all duration-1000" style={{ height: `${Math.random() * 100}%` }}></div></div>)}
                            </div>
                        </Card>
                    </div>

                    {/* Technical Specs */}
                    <Card className="border-none shadow-sm rounded-[2rem] bg-default-50 p-8 border border-default-100">
                        <h4 className="text-[10px] font-black uppercase tracking-[0.2em] mb-4 text-default-400">Ingestion Protocol</h4>
                        <ul className="space-y-3">
                            {[
                                "Async CSV Multi-threaded Parsing",
                                "Incremental Identity Resolution",
                                "Embedding Vectorization (mesh-v4)",
                                "Qdrant Cluster Synchronization"
                            ].map((spec, i) => (
                                <li key={i} className="flex items-center gap-3 text-xs font-bold text-default-700">
                                    <div className="w-1 h-1 bg-primary rounded-full"></div>
                                    {spec}
                                </li>
                            ))}
                        </ul>
                    </Card>
                </div>
            </div>
        </div>
    );
}
