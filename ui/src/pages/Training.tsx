import { useState, useRef } from 'react';
import {
    Button,
    Card,
    CardBody,
    CardHeader,
    Progress,
    Chip,
    Spinner,
    Select,
    SelectItem,
    Table,
    TableHeader,
    TableColumn,
    TableBody,
    TableRow,
    TableCell,
    Slider
} from "@heroui/react";
import Papa from 'papaparse';
import client from '../api/client';
import { LoadingOverlay } from '../components/Loading';

type IngestionStep = 'SELECT' | 'MAP' | 'PROGRESS';

const REQUIRED_FIELDS = [
    { key: 'name', label: 'Borrower Name', required: true },
    { key: 'email', label: 'Email Address', required: true },
    { key: 'income', label: 'Annual Income', required: true },
    { key: 'credit_score', label: 'Credit Score', required: true },
    { key: 'decision', label: 'Final Decision', required: true },
    { key: 'reason', label: 'Decision Reason', required: true }
];

const OPTIONAL_FIELDS = [
    { key: 'employment_years', label: 'Employment Years' },
    { key: 'debt_to_income_ratio', label: 'DTI Ratio' },
    { key: 'loan_amount', label: 'Loan Amount' },
    { key: 'loan_purpose', label: 'Loan Purpose' },
    { key: 'term_months', label: 'Term (Months)' },
    { key: 'interest_rate', label: 'Interest Rate' },
    { key: 'expert_notes', label: 'Expert Notes' }
];

export default function Training() {
    const [step, setStep] = useState<IngestionStep>('SELECT');
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState<any>(null);
    const [logs, setLogs] = useState<string[]>(["[SYSTEM] Kernel initialized.", "[SYSTEM] Awaiting ingestion signal..."]);
    const [csvFile, setCsvFile] = useState<File | null>(null);
    const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
    const [mapping, setMapping] = useState<Record<string, string>>({});
    const [consolidating, setConsolidating] = useState(false);
    const [consolidationResult, setConsolidationResult] = useState<any>(null);
    const [threshold, setThreshold] = useState(0.92);

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

    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setCsvFile(file);

        // Parse headers
        Papa.parse(file, {
            preview: 1,
            complete: (results) => {
                if (results.data && results.data.length > 0) {
                    const headers = results.data[0] as string[];
                    setCsvHeaders(headers);

                    // Attempt auto-mapping
                    const initialMapping: Record<string, string> = {};
                    [...REQUIRED_FIELDS, ...OPTIONAL_FIELDS].forEach(field => {
                        const match = headers.find(h => h.toLowerCase() === field.key.toLowerCase() || h.toLowerCase().includes(field.key.toLowerCase()));
                        if (match) initialMapping[field.key] = match;
                    });
                    setMapping(initialMapping);
                    setStep('MAP');
                }
            }
        });
    };

    const handleIngest = async () => {
        if (!csvFile) return;

        setStep('PROGRESS');
        setLoading(true);
        setStatus(null);
        setProgress(0);
        setLogs(["[SYSTEM] Connection established.", `[INGEST] Mapping confirmed. Starting ingestion...`]);

        const formData = new FormData();
        formData.append('file', csvFile);
        formData.append('mapping', JSON.stringify(mapping));

        try {
            setProgress(30);
            addLog("Uploading mapped dataset to MESH core...");

            const res = await client.post('/train/bulk', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 1));
                    setProgress(Math.min(percentCompleted * 0.5, 50));
                }
            });

            setProgress(80);
            addLog("Vectorizing nodes and rebuilding topology...");

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

    const updateMapping = (field: string, csvHeader: string) => {
        setMapping(prev => ({ ...prev, [field]: csvHeader }));
    };

    const isMappingValid = REQUIRED_FIELDS.every(f => mapping[f.key]);

    const handleConsolidate = async () => {
        setConsolidating(true);
        setConsolidationResult(null);
        addLog("Initiating memory consolidation...");
        try {
            const res = await client.post(`/consolidate?merge_threshold=${threshold}`);
            setConsolidationResult(res.data);
            addLog(`Consolidation complete: ${res.data.original_count} → ${res.data.consolidated_count} factors (${res.data.merged_count} merged).`);
        } catch (err: any) {
            addLog(`!! CONSOLIDATION FAILED: ${err.response?.data?.detail || err.message}`);
        } finally {
            setConsolidating(false);
        }
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
                <Card className="lg:col-span-8 border-none shadow-2xl rounded-[3rem] bg-white dark:bg-zinc-950 overflow-hidden relative min-h-[500px]">
                    {step === 'SELECT' && (
                        <div className="p-10 flex flex-col h-full animate-in fade-in duration-500">
                            <div className="mb-10">
                                <h3 className="text-2xl font-black tracking-tight">Bulk Ingestion</h3>
                                <p className="text-sm text-default-400">Step 1: Upload your historical datasets.</p>
                            </div>

                            <div
                                className="group relative flex-1 border-2 border-dashed border-default-200 rounded-[2.5rem] flex flex-col items-center justify-center gap-4 transition-all duration-500 cursor-pointer hover:border-primary hover:bg-primary/5"
                                onClick={triggerFilePicker}
                            >
                                <input
                                    type="file"
                                    ref={fileInputRef}
                                    className="hidden"
                                    accept=".csv"
                                    onChange={handleFileSelect}
                                />
                                <div className="w-20 h-20 rounded-[2rem] bg-default-100 flex items-center justify-center group-hover:scale-110 group-hover:bg-primary transition-all duration-500">
                                    <span className="text-3xl font-black group-hover:text-white transition-colors">+</span>
                                </div>
                                <div className="text-center">
                                    <p className="font-black text-xl tracking-tight">Select CSV Dataset</p>
                                    <p className="text-sm text-default-400 font-medium">Click here or drag files into this sector</p>
                                </div>
                            </div>
                        </div>
                    )}

                    {step === 'MAP' && (
                        <div className="p-10 flex flex-col h-full animate-in slide-in-from-right-10 duration-500">
                            <div className="mb-10 flex justify-between items-start">
                                <div>
                                    <h3 className="text-2xl font-black tracking-tight">Field Mapping</h3>
                                    <p className="text-sm text-default-400">Step 2: Align CSV columns with system requirements.</p>
                                </div>
                                <Button variant="flat" size="sm" className="rounded-xl font-bold" onPress={() => setStep('SELECT')}>Change File</Button>
                            </div>

                            <div className="flex-1 overflow-y-auto max-h-[600px] pr-2 custom-scrollbar">
                                <Table aria-label="Field Mapping Table" removeWrapper className="mb-6">
                                    <TableHeader>
                                        <TableColumn className="bg-default-50 font-black uppercase text-[10px] tracking-widest">System Field</TableColumn>
                                        <TableColumn className="bg-default-50 font-black uppercase text-[10px] tracking-widest">CSV Column Match</TableColumn>
                                    </TableHeader>
                                    <TableBody>
                                        {[...REQUIRED_FIELDS, ...OPTIONAL_FIELDS].map((field) => (
                                            <TableRow key={field.key} className="border-b border-default-100">
                                                <TableCell>
                                                    <div className="flex flex-col">
                                                        <span className="font-bold text-sm">{field.label}</span>
                                                        {'required' in field && <span className="text-[10px] text-danger font-black uppercase">Required</span>}
                                                    </div>
                                                </TableCell>
                                                <TableCell>
                                                    <Select
                                                        placeholder="Select column"
                                                        aria-label={`Match ${field.label}`}
                                                        selectedKeys={mapping[field.key] ? [mapping[field.key]] : []}
                                                        onSelectionChange={(keys) => updateMapping(field.key, Array.from(keys)[0] as string)}
                                                        variant="bordered"
                                                        size="sm"
                                                        classNames={{
                                                            trigger: "rounded-xl border-default-200",
                                                        }}
                                                    >
                                                        {csvHeaders.map((header) => (
                                                            <SelectItem key={header} textValue={header}>{header}</SelectItem>
                                                        ))}
                                                    </Select>
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </div>

                            <div className="pt-6 border-t border-default-100 flex justify-end gap-3 bg-white/80 backdrop-blur-md sticky bottom-0 z-20">
                                <Button
                                    color="primary"
                                    size="lg"
                                    className="font-black px-12 rounded-2xl shadow-xl shadow-primary/30"
                                    isDisabled={!isMappingValid}
                                    onPress={handleIngest}
                                >
                                    Confirm & Ingest
                                </Button>
                            </div>
                        </div>
                    )}

                    {step === 'PROGRESS' && (
                        <div className="p-10 flex flex-col h-full items-center justify-center text-center animate-in zoom-in-95 duration-500">
                            {loading && <LoadingOverlay label="Synchronizing Knowledge Mesh..." />}

                            {!status ? (
                                <div className="space-y-8 w-full max-w-md">
                                    <div className="w-24 h-24 bg-primary/10 rounded-[2.5rem] flex items-center justify-center mx-auto mb-6">
                                        <Spinner size="lg" color="primary" />
                                    </div>
                                    <div className="space-y-2">
                                        <h4 className="text-2xl font-black">{progress}% Syncing</h4>
                                        <p className="text-default-400 font-medium">Injecting data into the Episodic Knowledge Mesh core.</p>
                                    </div>
                                    <Progress
                                        value={progress}
                                        className="h-3"
                                        color="primary"
                                        isStriped
                                        classNames={{
                                            indicator: "bg-gradient-to-r from-primary to-indigo-600",
                                            track: "bg-default-100"
                                        }}
                                    />
                                </div>
                            ) : (
                                <div className="space-y-8 animate-in zoom-in-95 duration-500">
                                    <div className="w-24 h-24 bg-success rounded-[2.5rem] flex items-center justify-center mx-auto shadow-2xl shadow-success/40 scale-110">
                                        <span className="text-white text-4xl font-black">✓</span>
                                    </div>
                                    <div className="space-y-2">
                                        <h4 className="text-3xl font-black text-success-900 tracking-tight">Mesh Synchronized</h4>
                                        <p className="text-default-500 font-medium">Historical precedent successfully integrated.</p>
                                    </div>
                                    <div className="flex justify-center gap-4">
                                        <Chip variant="flat" color="success" className="font-black h-10 px-6 border-none text-lg">
                                            {status.trained_models} Entities
                                        </Chip>
                                        <Chip variant="flat" color="success" className="font-black h-10 px-6 border-none text-lg">
                                            {status.training_duration.toFixed(1)}s Logged
                                        </Chip>
                                    </div>
                                    <Button
                                        color="success"
                                        size="lg"
                                        className="font-black px-12 h-14 rounded-2xl shadow-xl shadow-success/20 mt-4"
                                        onPress={() => { setStep('SELECT'); setStatus(null); setProgress(0); setLogs([]); }}
                                    >
                                        Start New Cycle
                                    </Button>
                                </div>
                            )}
                        </div>
                    )}
                </Card>

                <div className="lg:col-span-4 flex flex-col gap-6">
                    <Card className="border-none shadow-xl rounded-[2.5rem] bg-zinc-900 text-white overflow-hidden">
                        <CardHeader className="p-8 pb-4 flex items-center gap-3">
                            <div className="w-2 h-2 rounded-full bg-primary animate-pulse shadow-[0_0_8px_primary]"></div>
                            <span className="text-[10px] font-black uppercase tracking-widest opacity-60">Session Logs</span>
                        </CardHeader>
                        <CardBody className="p-8 pt-0">
                            <div className="bg-black/40 rounded-2xl p-6 font-mono text-[11px] leading-relaxed overflow-hidden">
                                {logs.map((log, i) => (
                                    <div key={i} className={`mb-1.5 ${log.includes('!!') ? 'text-red-400' : 'text-primary'}`}>
                                        <span className="opacity-40">{log.split('] ')[0]}]</span> {log.split('] ')[1]}
                                    </div>
                                ))}
                                {loading && (
                                    <div className="text-white animate-pulse">_</div>
                                )}
                            </div>
                        </CardBody>
                    </Card>

                    <Card className="border-none shadow-sm rounded-[2.5rem] bg-default-50 p-8 border border-default-100 h-full">
                        <h4 className="text-[10px] font-black uppercase tracking-[0.2em] mb-6 text-default-400">Mesh Protocol</h4>
                        <div className="space-y-6">
                            {[
                                { title: "Step 1: Selection", desc: "Binary dataset identification and core validation." },
                                { title: "Step 2: Mapping", desc: "Conceptual alignment of external dimensions." },
                                { title: "Step 3: Ingestion", desc: "Topological re-linking and vector persistence." }
                            ].map((s, i) => (
                                <div key={i} className="flex gap-4">
                                    <div className={`w-8 h-8 rounded-xl flex items-center justify-center font-black text-xs shrink-0 ${(i === 0 && step === 'SELECT') || (i === 1 && step === 'MAP') || (i === 2 && step === 'PROGRESS')
                                        ? 'bg-primary text-white' : 'bg-default-200 text-default-400'
                                        }`}>
                                        {i + 1}
                                    </div>
                                    <div className="flex flex-col">
                                        <span className="text-sm font-black">{s.title}</span>
                                        <span className="text-xs text-default-400 leading-tight">{s.desc}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </Card>

                    <Card className="border-none shadow-xl rounded-[2.5rem] bg-gradient-to-br from-indigo-900 to-purple-900 text-white overflow-hidden">
                        <CardHeader className="p-8 pb-4">
                            <span className="text-[10px] font-black uppercase tracking-widest opacity-60">Memory Optimization</span>
                        </CardHeader>
                        <CardBody className="p-8 pt-0">
                            <p className="text-sm opacity-80 mb-6">Merge similar risk factors to reduce noise and improve retrieval quality.</p>

                            <div className="mb-8 space-y-4">
                                <div className="flex justify-between items-center text-xs font-bold uppercase tracking-widest opacity-60">
                                    <span>Merge Threshold</span>
                                    <span>{(threshold * 100).toFixed(0)}%</span>
                                </div>
                                <Slider
                                    step={0.01}
                                    maxValue={1}
                                    minValue={0.5}
                                    value={threshold}
                                    onChange={(val) => setThreshold(val as number)}
                                    className="max-w-md"
                                    color="secondary"
                                    size="sm"
                                />
                                <p className="text-[10px] opacity-40 italic">Higher values are more conservative; lower values merge more aggressively.</p>
                            </div>

                            <Button
                                color="secondary"
                                size="lg"
                                className="w-full font-black rounded-2xl h-14"
                                isLoading={consolidating}
                                onPress={handleConsolidate}
                            >
                                {consolidating ? "Consolidating..." : "Consolidate Memory"}
                            </Button>
                            {consolidationResult && (
                                <div className="mt-6 bg-black/30 rounded-2xl p-4 space-y-2 text-sm">
                                    <div className="flex justify-between"><span className="opacity-60">Original:</span><span className="font-bold">{consolidationResult.original_count}</span></div>
                                    <div className="flex justify-between"><span className="opacity-60">After:</span><span className="font-bold text-green-400">{consolidationResult.consolidated_count}</span></div>
                                    <div className="flex justify-between"><span className="opacity-60">Merged:</span><span className="font-bold text-yellow-400">{consolidationResult.merged_count}</span></div>
                                    <div className="flex justify-between"><span className="opacity-60">Duration:</span><span className="font-bold">{consolidationResult.duration_seconds}s</span></div>
                                </div>
                            )}
                        </CardBody>
                    </Card>
                </div>
            </div>
        </div>
    );
}
