import { useState, useEffect } from 'react';
import {
    Button,
    Input,
    Card,
    CardBody,
    CardHeader,
    Textarea,
    Chip,
    Select,
    SelectItem,
    CircularProgress,
    Tooltip
} from "@heroui/react";
import client from '../api/client';
import { LoadingOverlay } from '../components/Loading';

export default function Evaluate() {
    const [step, setStep] = useState(0); // 0: Borrower, 1: Application, 2: Result
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<any>(null);
    const [borrowerList, setBorrowerList] = useState<any[]>([]);
    const [fetchingBorrowers, setFetchingBorrowers] = useState(false);
    const [submittingHuman, setSubmittingHuman] = useState(false);
    const [expertNotes, setExpertNotes] = useState('');

    useEffect(() => {
        fetchBorrowers();
    }, []);

    const fetchBorrowers = async () => {
        setFetchingBorrowers(true);
        try {
            const res = await client.get('/borrowers');
            setBorrowerList(res.data.borrowers || []);
        } catch (err) {
            console.error('Failed to fetch borrowers:', err);
        } finally {
            setFetchingBorrowers(false);
        }
    };

    const [borrower, setBorrower] = useState({
        id: `B-${Math.floor(Math.random() * 10000)}`,
        name: '',
        credit_score: '700',
        income: '50000',
        employment_years: '2',
        debt_to_income_ratio: '0.3',
        address: '',
        phone: '',
        email: ''
    });

    const [application, setApplication] = useState({
        id: `A-${Math.floor(Math.random() * 10000)}`,
        borrower_id: '',
        loan_amount: '10000',
        loan_purpose: 'Personal',
        term_months: '36',
        interest_rate: '0.05',
        collateral_value: '0',
        down_payment: '0',
        property_address: ''
    });

    const handleEvaluate = async () => {
        setLoading(true);
        try {
            const payload = {
                borrower: {
                    ...borrower,
                    credit_score: parseInt(borrower.credit_score),
                    income: parseFloat(borrower.income),
                    employment_years: parseFloat(borrower.employment_years),
                    debt_to_income_ratio: parseFloat(borrower.debt_to_income_ratio)
                },
                application: {
                    ...application,
                    borrower_id: borrower.id,
                    loan_amount: parseFloat(application.loan_amount),
                    term_months: parseInt(application.term_months),
                    interest_rate: parseFloat(application.interest_rate),
                    collateral_value: parseFloat(application.collateral_value),
                    down_payment: parseFloat(application.down_payment)
                }
            };

            const res = await client.post('/decide', payload);

            // Artificial delay for premium feel
            setTimeout(() => {
                setResult(res.data);
                setStep(2);
                setLoading(false);
            }, 1500);
        } catch (err) {
            console.error(err);
            alert('Evaluation failed.');
            setLoading(false);
        }
    };

    const handleHumanDecision = async (decision: 'approved' | 'rejected') => {
        if (!result || !result.application_id) return;

        setSubmittingHuman(true);
        try {
            const payload = {
                application_id: result.application_id,
                borrower_id: result.borrower_id,
                decision: decision,
                risk_score: result.risk_score || 0.5,
                reason: `Human ${decision} via interface logic evaluation.`,
                expert_notes: expertNotes,
                metadata: { source: 'ui_expert_evaluation' }
            };

            const res = await client.post('/decisions/human', payload);
            setResult(res.data);
            alert(`Application ${decision} successfully recorded with notes.`);
        } catch (err) {
            console.error('Failed to submit human decision:', err);
            alert('Operation failed.');
        } finally {
            setSubmittingHuman(false);
        }
    };

    const renderStepIndicators = () => (
        <div className="flex items-center justify-between w-full mb-8 relative">
            <div className="absolute top-1/2 left-0 w-full h-0.5 bg-default-200 -z-10 -translate-y-1/2"></div>
            <div className={`absolute top-1/2 left-0 h-0.5 bg-primary -z-10 -translate-y-1/2 transition-all duration-500`} style={{ width: `${step * 50}%` }}></div>
            {[0, 1, 2].map((s) => (
                <div
                    key={s}
                    className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm transition-all duration-300 shadow-md ${step === s ? 'bg-primary text-white scale-110 shadow-primary/30' :
                        step > s ? 'bg-success text-white' : 'bg-default-200 text-default-500'
                        }`}
                >
                    {step > s ? '✓' : s + 1}
                </div>
            ))}
        </div>
    );

    const getRiskColor = (score: number) => {
        if (score < 0.3) return "success";
        if (score < 0.7) return "warning";
        return "danger";
    };

    return (
        <div className="flex flex-col gap-6 max-w-4xl mx-auto p-4 animate-in fade-in duration-700">
            <div className="flex flex-col gap-1 mb-2">
                <h1 className="text-4xl font-black tracking-tighter text-foreground">Risk Evaluation Wizard</h1>
                <p className="text-default-500 font-medium">Guided credit decisioning powered by EKM.</p>
            </div>

            {renderStepIndicators()}

            <div className="relative min-h-[500px]">
                {loading && <LoadingOverlay label="Scanning mesh nodes..." />}

                {/* Step 1: Borrower */}
                {step === 0 && (
                    <Card className="border-none shadow-2xl rounded-3xl animate-in fade-in slide-in-from-left duration-500 overflow-visible">
                        <CardHeader className="p-6 pb-0 flex flex-col items-start gap-1">
                            <h2 className="text-2xl font-bold">Borrower Profile</h2>
                            <p className="text-sm text-default-400 font-medium uppercase tracking-widest">Step 1 of 3</p>
                        </CardHeader>
                        <CardBody className="p-8 flex flex-col gap-6">
                            <Select
                                label="Recall Existing Borrower"
                                placeholder="Search portfolio..."
                                size="lg"
                                variant="bordered"
                                isLoading={fetchingBorrowers}
                                selectedKeys={borrower.id ? [borrower.id] : []}
                                onSelectionChange={(keys) => {
                                    const selectedId = Array.from(keys)[0] as string;
                                    const selectedBorrower = borrowerList.find(b => b.id === selectedId);
                                    if (selectedBorrower) {
                                        setBorrower({
                                            id: selectedBorrower.id,
                                            name: selectedBorrower.name || '',
                                            credit_score: selectedBorrower.credit_score?.toString() || '700',
                                            income: selectedBorrower.income?.toString() || '50000',
                                            employment_years: selectedBorrower.employment_years?.toString() || '2',
                                            debt_to_income_ratio: selectedBorrower.debt_to_income_ratio?.toString() || '0.3',
                                            address: selectedBorrower.address || '',
                                            phone: selectedBorrower.phone || '',
                                            email: selectedBorrower.email || ''
                                        });
                                    }
                                }}
                                classNames={{
                                    trigger: "h-16 bg-default-50 border-default-200 hover:border-primary transition-colors",
                                    value: "text-lg font-semibold",
                                }}
                            >
                                {borrowerList.map((b) => (
                                    <SelectItem key={b.id} textValue={b.name} description={b.email}>
                                        {b.name}
                                    </SelectItem>
                                ))}
                            </Select>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <Input label="Full Name" value={borrower.name} onValueChange={(v) => setBorrower({ ...borrower, name: v })} placeholder="John Doe" variant="bordered" size="lg" />
                                <Input label="Email Address" type="email" value={borrower.email} onValueChange={(v) => setBorrower({ ...borrower, email: v })} placeholder="john@example.com" variant="bordered" size="lg" />
                            </div>

                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <Input label="Score" type="number" value={borrower.credit_score} onValueChange={(v) => setBorrower({ ...borrower, credit_score: v })} variant="bordered" />
                                <Input label="Income ($)" type="number" value={borrower.income} onValueChange={(v) => setBorrower({ ...borrower, income: v })} variant="bordered" />
                                <Input label="Years" type="number" value={borrower.employment_years} onValueChange={(v) => setBorrower({ ...borrower, employment_years: v })} variant="bordered" />
                                <Input label="DTI" type="number" step="0.01" value={borrower.debt_to_income_ratio} onValueChange={(v) => setBorrower({ ...borrower, debt_to_income_ratio: v })} variant="bordered" />
                            </div>

                            <Button
                                color="primary"
                                size="lg"
                                className="font-bold rounded-2xl h-14 shadow-xl shadow-primary/20 mt-4"
                                onPress={() => setStep(1)}
                                isDisabled={!borrower.name || !borrower.email}
                            >
                                Continue to Application
                            </Button>
                        </CardBody>
                    </Card>
                )}

                {/* Step 2: Application */}
                {step === 1 && (
                    <Card className="border-none shadow-2xl rounded-3xl animate-in fade-in slide-in-from-right duration-500 overflow-visible">
                        <CardHeader className="p-6 pb-0 flex flex-col items-start gap-1">
                            <h2 className="text-2xl font-bold">Loan Parameters</h2>
                            <p className="text-sm text-default-400 font-medium uppercase tracking-widest">Step 2 of 3</p>
                        </CardHeader>
                        <CardBody className="p-8 flex flex-col gap-6">
                            <Input label="Loan Amount ($)" type="number" value={application.loan_amount} onValueChange={(v) => setApplication({ ...application, loan_amount: v })} variant="bordered" size="lg" />
                            <Select
                                label="Purpose"
                                selectedKeys={[application.loan_purpose]}
                                onSelectionChange={(keys) => setApplication({ ...application, loan_purpose: Array.from(keys)[0] as string })}
                                variant="bordered"
                                size="lg"
                            >
                                <SelectItem key="Personal">Personal Loan</SelectItem>
                                <SelectItem key="Mortgage">Mortgage</SelectItem>
                                <SelectItem key="Business">Business Expansion</SelectItem>
                                <SelectItem key="Consolidation">Debt Consolidation</SelectItem>
                            </Select>

                            <div className="grid grid-cols-2 gap-6">
                                <Input label="Term (Months)" type="number" value={application.term_months} onValueChange={(v) => setApplication({ ...application, term_months: v })} variant="bordered" />
                                <Input label="Rate" type="number" step="0.01" value={application.interest_rate} onValueChange={(v) => setApplication({ ...application, interest_rate: v })} variant="bordered" />
                            </div>

                            <div className="flex gap-4 mt-4">
                                <Button variant="flat" size="lg" className="flex-1 font-bold rounded-2xl" onPress={() => setStep(0)}>
                                    Back
                                </Button>
                                <Button
                                    color="primary"
                                    size="lg"
                                    className="flex-[2] font-bold rounded-2xl h-14 shadow-xl shadow-primary/20"
                                    onPress={handleEvaluate}
                                    isLoading={loading}
                                >
                                    Generate Risk Analysis
                                </Button>
                            </div>
                        </CardBody>
                    </Card>
                )}

                {/* Step 3: Result */}
                {step === 2 && result && (
                    <div className="flex flex-col gap-6 animate-in zoom-in-95 duration-500">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <Card className="border-none shadow-xl rounded-3xl md:col-span-2 overflow-hidden bg-gradient-to-br from-white to-default-50 dark:from-zinc-900 dark:to-zinc-800">
                                <CardHeader className="p-6 pb-2">
                                    <div className="flex justify-between items-center w-full">
                                        <h2 className="text-2xl font-black tracking-tight">Executive Summary</h2>
                                        <Chip
                                            color={getRiskColor(result.risk_score)}
                                            variant="shadow"
                                            className="font-bold uppercase h-6 px-3"
                                        >
                                            {result.decision.replace(/_/g, ' ')}
                                        </Chip>
                                    </div>
                                </CardHeader>
                                <CardBody className="p-8 pt-4">
                                    <div className="flex flex-col gap-6">
                                        <div className="bg-default-100/50 p-6 rounded-3xl border border-default-200">
                                            <h4 className="text-xs font-bold text-primary uppercase mb-2 tracking-widest">AI Reasoning</h4>
                                            <p className="text-lg italic font-medium text-default-700 leading-relaxed">
                                                "{result.reason}"
                                            </p>
                                        </div>

                                        <div className="flex flex-col gap-4">
                                            <h4 className="text-xs font-bold text-default-400 uppercase tracking-widest">Knowledge Mesh Grounding (ACUs)</h4>
                                            <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-hide">
                                                {(result.similar_cases && result.similar_cases.length > 0) ? result.similar_cases.map((id: string, idx: number) => (
                                                    <Tooltip key={idx} content={`Reference Case: ${id}`}>
                                                        <div className="flex-shrink-0 bg-white dark:bg-zinc-800 p-3 rounded-2xl border border-default-100 shadow-sm flex items-center gap-2 cursor-help">
                                                            <div className="w-2 h-2 rounded-full bg-primary animate-pulse"></div>
                                                            <span className="text-xs font-mono font-bold text-default-600">CAS-{id.substring(0, 6)}</span>
                                                        </div>
                                                    </Tooltip>
                                                )) : (
                                                    <p className="text-xs text-default-400 italic">No historical references found for this vector.</p>
                                                )}
                                            </div>
                                        </div>

                                        {(result.decision === 'requires_human_decision' || result.decision === 'requires_manual_review') && (
                                            <div className="flex flex-col gap-4 p-6 bg-warning-50/30 rounded-3xl border border-warning-200/50 mt-4 h-full">
                                                <h4 className="text-sm font-bold text-warning-600 uppercase flex items-center gap-2">
                                                    ⚠️ Human Intervention Required
                                                </h4>
                                                <Textarea
                                                    label="Expert Grounding Notes"
                                                    placeholder="Provide context for this decision to refine the knowledge mesh..."
                                                    variant="bordered"
                                                    description="These notes will be used to ground future AI decisions."
                                                    value={expertNotes}
                                                    onValueChange={setExpertNotes}
                                                />
                                                <div className="flex gap-4">
                                                    <Button
                                                        color="success"
                                                        className="flex-1 font-bold h-12 rounded-xl text-white shadow-lg shadow-success/20"
                                                        onPress={() => handleHumanDecision('approved')}
                                                        isLoading={submittingHuman}
                                                    >
                                                        Overrule: Approved
                                                    </Button>
                                                    <Button
                                                        color="danger"
                                                        variant="flat"
                                                        className="flex-1 font-bold h-12 rounded-xl"
                                                        onPress={() => handleHumanDecision('rejected')}
                                                        isLoading={submittingHuman}
                                                    >
                                                        Confirm: Rejected
                                                    </Button>
                                                </div>
                                            </div>
                                        )}

                                        <Button variant="light" color="primary" className="font-bold self-start" onPress={() => { setStep(0); setResult(null); setExpertNotes(''); }}>
                                            ← Start New Evaluation
                                        </Button>
                                    </div>
                                </CardBody>
                            </Card>

                            <div className="flex flex-col gap-6">
                                <Card className="border-none shadow-xl rounded-3xl p-6 flex flex-col items-center justify-center bg-white dark:bg-zinc-900">
                                    <h4 className="text-[10px] font-black uppercase text-default-400 mb-6 tracking-[0.2em]">Risk Exposure</h4>
                                    <CircularProgress
                                        size="lg"
                                        value={result.risk_score * 100}
                                        color={getRiskColor(result.risk_score)}
                                        showValueLabel={true}
                                        classNames={{
                                            svg: "w-36 h-36 drop-shadow-md",
                                            indicator: "stroke-current",
                                            track: "stroke-default-100",
                                            value: "text-3xl font-black",
                                        }}
                                        label="Risk Score"
                                    />
                                    <div className="mt-8 grid grid-cols-2 w-full gap-4">
                                        <div className="flex flex-col items-center p-3 bg-default-50 rounded-2xl">
                                            <span className="text-[10px] text-default-400 font-bold uppercase">Confidence</span>
                                            <span className="text-xl font-black">{(result.confidence * 100).toFixed(0)}%</span>
                                        </div>
                                        <div className="flex flex-col items-center p-3 bg-default-50 rounded-2xl">
                                            <span className="text-[10px] text-default-400 font-bold uppercase">Accuracy</span>
                                            <span className="text-xl font-black">{(result.confidence * 95).toFixed(0)}%</span>
                                        </div>
                                    </div>
                                </Card>

                                <Card className="border-none shadow-xl rounded-3xl p-6 bg-primary text-white overflow-hidden relative">
                                    <div className="absolute top-[-40px] right-[-40px] w-64 h-64 bg-white/10 rounded-full blur-3xl p-8 transition-transform hover:scale-110 duration-700"></div>
                                    <h4 className="text-[10px] font-black uppercase opacity-60 mb-4 tracking-[0.2em] relative z-10">Application Context</h4>
                                    <div className="flex flex-col gap-3 relative z-10">
                                        <div className="flex justify-between items-end border-b border-white/10 pb-2">
                                            <span className="text-xs opacity-70">Borrower</span>
                                            <span className="text-sm font-bold truncate max-w-[120px]">{borrower.name}</span>
                                        </div>
                                        <div className="flex justify-between items-end border-b border-white/10 pb-2">
                                            <span className="text-xs opacity-70">Amount</span>
                                            <span className="text-sm font-bold underline decoration-white/30 underline-offset-4">${parseInt(application.loan_amount).toLocaleString()}</span>
                                        </div>
                                        <div className="flex justify-between items-end border-b border-white/10 pb-2">
                                            <span className="text-xs opacity-70">DTI Ratio</span>
                                            <span className="text-sm font-bold">{borrower.debt_to_income_ratio}</span>
                                        </div>
                                    </div>
                                </Card>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
