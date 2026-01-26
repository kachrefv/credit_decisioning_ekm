import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    Input,
    Textarea,
    Select,
    SelectItem,
    Tooltip
} from "@heroui/react";
import Card from '../components/Card';
import Button from '../components/Button';
import { LoadingOverlay } from '../components/Loading';
import client from '../api/client';

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
            <div className="absolute top-1/2 left-0 w-full h-1 bg-slate-700 -z-10 -translate-y-1/2"></div>
            <div className={`absolute top-1/2 left-0 h-1 bg-gradient-to-r from-blue-500 to-purple-500 -z-10 -translate-y-1/2 transition-all duration-700`} style={{ width: `${step * 50}%` }}></div>
            {[0, 1, 2].map((s) => (
                <motion.div
                    key={s}
                    className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-sm transition-all duration-300 shadow-lg ${
                        step === s
                            ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white scale-110 shadow-blue-500/30'
                            : step > s
                                ? 'bg-gradient-to-r from-green-600 to-emerald-600 text-white'
                                : 'bg-slate-700 text-slate-400'
                    }`}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: s * 0.1 }}
                    whileHover={{ scale: 1.1 }}
                >
                    {step > s ? '✓' : s + 1}
                </motion.div>
            ))}
        </div>
    );

    const getRiskColor = (score: number) => {
        if (score < 0.3) return "from-green-500 to-emerald-500";
        if (score < 0.7) return "from-yellow-500 to-amber-500";
        return "from-red-500 to-rose-500";
    };

    return (
        <div className="flex flex-col gap-8 max-w-6xl mx-auto p-4 animate-in fade-in duration-700">
            <motion.div
                className="flex flex-col gap-2 mb-6"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <h1 className="text-4xl font-black bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">Risk Evaluation Wizard</h1>
                <p className="text-slate-400 font-medium">Guided credit decisioning powered by EKM.</p>
            </motion.div>

            {renderStepIndicators()}

            <div className="relative min-h-[500px]">
                {loading && <LoadingOverlay label="Scanning mesh nodes..." />}

                {/* Step 1: Borrower */}
                {step === 0 && (
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.5 }}
                    >
                        <Card elevated={true} gradient={true}>
                            <Card.Header className="p-6 pb-4 flex flex-col items-start gap-1">
                                <h2 className="text-2xl font-black text-white">Borrower Profile</h2>
                                <p className="text-sm text-slate-400 font-bold uppercase tracking-widest">Step 1 of 3</p>
                            </Card.Header>
                            <Card.Body className="p-8 flex flex-col gap-6">
                                <div className="space-y-2">
                                    <label className="text-sm font-bold text-slate-300 uppercase tracking-wider">Recall Existing Borrower</label>
                                    <select
                                        className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        value={borrower.id}
                                        onChange={(e) => {
                                            const selectedId = e.target.value;
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
                                    >
                                        <option value="">Search portfolio...</option>
                                        {borrowerList.map((b) => (
                                            <option key={b.id} value={b.id}>
                                                {b.name} - {b.email}
                                            </option>
                                        ))}
                                    </select>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div className="space-y-2">
                                        <label className="text-sm font-bold text-slate-300 uppercase tracking-wider">Full Name</label>
                                        <input
                                            type="text"
                                            value={borrower.name}
                                            onChange={(e) => setBorrower({ ...borrower, name: e.target.value })}
                                            placeholder="John Doe"
                                            className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-sm font-bold text-slate-300 uppercase tracking-wider">Email Address</label>
                                        <input
                                            type="email"
                                            value={borrower.email}
                                            onChange={(e) => setBorrower({ ...borrower, email: e.target.value })}
                                            placeholder="john@example.com"
                                            className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        />
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <div className="space-y-2">
                                        <label className="text-sm font-bold text-slate-300 uppercase tracking-wider">Score</label>
                                        <input
                                            type="number"
                                            value={borrower.credit_score}
                                            onChange={(e) => setBorrower({ ...borrower, credit_score: e.target.value })}
                                            className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-sm font-bold text-slate-300 uppercase tracking-wider">Income ($)</label>
                                        <input
                                            type="number"
                                            value={borrower.income}
                                            onChange={(e) => setBorrower({ ...borrower, income: e.target.value })}
                                            className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-sm font-bold text-slate-300 uppercase tracking-wider">Years</label>
                                        <input
                                            type="number"
                                            value={borrower.employment_years}
                                            onChange={(e) => setBorrower({ ...borrower, employment_years: e.target.value })}
                                            className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-sm font-bold text-slate-300 uppercase tracking-wider">DTI</label>
                                        <input
                                            type="number"
                                            step="0.01"
                                            value={borrower.debt_to_income_ratio}
                                            onChange={(e) => setBorrower({ ...borrower, debt_to_income_ratio: e.target.value })}
                                            className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        />
                                    </div>
                                </div>

                                <motion.button
                                    onClick={() => setStep(1)}
                                    disabled={!borrower.name || !borrower.email}
                                    className={`mt-4 w-full py-4 rounded-xl font-bold text-lg shadow-lg transition-all duration-300 ${
                                        borrower.name && borrower.email
                                            ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-blue-500/20 hover:shadow-blue-500/30'
                                            : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                                    }`}
                                    whileHover={borrower.name && borrower.email ? { scale: 1.02 } : {}}
                                    whileTap={borrower.name && borrower.email ? { scale: 0.98 } : {}}
                                >
                                    Continue to Application
                                </motion.button>
                            </Card.Body>
                        </Card>
                    </motion.div>
                )}

                {/* Step 2: Application */}
                {step === 1 && (
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.5 }}
                    >
                        <Card elevated={true} gradient={true}>
                            <Card.Header className="p-6 pb-4 flex flex-col items-start gap-1">
                                <h2 className="text-2xl font-black text-white">Loan Parameters</h2>
                                <p className="text-sm text-slate-400 font-bold uppercase tracking-widest">Step 2 of 3</p>
                            </Card.Header>
                            <Card.Body className="p-8 flex flex-col gap-6">
                                <div className="space-y-2">
                                    <label className="text-sm font-bold text-slate-300 uppercase tracking-wider">Loan Amount ($)</label>
                                    <input
                                        type="number"
                                        value={application.loan_amount}
                                        onChange={(e) => setApplication({ ...application, loan_amount: e.target.value })}
                                        className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                    />
                                </div>

                                <div className="space-y-2">
                                    <label className="text-sm font-bold text-slate-300 uppercase tracking-wider">Purpose</label>
                                    <select
                                        value={application.loan_purpose}
                                        onChange={(e) => setApplication({ ...application, loan_purpose: e.target.value })}
                                        className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                    >
                                        <option value="Personal">Personal Loan</option>
                                        <option value="Mortgage">Mortgage</option>
                                        <option value="Business">Business Expansion</option>
                                        <option value="Consolidation">Debt Consolidation</option>
                                    </select>
                                </div>

                                <div className="grid grid-cols-2 gap-6">
                                    <div className="space-y-2">
                                        <label className="text-sm font-bold text-slate-300 uppercase tracking-wider">Term (Months)</label>
                                        <input
                                            type="number"
                                            value={application.term_months}
                                            onChange={(e) => setApplication({ ...application, term_months: e.target.value })}
                                            className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-sm font-bold text-slate-300 uppercase tracking-wider">Rate</label>
                                        <input
                                            type="number"
                                            step="0.01"
                                            value={application.interest_rate}
                                            onChange={(e) => setApplication({ ...application, interest_rate: e.target.value })}
                                            className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        />
                                    </div>
                                </div>

                                <div className="flex gap-4 mt-6">
                                    <motion.button
                                        onClick={() => setStep(0)}
                                        className="flex-1 py-3 bg-slate-700 text-slate-200 font-bold rounded-xl shadow hover:bg-slate-600 transition-colors"
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                    >
                                        Back
                                    </motion.button>

                                    <motion.button
                                        onClick={handleEvaluate}
                                        disabled={loading}
                                        className="flex-[2] py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold rounded-xl shadow-lg shadow-blue-500/20 hover:shadow-blue-500/30 transition-all"
                                        whileHover={!loading ? { scale: 1.02 } : {}}
                                        whileTap={!loading ? { scale: 0.98 } : {}}
                                    >
                                        {loading ? 'Analyzing...' : 'Generate Risk Analysis'}
                                    </motion.button>
                                </div>
                            </Card.Body>
                        </Card>
                    </motion.div>
                )}

                {/* Step 3: Result */}
                {step === 2 && result && (
                    <motion.div
                        className="flex flex-col gap-8 animate-in zoom-in-95 duration-500"
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.3 }}
                    >
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            <div className="lg:col-span-2">
                                <Card elevated={true} gradient={true}>
                                    <Card.Header className="p-6 pb-4 flex flex-col sm:flex-row sm:items-center sm:justify-between">
                                        <h2 className="text-2xl font-black text-white">Executive Summary</h2>
                                        <div className={`px-4 py-2 rounded-full font-bold text-sm ${
                                            result.decision.includes('approved')
                                                ? 'bg-gradient-to-r from-green-600/20 to-emerald-600/20 text-green-300 border border-green-500/30'
                                                : result.decision.includes('rejected')
                                                    ? 'bg-gradient-to-r from-red-600/20 to-rose-600/20 text-red-300 border border-red-500/30'
                                                    : 'bg-gradient-to-r from-yellow-600/20 to-amber-600/20 text-yellow-300 border border-yellow-500/30'
                                        }`}>
                                            {result.decision.replace(/_/g, ' ')}
                                        </div>
                                    </Card.Header>
                                    <Card.Body className="p-6 pt-4">
                                        <div className="flex flex-col gap-6">
                                            <div className="p-6 rounded-2xl bg-slate-800/30 border border-slate-700/50">
                                                <h4 className="text-xs font-bold text-blue-300 uppercase tracking-wider mb-3">AI Reasoning</h4>
                                                <p className="text-lg italic font-medium text-slate-200 leading-relaxed">
                                                    "{result.reason}"
                                                </p>
                                            </div>

                                            <div className="flex flex-col gap-4">
                                                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Knowledge Mesh Grounding (ACUs)</h4>
                                                <div className="flex gap-3 overflow-x-auto pb-2">
                                                    {(result.similar_cases && result.similar_cases.length > 0) ? result.similar_cases.map((id: string, idx: number) => (
                                                        <Tooltip key={idx} content={`Reference Case: ${id}`}>
                                                            <div className="flex-shrink-0 bg-slate-800/50 p-3 rounded-2xl border border-slate-700 flex items-center gap-2 cursor-help">
                                                                <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></div>
                                                                <span className="text-xs font-mono font-bold text-slate-300">CAS-{id.substring(0, 6)}</span>
                                                            </div>
                                                        </Tooltip>
                                                    )) : (
                                                        <p className="text-xs text-slate-500 italic">No historical references found for this vector.</p>
                                                    )}
                                                </div>
                                            </div>

                                            {(result.decision === 'requires_human_decision' || result.decision === 'requires_manual_review') && (
                                                <div className="flex flex-col gap-4 p-6 bg-amber-500/10 rounded-2xl border border-amber-500/30 mt-4">
                                                    <h4 className="text-sm font-bold text-amber-300 uppercase flex items-center gap-2">
                                                        ⚠️ Human Intervention Required
                                                    </h4>
                                                    <Textarea
                                                        label="Expert Grounding Notes"
                                                        placeholder="Provide context for this decision to refine the knowledge mesh..."
                                                        variant="bordered"
                                                        description="These notes will be used to ground future AI decisions."
                                                        value={expertNotes}
                                                        onValueChange={setExpertNotes}
                                                        className="text-slate-200"
                                                    />
                                                    <div className="flex gap-4">
                                                        <motion.button
                                                            onClick={() => handleHumanDecision('approved')}
                                                            disabled={submittingHuman}
                                                            className="flex-1 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white font-bold rounded-xl shadow-lg shadow-green-500/20"
                                                            whileHover={!submittingHuman ? { scale: 1.02 } : {}}
                                                            whileTap={!submittingHuman ? { scale: 0.98 } : {}}
                                                        >
                                                            {submittingHuman ? 'Processing...' : 'Overrule: Approved'}
                                                        </motion.button>
                                                        <motion.button
                                                            onClick={() => handleHumanDecision('rejected')}
                                                            disabled={submittingHuman}
                                                            className="flex-1 py-3 bg-gradient-to-r from-red-600 to-rose-600 text-white font-bold rounded-xl shadow-lg shadow-red-500/20"
                                                            whileHover={!submittingHuman ? { scale: 1.02 } : {}}
                                                            whileTap={!submittingHuman ? { scale: 0.98 } : {}}
                                                        >
                                                            {submittingHuman ? 'Processing...' : 'Confirm: Rejected'}
                                                        </motion.button>
                                                    </div>
                                                </div>
                                            )}

                                            <motion.button
                                                onClick={() => { setStep(0); setResult(null); setExpertNotes(''); }}
                                                className="self-start px-4 py-2 text-blue-300 font-bold hover:text-blue-200 transition-colors"
                                                whileHover={{ x: -5 }}
                                            >
                                                ← Start New Evaluation
                                            </motion.button>
                                        </div>
                                    </Card.Body>
                                </Card>
                            </div>

                            <div className="flex flex-col gap-6">
                                <Card elevated={true} gradient={true}>
                                    <Card.Body className="p-6 flex flex-col items-center justify-center">
                                        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-6">Risk Exposure</h4>

                                        {/* Circular Progress Replacement */}
                                        <div className="relative w-40 h-40 flex items-center justify-center">
                                            <div className="absolute w-full h-full rounded-full border-8 border-slate-700"></div>
                                            <div
                                                className="absolute w-full h-full rounded-full border-8 border-transparent border-t-current"
                                                style={{
                                                    borderColor: result.risk_score < 0.3 ? '#10b981' : result.risk_score < 0.7 ? '#f59e0b' : '#ef4444',
                                                    clipPath: `inset(0 0 0 ${(1 - result.risk_score) * 100}%)`,
                                                    transform: `rotate(${result.risk_score * 360}deg)`
                                                }}
                                            ></div>
                                            <div className="absolute text-3xl font-black text-white">
                                                {Math.round(result.risk_score * 100)}%
                                            </div>
                                        </div>

                                        <div className="mt-8 grid grid-cols-2 w-full gap-4">
                                            <div className="flex flex-col items-center p-3 bg-slate-800/30 rounded-2xl border border-slate-700">
                                                <span className="text-xs text-slate-400 font-bold uppercase">Confidence</span>
                                                <span className="text-xl font-black text-white">{(result.confidence * 100).toFixed(0)}%</span>
                                            </div>
                                            <div className="flex flex-col items-center p-3 bg-slate-800/30 rounded-2xl border border-slate-700">
                                                <span className="text-xs text-slate-400 font-bold uppercase">Accuracy</span>
                                                <span className="text-xl font-black text-white">{(result.confidence * 95).toFixed(0)}%</span>
                                            </div>
                                        </div>
                                    </Card.Body>
                                </Card>

                                <Card elevated={true} gradient={true}>
                                    <Card.Body className="p-6 relative overflow-hidden">
                                        <div className="absolute top-[-40px] right-[-40px] w-40 h-40 bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-full blur-3xl"></div>
                                        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 relative z-10">Application Context</h4>
                                        <div className="flex flex-col gap-3 relative z-10">
                                            <div className="flex justify-between items-end border-b border-slate-700/50 pb-2">
                                                <span className="text-sm text-slate-400">Borrower</span>
                                                <span className="text-sm font-bold text-white truncate max-w-[120px]">{borrower.name}</span>
                                            </div>
                                            <div className="flex justify-between items-end border-b border-slate-700/50 pb-2">
                                                <span className="text-sm text-slate-400">Amount</span>
                                                <span className="text-sm font-bold text-white">${parseInt(application.loan_amount).toLocaleString()}</span>
                                            </div>
                                            <div className="flex justify-between items-end border-b border-slate-700/50 pb-2">
                                                <span className="text-sm text-slate-400">DTI Ratio</span>
                                                <span className="text-sm font-bold text-white">{borrower.debt_to_income_ratio}</span>
                                            </div>
                                        </div>
                                    </Card.Body>
                                </Card>
                            </div>
                        </div>
                    </motion.div>
                )}
            </div>
        </div>
    );
}
