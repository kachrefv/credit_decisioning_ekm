import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Button, Chip, Skeleton } from "@heroui/react";
import Card from '../components/Card';
import client from '../api/client';

interface HomeProps {
    onNavigate: (view: any) => void;
}

export default function Home({ onNavigate }: HomeProps) {
    const [loading, setLoading] = useState(true);
    const [data, setData] = useState<any>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [statusRes, healthRes, riskRes] = await Promise.all([
                    client.get('/status'),
                    client.get('/health'),
                    client.get('/status/risk-factors')
                ]);
                setData({
                    status: statusRes.data,
                    health: healthRes.data,
                    riskFactors: riskRes.data
                });
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    return (
        <div className="flex flex-col gap-6 max-w-6xl mx-auto p-4 animate-in fade-in duration-500">
            <motion.div
                className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <div>
                    <h1 className="text-4xl font-black bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">Credithos Dashboard</h1>
                    <p className="text-slate-400 mt-2">System reliability and risk intelligence overview.</p>
                </div>
                <div className="flex gap-3">
                    <motion.button
                        onClick={() => onNavigate('evaluate')}
                        className="px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl font-bold text-white shadow-lg shadow-blue-500/20 hover:shadow-blue-500/30 transition-all duration-300 hover:scale-[1.02]"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                    >
                        New Evaluation
                    </motion.button>
                    <motion.button
                        onClick={() => onNavigate('train')}
                        className="px-6 py-3 bg-slate-700 rounded-xl font-bold text-slate-200 shadow hover:bg-slate-600 transition-all duration-300"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                    >
                        Train Model
                    </motion.button>
                </div>
            </motion.div>

            {/* Quick Stats Row (Modern Cards) */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.1 }}
                >
                    <Card gradient={true}>
                        <Card.Body className="p-5 flex flex-col gap-1">
                            <div className="flex items-center gap-2">
                                <div className="p-2 bg-blue-500/10 rounded-lg">
                                    <div className="w-5 h-5 bg-blue-400 rounded-sm rotate-45"></div>
                                </div>
                                <span className="text-xs font-bold text-blue-300 uppercase tracking-wider">Engine Mode</span>
                            </div>
                            <span className="text-2xl font-black mt-2 text-white">{loading ? "..." : (data?.status?.mode || 'Episodic')}</span>
                            <span className="text-xs text-slate-400 mt-1">Current operational mode</span>
                        </Card.Body>
                    </Card>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.2 }}
                >
                    <Card gradient={true}>
                        <Card.Body className="p-5 flex flex-col gap-1">
                            <div className="flex items-center gap-2">
                                <div className="p-2 bg-green-500/10 rounded-lg">
                                    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-green-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                                        <polyline points="22 4 12 14.01 9 11.01" />
                                    </svg>
                                </div>
                                <span className="text-xs font-bold text-green-300 uppercase tracking-wider">System Health</span>
                            </div>
                            <span className="text-2xl font-black mt-2 text-white">Healthy</span>
                            <span className="text-xs text-slate-400 mt-1">All systems operational</span>
                        </Card.Body>
                    </Card>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.3 }}
                >
                    <Card gradient={true}>
                        <Card.Body className="p-5 flex flex-col gap-1">
                            <div className="flex items-center gap-2">
                                <div className="p-2 bg-yellow-500/10 rounded-lg">
                                    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-yellow-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                                        <path d="M9.5 9.5L12 12l2.5-2.5" />
                                    </svg>
                                </div>
                                <span className="text-xs font-bold text-yellow-300 uppercase tracking-wider">Risk Nodes</span>
                            </div>
                            <span className="text-2xl font-black mt-2 text-white">{loading ? "..." : (data?.status?.risk_factors || 0)}</span>
                            <span className="text-xs text-slate-400 mt-1">Active risk factors</span>
                        </Card.Body>
                    </Card>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.4 }}
                >
                    <Card gradient={true}>
                        <Card.Body className="p-5 flex flex-col gap-1">
                            <div className="flex items-center gap-2">
                                <div className="p-2 bg-purple-500/10 rounded-lg">
                                    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-purple-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <polygon points="12 2 2 7 12 12 22 7 12 2" />
                                        <path d="M2 17l10 5 10-5" />
                                        <path d="M2 12l10 5 10-5" />
                                    </svg>
                                </div>
                                <span className="text-xs font-bold text-purple-300 uppercase tracking-wider">Version</span>
                            </div>
                            <span className="text-2xl font-black mt-2 text-white">{loading ? "..." : (data?.health?.version || '1.0.0')}</span>
                            <span className="text-xs text-slate-400 mt-1">Current build</span>
                        </Card.Body>
                    </Card>
                </motion.div>
            </div>

            {/* Main Status Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Risk Distribution */}
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 0.5 }}
                >
                    <Card gradient={true}>
                        <Card.Header className="p-6 pb-4 flex flex-col sm:flex-row sm:items-center sm:justify-between">
                            <h3 className="text-xl font-black text-white">Risk Distribution</h3>
                            <Chip size="sm" variant="flat" color="primary" className="font-bold bg-blue-500/20 text-blue-300 border-blue-500/30">LIVE</Chip>
                        </Card.Header>
                        <Card.Body className="p-6">
                            {loading ? (
                                <Skeleton className="h-32 rounded-xl" />
                            ) : (
                                <div className="flex flex-col gap-6">
                                    <div className="flex gap-1 h-3 w-full rounded-full overflow-hidden bg-slate-700">
                                        {Object.entries(data?.riskFactors?.analytics?.level_distribution || {}).map(([level, count]: [string, any]) => {
                                            const total = data?.riskFactors?.analytics?.total_count || 1;
                                            const percentage = ((count as number) / total) * 100;
                                            const colors: any = {
                                                critical: "bg-red-500",
                                                high: "bg-orange-500",
                                                medium: "bg-yellow-500",
                                                low: "bg-green-500"
                                            };
                                            return percentage > 0 ? (
                                                <div
                                                    key={level}
                                                    style={{ width: `${percentage}%` }}
                                                    className={colors[level] || 'bg-slate-600'}
                                                    title={`${level}: ${count}`}
                                                />
                                            ) : null;
                                        })}
                                    </div>
                                    <div className="grid grid-cols-2 gap-4">
                                        {['critical', 'high', 'medium', 'low'].map((level) => {
                                            const colorClasses = {
                                                critical: 'bg-red-500/20 border-red-500/30 text-red-300',
                                                high: 'bg-orange-500/20 border-orange-500/30 text-orange-300',
                                                medium: 'bg-yellow-500/20 border-yellow-500/30 text-yellow-300',
                                                low: 'bg-green-500/20 border-green-500/30 text-green-300'
                                            };

                                            return (
                                                <div key={level} className="flex items-center gap-3 p-3 rounded-lg border bg-slate-800/50 border-slate-700">
                                                    <span className={`w-3 h-3 rounded-full ${
                                                        level === 'critical' ? 'bg-red-500' :
                                                        level === 'high' ? 'bg-orange-500' :
                                                        level === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                                                    }`} />
                                                    <div className="flex-1">
                                                        <span className="text-sm font-bold capitalize text-white">{level}</span>
                                                        <span className="block text-xs text-slate-400">{data?.riskFactors?.analytics?.level_distribution[level] || 0} nodes</span>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}
                        </Card.Body>
                    </Card>
                </motion.div>

                {/* Recent Signals */}
                <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 0.6 }}
                >
                    <Card gradient={true}>
                        <Card.Header className="p-6 pb-4 flex flex-col sm:flex-row sm:items-center sm:justify-between">
                            <h3 className="text-xl font-black text-white">Recent Signals</h3>
                            <Chip variant="dot" color="success" size="sm" className="border-none bg-green-500/20 text-green-300">Syncing</Chip>
                        </Card.Header>
                        <Card.Body className="p-0">
                            <div className="max-h-[300px] overflow-y-auto">
                                <table className="w-full text-left border-collapse">
                                    <tbody>
                                        {loading ? (
                                            [1, 2, 3].map(i => (
                                                <tr key={i} className="border-b border-slate-700/50">
                                                    <td className="p-4">
                                                        <Skeleton className="h-4 w-full rounded" />
                                                    </td>
                                                </tr>
                                            ))
                                        ) : (
                                            (data?.riskFactors?.factors || []).slice(0, 5).map((rf: any) => {
                                                const levelColors = {
                                                    critical: 'bg-red-500/20 border-red-500/30 text-red-300',
                                                    high: 'bg-orange-500/20 border-orange-500/30 text-orange-300',
                                                    medium: 'bg-yellow-500/20 border-yellow-500/30 text-yellow-300',
                                                    low: 'bg-green-500/20 border-green-500/30 text-green-300'
                                                };

                                                return (
                                                    <tr key={rf.id} className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors">
                                                        <td className="p-4">
                                                            <div className="flex justify-between items-start">
                                                                <div className="flex flex-col gap-1">
                                                                    <span className="text-sm font-medium text-white">{rf.risk_factor.replace(/_/g, ' ')}</span>
                                                                    <span className="text-xs text-slate-400 font-mono">{rf.id}</span>
                                                                </div>
                                                                <Chip
                                                                    size="sm"
                                                                    variant="flat"
                                                                    className={`font-bold text-[10px] h-5 ${levelColors[rf.risk_level as keyof typeof levelColors] || 'bg-slate-700/50 text-slate-300'}`}
                                                                >
                                                                    {rf.risk_level.toUpperCase()}
                                                                </Chip>
                                                            </div>
                                                        </td>
                                                    </tr>
                                                );
                                            })
                                        )}
                                        {(!data?.riskFactors?.factors || data?.riskFactors?.factors.length === 0) && !loading && (
                                            <tr>
                                                <td className="p-8 text-center text-slate-500">
                                                    No recent signals
                                                </td>
                                            </tr>
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </Card.Body>
                    </Card>
                </motion.div>
            </div>
        </div>
    );
}
