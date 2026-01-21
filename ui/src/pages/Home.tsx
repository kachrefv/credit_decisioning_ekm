import { useState, useEffect } from 'react';
import {
    Card,
    CardBody,
    Chip,
    Skeleton,
    Button
} from "@heroui/react";
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
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">Credithos Dashboard</h1>
                    <p className="text-default-500">System reliability and risk intelligence overview.</p>
                </div>
                <div className="flex gap-2">
                    <Button size="sm" variant="flat" onPress={() => onNavigate('evaluate')}>Scale Evaluation</Button>
                    <Button size="sm" variant="flat" onPress={() => onNavigate('train')}>Train Model</Button>
                </div>
            </div>

            {/* Quick Stats Row (Tiny Infos) */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card className="border-none shadow-sm bg-primary-50 dark:bg-primary-900/10">
                    <CardBody className="p-4 flex flex-col gap-1">
                        <span className="text-tiny text-primary font-bold uppercase">Engine Mode</span>
                        <span className="text-lg font-bold capitalize">{loading ? "..." : (data?.status?.mode || 'Episodic')}</span>
                    </CardBody>
                </Card>
                <Card className="border-none shadow-sm bg-success-50 dark:bg-success-900/10">
                    <CardBody className="p-4 flex flex-col gap-1">
                        <span className="text-tiny text-success font-bold uppercase">System Health</span>
                        <span className="text-lg font-bold">Healthy</span>
                    </CardBody>
                </Card>
                <Card className="border-none shadow-sm">
                    <CardBody className="p-4 flex flex-col gap-1">
                        <span className="text-tiny text-default-500 font-bold uppercase">Risk Nodes</span>
                        <span className="text-lg font-bold">{loading ? "..." : (data?.status?.risk_factors || 0)}</span>
                    </CardBody>
                </Card>
                <Card className="border-none shadow-sm">
                    <CardBody className="p-4 flex flex-col gap-1">
                        <span className="text-tiny text-default-500 font-bold uppercase">Version</span>
                        <span className="text-lg font-bold">{loading ? "..." : (data?.health?.version || '1.0.0')}</span>
                    </CardBody>
                </Card>
            </div>

            {/* Main Status Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Risk Distribution */}
                <Card className="border-none shadow-sm rounded-2xl">
                    <CardBody className="p-6">
                        <div className="flex justify-between items-center mb-6">
                            <h3 className="text-lg font-bold">Risk Distribution</h3>
                            <Chip size="sm" variant="flat" color="primary" className="font-bold">LIVE</Chip>
                        </div>
                        {loading ? (
                            <Skeleton className="h-40 rounded-lg" />
                        ) : (
                            <div className="flex flex-col gap-6">
                                <div className="flex gap-1 h-4 w-full rounded-full overflow-hidden bg-default-100">
                                    {Object.entries(data?.riskFactors?.analytics?.level_distribution || {}).map(([level, count]: [string, any]) => {
                                        const total = data?.riskFactors?.analytics?.total_count || 1;
                                        const percentage = ((count as number) / total) * 100;
                                        const colors: any = {
                                            critical: "bg-danger",
                                            high: "bg-warning",
                                            medium: "bg-primary",
                                            low: "bg-success"
                                        };
                                        return percentage > 0 ? (
                                            <div
                                                key={level}
                                                style={{ width: `${percentage}%` }}
                                                className={colors[level] || 'bg-default-300'}
                                                title={`${level}: ${count}`}
                                            />
                                        ) : null;
                                    })}
                                </div>
                                <div className="grid grid-cols-2 gap-4">
                                    {['critical', 'high', 'medium', 'low'].map((level) => (
                                        <div key={level} className="flex items-center gap-2">
                                            <span className={`w-2 h-2 rounded-full ${level === 'critical' ? 'bg-danger' :
                                                level === 'high' ? 'bg-warning' :
                                                    level === 'medium' ? 'bg-primary' : 'bg-success'
                                                }`} />
                                            <div className="flex flex-col">
                                                <span className="text-xs font-medium capitalize">{level}</span>
                                                <span className="text-tiny text-default-400">{data?.riskFactors?.analytics?.level_distribution[level] || 0} nodes</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </CardBody>
                </Card>

                {/* Recent Nodes */}
                <Card className="border-none shadow-sm rounded-2xl overflow-hidden">
                    <CardBody className="p-0">
                        <div className="p-6 border-b border-default-100 flex justify-between items-center">
                            <h3 className="text-lg font-bold">Recent Signals</h3>
                            <Chip variant="dot" color="success" size="sm" className="border-none">Syncing</Chip>
                        </div>
                        <div className="max-h-[300px] overflow-y-auto">
                            <table className="w-full text-left border-collapse">
                                <tbody>
                                    {loading ? (
                                        [1, 2, 3].map(i => (
                                            <tr key={i} className="border-b border-default-50">
                                                <td className="p-4"><Skeleton className="h-4 w-full rounded" /></td>
                                            </tr>
                                        ))
                                    ) : (
                                        (data?.riskFactors?.factors || []).slice(0, 5).map((rf: any) => (
                                            <tr key={rf.id} className="border-b border-default-50 hover:bg-default-50/50">
                                                <td className="p-4">
                                                    <div className="flex justify-between items-start">
                                                        <div className="flex flex-col gap-1">
                                                            <span className="text-sm font-medium">{rf.risk_factor.replace(/_/g, ' ')}</span>
                                                            <span className="text-tiny text-default-400 font-mono">{rf.id}</span>
                                                        </div>
                                                        <Chip
                                                            size="sm"
                                                            variant="flat"
                                                            color={
                                                                rf.risk_level === 'critical' ? 'danger' :
                                                                    rf.risk_level === 'high' ? 'warning' :
                                                                        rf.risk_level === 'medium' ? 'primary' : 'success'
                                                            }
                                                            className="font-bold text-[10px] h-5"
                                                        >
                                                            {rf.risk_level.toUpperCase()}
                                                        </Chip>
                                                    </div>
                                                </td>
                                            </tr>
                                        ))
                                    )}
                                    {(!data?.riskFactors?.factors || data?.riskFactors?.factors.length === 0) && !loading && (
                                        <tr>
                                            <td className="p-8 text-center text-default-300">
                                                No recent signals
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </CardBody>
                </Card>
            </div>
        </div>
    );
}
