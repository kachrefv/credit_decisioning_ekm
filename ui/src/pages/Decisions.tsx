import { useState, useEffect } from 'react';
import {
    Table,
    TableHeader,
    TableColumn,
    TableBody,
    TableRow,
    TableCell,
    Chip,
    Button,
    Card,
    CardBody,
    Spinner,
    Modal,
    ModalContent,
    ModalHeader,
    ModalBody,
    ModalFooter,
    useDisclosure,
    CircularProgress,
    Tooltip,
    Divider
} from "@heroui/react";
import client from '../api/client';

export default function Decisions() {
    const [decisions, setDecisions] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedDecision, setSelectedDecision] = useState<any>(null);
    const { isOpen, onOpen, onOpenChange } = useDisclosure();

    const fetchDecisions = async () => {
        setLoading(true);
        try {
            const res = await client.get('/decisions');
            // Sort by latest first
            const sorted = (res.data.decisions || []).sort((a: any, b: any) =>
                new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
            );
            setDecisions(sorted);
        } catch (err) {
            console.error('Failed to fetch decisions:', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchDecisions();
    }, []);

    const columns = [
        { name: "APPLICATION", uid: "application_id" },
        { name: "DECISION", uid: "decision" },
        { name: "RISK", uid: "risk_score" },
        { name: "GROUNDING", uid: "grounding" },
        { name: "REASON", uid: "reason" },
        { name: "TIMESTAMP", uid: "timestamp" }
    ];

    const getDecisionColor = (decision: string) => {
        const d = (decision || '').toLowerCase();
        if (d === 'approved') return 'success';
        if (d.includes('human') || d.includes('review') || d.includes('manual')) return 'warning';
        return 'danger';
    };

    const getRiskColor = (score: number) => {
        if (score < 0.3) return "success";
        if (score < 0.7) return "warning";
        return "danger";
    };

    const handleRowClick = (decision: any) => {
        setSelectedDecision(decision);
        onOpen();
    };

    const renderCell = (decision: any, columnKey: React.Key) => {
        const cellValue = decision[columnKey as keyof any];

        switch (columnKey) {
            case "application_id":
                return (
                    <div className="flex flex-col">
                        <span className="text-sm font-bold text-primary">APP-{(cellValue || '').substring(0, 8)}</span>
                        <span className="text-[10px] text-default-400 font-mono italic">ID: {cellValue}</span>
                    </div>
                );
            case "decision":
                return (
                    <div className="flex items-center gap-2">
                        <Chip
                            className="capitalize font-bold"
                            color={getDecisionColor(cellValue)}
                            size="sm"
                            variant="shadow"
                        >
                            {(cellValue || '').replace(/_/g, ' ')}
                        </Chip>
                        {decision.expert_notes && (
                            <Tooltip content="Has Expert Grounding Notes">
                                <span className="text-warning-500 text-lg">📝</span>
                            </Tooltip>
                        )}
                    </div>
                );
            case "risk_score":
                return (
                    <div className="flex flex-col gap-1 w-20">
                        <div className="flex justify-between items-center px-1">
                            <span className="text-[10px] font-bold text-default-400">{(cellValue * 100).toFixed(0)}%</span>
                        </div>
                        <div className="w-full h-1.5 bg-default-100 rounded-full overflow-hidden">
                            <div
                                className={`h-full bg-${getRiskColor(cellValue)} transition-all duration-1000`}
                                style={{ width: `${cellValue * 100}%` }}
                            ></div>
                        </div>
                    </div>
                );
            case "grounding":
                const acuCount = decision.similar_cases?.length || 0;
                return (
                    <Tooltip content={`${acuCount} Similar Cases used for grounding`}>
                        <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full border ${acuCount > 0 ? 'bg-primary-50 border-primary-100 text-primary-600' : 'bg-default-50 border-default-100 text-default-400'}`}>
                            <div className={`w-1.5 h-1.5 rounded-full ${acuCount > 0 ? 'bg-primary animate-pulse' : 'bg-default-300'}`}></div>
                            <span className="text-[10px] font-black">{acuCount} ACUs</span>
                        </div>
                    </Tooltip>
                );
            case "timestamp":
                return (
                    <div className="text-tiny text-default-400 font-medium">
                        {new Date(cellValue).toLocaleString()}
                    </div>
                );
            case "reason":
                return (
                    <p className="text-xs text-default-500 max-w-[250px] italic font-medium truncate" title={cellValue}>
                        "{cellValue}"
                    </p>
                );
            default:
                return cellValue;
        }
    };

    return (
        <div className="flex flex-col gap-6 max-w-6xl mx-auto p-4 animate-in fade-in duration-700">
            <div className="flex justify-between items-end mb-4">
                <div className="flex flex-col gap-1">
                    <h1 className="text-4xl font-black tracking-tighter text-foreground">Decision Ledger</h1>
                    <p className="text-default-500 font-medium">Audit trail of system logic and expert interventions.</p>
                </div>
                <Button
                    color="primary"
                    variant="flat"
                    isLoading={loading}
                    onPress={fetchDecisions}
                    className="font-bold rounded-2xl px-6 h-12 shadow-sm"
                >
                    Refresh Audit
                </Button>
            </div>

            <Card className="border-none shadow-2xl rounded-[2.5rem] bg-white dark:bg-zinc-950 overflow-hidden">
                <CardBody className="p-0">
                    <Table
                        aria-label="Decisions history table"
                        selectionMode="single"
                        onRowAction={(key) => handleRowClick(decisions.find(d => d.id === key))}
                        classNames={{
                            base: "bg-transparent",
                            table: "min-h-[400px]",
                            thead: "[&>tr]:first:rounded-none bg-default-50/50 text-default-400 font-black text-[10px] uppercase tracking-[0.2em] h-14",
                            tr: "cursor-pointer hover:bg-default-50/80 transition-all border-b border-default-100 last:border-none group",
                            td: "py-5 group-hover:px-6 transition-all"
                        }}
                    >
                        <TableHeader columns={columns}>
                            {(column) => (
                                <TableColumn
                                    key={column.uid}
                                    align="start"
                                >
                                    {column.name}
                                </TableColumn>
                            )}
                        </TableHeader>
                        <TableBody
                            items={decisions}
                            loadingContent={<Spinner size="lg" color="primary" />}
                            loadingState={loading ? "loading" : "idle"}
                            emptyContent={"No decisions found in the mesh."}
                        >
                            {(item) => (
                                <TableRow key={item.id}>
                                    {(columnKey) => <TableCell>{renderCell(item, columnKey)}</TableCell>}
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </CardBody>
            </Card>

            {/* Decision Details Modal */}
            <Modal
                isOpen={isOpen}
                onOpenChange={onOpenChange}
                size="3xl"
                backdrop="blur"
                placement="center"
                scrollBehavior="inside"
                classNames={{
                    base: "bg-white dark:bg-zinc-900 rounded-[2.5rem] border-none",
                    header: "p-8 pb-0",
                    body: "p-8 pt-6",
                    footer: "p-8 pt-0 border-none"
                }}
            >
                <ModalContent>
                    {(onClose) => (
                        <>
                            <ModalHeader className="flex flex-col gap-1">
                                <div className="flex items-center gap-3">
                                    <div className="p-3 bg-primary/10 rounded-2xl text-primary font-black text-xs">
                                        DEC-CORE
                                    </div>
                                    <h2 className="text-2xl font-black tracking-tight">Decision Audit</h2>
                                </div>
                                <p className="text-tiny text-default-400 font-mono tracking-widest">{selectedDecision?.id}</p>
                            </ModalHeader>
                            <ModalBody>
                                {selectedDecision && (
                                    <div className="grid grid-cols-1 md:grid-cols-12 gap-8">
                                        <div className="md:col-span-8 flex flex-col gap-6">
                                            <div className="bg-default-50 p-6 rounded-[2rem] border border-default-100">
                                                <h4 className="text-[10px] font-black text-primary uppercase mb-3 tracking-widest">Grounded Reasoning</h4>
                                                <p className="text-lg italic font-medium text-default-700 leading-relaxed">
                                                    "{selectedDecision.reason}"
                                                </p>
                                            </div>

                                            {selectedDecision.expert_notes && (
                                                <div className="bg-warning-50/30 p-6 rounded-[2rem] border border-warning-200">
                                                    <h4 className="text-[10px] font-black text-warning-600 uppercase mb-3 tracking-widest">Expert Intervention Notes</h4>
                                                    <p className="text-sm font-medium text-warning-700 leading-relaxed italic">
                                                        {selectedDecision.expert_notes}
                                                    </p>
                                                </div>
                                            )}

                                            <div className="flex flex-col gap-4">
                                                <h4 className="text-[10px] font-black text-default-400 uppercase tracking-widest">Mesh Context References (ACUs)</h4>
                                                <div className="flex flex-wrap gap-3">
                                                    {(selectedDecision.similar_cases && selectedDecision.similar_cases.length > 0) ? selectedDecision.similar_cases.map((id: string, idx: number) => (
                                                        <div key={idx} className="bg-white dark:bg-zinc-800 px-4 py-2 rounded-xl border border-default-100 shadow-sm flex items-center gap-2">
                                                            <div className="w-1.5 h-1.5 rounded-full bg-primary"></div>
                                                            <span className="text-[10px] font-mono font-bold text-default-600">CAS-{id.substring(0, 8)}</span>
                                                        </div>
                                                    )) : (
                                                        <p className="text-xs text-default-400 italic">No historical references were explicitly flagged for this decision.</p>
                                                    )}
                                                </div>
                                            </div>
                                        </div>

                                        <div className="md:col-span-4 flex flex-col gap-6">
                                            <Card className="bg-gradient-to-br from-primary to-primary-600 text-white p-6 rounded-[2rem] shadow-xl shadow-primary/20 relative overflow-hidden">
                                                <div className="absolute top-[-20px] right-[-20px] w-32 h-32 bg-white/10 rounded-full blur-2xl"></div>
                                                <div className="relative z-10 flex flex-col items-center gap-4">
                                                    <CircularProgress
                                                        size="lg"
                                                        value={selectedDecision.risk_score * 100}
                                                        color="default"
                                                        showValueLabel={true}
                                                        classNames={{
                                                            svg: "w-24 h-24 stroke-white/20",
                                                            indicator: "stroke-white",
                                                            value: "text-xl font-black text-white",
                                                            label: "text-[10px] text-white/70"
                                                        }}
                                                        label="Risk Score"
                                                    />
                                                    <Divider className="bg-white/10" />
                                                    <div className="w-full">
                                                        <div className="flex justify-between items-center text-[10px] uppercase opacity-60 font-black mb-1">
                                                            <span>Confidence</span>
                                                            <span>{(selectedDecision.confidence * 100).toFixed(0)}%</span>
                                                        </div>
                                                        <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                                                            <div
                                                                className="h-full bg-white transition-all duration-1000"
                                                                style={{ width: `${selectedDecision.confidence * 100}%` }}
                                                            ></div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </Card>

                                            <div className="bg-default-50 p-6 rounded-[2rem] border border-default-100 flex flex-col gap-4">
                                                <div className="flex justify-between items-center">
                                                    <span className="text-[10px] font-black text-default-400 uppercase tracking-[0.1em]">Status</span>
                                                    <Chip color={getDecisionColor(selectedDecision.decision)} size="sm" variant="shadow" className="font-bold">
                                                        {(selectedDecision.decision || '').toUpperCase().replace(/_/g, ' ')}
                                                    </Chip>
                                                </div>
                                                <div className="flex justify-between items-center">
                                                    <span className="text-[10px] font-black text-default-400 uppercase tracking-[0.1em]">Application</span>
                                                    <span className="text-xs font-bold text-primary">APP-{selectedDecision.application_id?.substring(0, 8)}</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </ModalBody>
                            <ModalFooter>
                                <Button color="default" variant="light" onPress={onClose} className="font-bold h-12 rounded-2xl">
                                    Close Audit
                                </Button>
                            </ModalFooter>
                        </>
                    )}
                </ModalContent>
            </Modal>
        </div>
    );
}
