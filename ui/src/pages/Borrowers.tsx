import { useState, useEffect } from 'react';
import {
    Table,
    TableHeader,
    TableColumn,
    TableBody,
    TableRow,
    TableCell,
    User as UserComponent,
    Chip,
    Button,
    Input,
    Textarea,
    Modal,
    ModalContent,
    ModalHeader,
    ModalBody,
    ModalFooter,
    useDisclosure,
    Spinner,
    Card,
    CardBody,
    Divider
} from "@heroui/react";
import client from '../api/client';

export default function Borrowers() {
    const [borrowers, setBorrowers] = useState<any[]>([]);
    const [decisions, setDecisions] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [submitting, setSubmitting] = useState(false);
    const [selectedBorrower, setSelectedBorrower] = useState<any>(null);
    const [isEditMode, setIsEditMode] = useState(false);

    const { isOpen: isDetailsOpen, onOpen: onDetailsOpen, onOpenChange: onDetailsOpenChange } = useDisclosure();
    const { isOpen: isFormOpen, onOpen: onFormOpen, onOpenChange: onFormOpenChange, onClose: onFormClose } = useDisclosure();

    const [formState, setFormState] = useState({
        id: '',
        name: '',
        credit_score: '700',
        income: '50000',
        employment_years: '2',
        debt_to_income_ratio: '0.3',
        address: '',
        phone: '',
        email: ''
    });

    const fetchData = async () => {
        setLoading(true);
        try {
            const [bRes, dRes] = await Promise.all([
                client.get('/borrowers'),
                client.get('/decisions')
            ]);
            setBorrowers(bRes.data.borrowers || []);
            setDecisions(dRes.data.decisions || []);
        } catch (err) {
            console.error('Failed to fetch data:', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
    }, []);

    const handleViewDetails = (borrower: any) => {
        setSelectedBorrower(borrower);
        onDetailsOpen();
    };

    const handleCreateNew = () => {
        setIsEditMode(false);
        setFormState({
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
        onFormOpen();
    };

    const handleEdit = (borrower: any) => {
        setIsEditMode(true);
        setFormState({
            id: borrower.id,
            name: borrower.name,
            credit_score: borrower.credit_score.toString(),
            income: borrower.income.toString(),
            employment_years: borrower.employment_years.toString(),
            debt_to_income_ratio: borrower.debt_to_income_ratio.toString(),
            address: borrower.address || '',
            phone: borrower.phone || '',
            email: borrower.email
        });
        onFormOpen();
    };

    const handleSubmit = async () => {
        setSubmitting(true);
        try {
            const payload = {
                ...formState,
                credit_score: parseInt(formState.credit_score),
                income: parseFloat(formState.income),
                employment_years: parseFloat(formState.employment_years),
                debt_to_income_ratio: parseFloat(formState.debt_to_income_ratio)
            };

            if (isEditMode) {
                await client.put(`/borrowers/${formState.id}`, payload);
            } else {
                await client.post('/borrowers', payload);
            }

            await fetchData();
            onFormClose();
        } catch (err) {
            console.error('Failed to save borrower:', err);
            alert('Failed to save borrower.');
        } finally {
            setSubmitting(false);
        }
    };

    const borrowerDecisions = selectedBorrower
        ? decisions.filter(d => d.borrower_id === selectedBorrower.id)
        : [];

    const columns = [
        { name: "BORROWER", uid: "name" },
        { name: "CREDIT STANDING", uid: "credit_score" },
        { name: "INCOME / REVENUE", uid: "income" },
        { name: "TENURE", uid: "employment_years" },
        { name: "ACTIONS", uid: "actions" },
    ];

    const getScoreCategory = (score: number) => {
        if (score >= 750) return { label: 'Elite', color: 'success' as const };
        if (score >= 680) return { label: 'Prime', color: 'primary' as const };
        if (score >= 600) return { label: 'Sub-Prime', color: 'warning' as const };
        return { label: 'High Risk', color: 'danger' as const };
    };

    const renderCell = (borrower: any, columnKey: React.Key) => {
        const cellValue = borrower[columnKey as keyof any];

        switch (columnKey) {
            case "name":
                return (
                    <UserComponent
                        avatarProps={{ radius: "lg", src: `https://i.pravatar.cc/150?u=${borrower.id}`, className: "border-2 border-primary/20 p-0.5" }}
                        description={borrower.email}
                        name={borrower.name}
                        classNames={{
                            name: "text-sm font-black text-foreground",
                            description: "text-[10px] font-medium text-default-400"
                        }}
                    >
                        {borrower.email}
                    </UserComponent>
                );
            case "credit_score":
                const scoreInfo = getScoreCategory(parseInt(cellValue));
                return (
                    <div className="flex flex-col gap-1">
                        <span className="text-sm font-black">{cellValue}</span>
                        <Chip size="sm" variant="flat" color={scoreInfo.color} className="h-4 text-[10px] font-bold uppercase">
                            {scoreInfo.label}
                        </Chip>
                    </div>
                );
            case "income":
                return (
                    <div className="flex flex-col">
                        <span className="text-sm font-bold text-success-600">${parseInt(cellValue).toLocaleString()}</span>
                        <span className="text-[10px] font-bold text-default-400 uppercase tracking-tighter">Gross Annual</span>
                    </div>
                );
            case "employment_years":
                return (
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-black">{cellValue}Y</span>
                        <div className="w-12 h-1 bg-default-100 rounded-full overflow-hidden">
                            <div className="h-full bg-primary" style={{ width: `${Math.min(cellValue * 10, 100)}%` }}></div>
                        </div>
                    </div>
                );
            case "actions":
                return (
                    <div className="flex items-center gap-2">
                        <Button
                            isIconOnly
                            size="sm"
                            variant="flat"
                            color="primary"
                            className="bg-primary/10 hover:bg-primary/20 rounded-xl"
                            onPress={() => handleViewDetails(borrower)}
                        >
                            👁️
                        </Button>
                        <Button
                            isIconOnly
                            size="sm"
                            variant="flat"
                            color="default"
                            className="bg-default-100/50 hover:bg-default-200/50 rounded-xl"
                            onPress={() => handleEdit(borrower)}
                        >
                            📝
                        </Button>
                    </div>
                );
            default:
                return cellValue;
        }
    };

    return (
        <div className="flex flex-col gap-6 max-w-6xl mx-auto p-4 animate-in fade-in duration-700">
            <div className="flex justify-between items-end mb-4">
                <div className="flex flex-col gap-1">
                    <h1 className="text-4xl font-black tracking-tighter text-foreground">Borrower Portfolio</h1>
                    <p className="text-default-500 font-medium">Manage historical knowledge mesh nodes and credit profiles.</p>
                </div>
                <div className="flex gap-3">
                    <Button
                        color="default"
                        variant="flat"
                        isLoading={loading}
                        onPress={fetchData}
                        className="font-bold rounded-2xl h-12"
                    >
                        Refresh
                    </Button>
                    <Button
                        color="primary"
                        className="font-bold rounded-2xl h-12 shadow-xl shadow-primary/20 px-6"
                        onPress={handleCreateNew}
                    >
                        New Borrower
                    </Button>
                </div>
            </div>

            <Card className="border-none shadow-2xl rounded-[2.5rem] bg-white dark:bg-zinc-950 overflow-hidden">
                <CardBody className="p-0">
                    <Table
                        aria-label="Borrowers table"
                        selectionMode="none"
                        classNames={{
                            base: "bg-transparent",
                            thead: "[&>tr]:first:rounded-none bg-default-50/50 text-default-400 font-black text-[10px] uppercase tracking-[0.2em] h-14",
                            tr: "hover:bg-default-50/80 transition-all border-b border-default-100 last:border-none",
                            td: "py-5"
                        }}
                    >
                        <TableHeader columns={columns}>
                            {(column) => (
                                <TableColumn
                                    key={column.uid}
                                    align={column.uid === "actions" ? "center" : "start"}
                                >
                                    {column.name}
                                </TableColumn>
                            )}
                        </TableHeader>
                        <TableBody
                            items={borrowers}
                            loadingContent={<Spinner size="lg" color="primary" />}
                            loadingState={loading ? "loading" : "idle"}
                            emptyContent={"No knowledge nodes identified."}
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

            {/* Details Modal */}
            <Modal
                isOpen={isDetailsOpen}
                onOpenChange={onDetailsOpenChange}
                size="4xl"
                backdrop="blur"
                placement="center"
                scrollBehavior="inside"
                classNames={{
                    base: "bg-white dark:bg-zinc-900 rounded-[2.5rem] border-none",
                    header: "p-8 pb-0",
                    body: "p-8 pt-6 pb-12",
                    footer: "p-8 pt-0 border-none"
                }}
            >
                <ModalContent>
                    {(onClose) => (
                        <>
                            <ModalHeader className="flex flex-col gap-1">
                                <div className="flex items-center gap-3">
                                    <div className="p-3 bg-primary/10 rounded-2xl text-primary font-black text-xs uppercase tracking-widest">
                                        MESH-NODE
                                    </div>
                                    <h2 className="text-2xl font-black tracking-tight">Borrower Intelligence</h2>
                                </div>
                                <p className="text-tiny text-default-400 font-mono tracking-widest uppercase">{selectedBorrower?.id}</p>
                            </ModalHeader>
                            <ModalBody className="py-6 scrollbar-hide">
                                {selectedBorrower && (
                                    <div className="flex flex-col gap-10">
                                        <div className="grid grid-cols-1 md:grid-cols-12 gap-8">
                                            <div className="md:col-span-7 flex flex-col gap-8">
                                                <div>
                                                    <h4 className="text-[10px] font-black text-primary uppercase mb-4 tracking-[0.2em]">Profile Architecture</h4>
                                                    <div className="grid grid-cols-2 gap-y-4 gap-x-8">
                                                        <div className="flex flex-col">
                                                            <span className="text-[10px] font-bold text-default-400 uppercase">Entity Name</span>
                                                            <span className="text-lg font-black">{selectedBorrower.name}</span>
                                                        </div>
                                                        <div className="flex flex-col">
                                                            <span className="text-[10px] font-bold text-default-400 uppercase">Communication</span>
                                                            <span className="text-sm font-bold text-primary truncate">{selectedBorrower.email}</span>
                                                        </div>
                                                        <div className="flex flex-col">
                                                            <span className="text-[10px] font-bold text-default-400 uppercase">Employment Tenure</span>
                                                            <span className="text-sm font-bold">{selectedBorrower.employment_years} Years Active</span>
                                                        </div>
                                                        <div className="flex flex-col">
                                                            <span className="text-[10px] font-bold text-default-400 uppercase">Registered Address</span>
                                                            <span className="text-xs font-medium text-default-600 truncate">{selectedBorrower.address || 'Standard Validation Pending'}</span>
                                                        </div>
                                                    </div>
                                                </div>

                                                <Divider className="opacity-50" />

                                                <div>
                                                    <h4 className="text-[10px] font-black text-primary uppercase mb-4 tracking-[0.2em]">System Context</h4>
                                                    <div className="bg-default-50 p-4 rounded-2xl border border-default-100 flex flex-col gap-3">
                                                        <div className="flex justify-between items-center text-[10px] font-black uppercase tracking-widest text-default-400">
                                                            <span>Metadata Vector</span>
                                                            <span className="px-2 py-0.5 bg-primary/20 text-primary rounded-md">ID-SIGNED</span>
                                                        </div>
                                                        <pre className="text-[10px] font-mono text-default-500 overflow-auto max-h-32 leading-relaxed">
                                                            {JSON.stringify(selectedBorrower.metadata || {}, null, 2)}
                                                        </pre>
                                                    </div>
                                                </div>
                                            </div>

                                            <div className="md:col-span-5 flex flex-col gap-6">
                                                <Card className="bg-gradient-to-br from-primary-600 to-indigo-700 text-white border-none rounded-[2rem] p-8 shadow-2xl shadow-primary/30 relative overflow-hidden">
                                                    <div className="absolute top-[-40px] right-[-40px] w-48 h-48 bg-white/10 rounded-full blur-3xl transition-transform hover:scale-110 duration-1000"></div>
                                                    <h4 className="text-[10px] font-black uppercase mb-8 tracking-[0.2em] opacity-60">Financial Standing</h4>
                                                    <div className="flex flex-col gap-8">
                                                        <div className="flex flex-col">
                                                            <span className="text-[10px] uppercase font-black opacity-70 mb-1">Mesh Risk Index</span>
                                                            <div className="flex items-end gap-2">
                                                                <span className="text-5xl font-black">{selectedBorrower.credit_score}</span>
                                                                <Chip size="sm" variant="flat" className="bg-white/20 text-white font-bold h-6 mb-1">
                                                                    {getScoreCategory(selectedBorrower.credit_score).label}
                                                                </Chip>
                                                            </div>
                                                        </div>
                                                        <div className="grid grid-cols-2 gap-4">
                                                            <div className="flex flex-col">
                                                                <span className="text-[10px] uppercase font-black opacity-70 mb-1">DTI Ratio</span>
                                                                <span className="text-2xl font-black">{(selectedBorrower.debt_to_income_ratio * 100).toFixed(1)}%</span>
                                                            </div>
                                                            <div className="flex flex-col">
                                                                <span className="text-[10px] uppercase font-black opacity-70 mb-1">Revenue</span>
                                                                <span className="text-2xl font-black">${(selectedBorrower.income / 1000).toFixed(0)}k</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </Card>

                                                <div className="p-6 bg-success-50/30 rounded-[2rem] border border-success-200/50 flex flex-col gap-1">
                                                    <span className="text-[10px] font-black text-success-600 uppercase tracking-widest">Profile Stability</span>
                                                    <span className="text-xs font-medium text-success-700">Validated history suggests long-term node reliability.</span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Decision History */}
                                        <div className="flex flex-col gap-6">
                                            <div className="flex items-center gap-3">
                                                <h4 className="text-[10px] font-black text-primary uppercase tracking-[0.2em]">Decision Audit History</h4>
                                                <div className="flex-1 h-px bg-default-100"></div>
                                                <span className="text-[10px] font-black text-default-400 bg-default-100 px-3 py-1 rounded-full">{borrowerDecisions.length} EVENTS</span>
                                            </div>

                                            {borrowerDecisions.length > 0 ? (
                                                <div className="flex flex-col gap-3">
                                                    {borrowerDecisions.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()).map((d) => (
                                                        <div key={d.id} className="flex items-center justify-between p-4 bg-default-50/50 hover:bg-default-50 rounded-2xl border border-default-100 transition-all group">
                                                            <div className="flex items-center gap-4">
                                                                <div className={`w-2 h-2 rounded-full ${d.decision === 'approved' ? 'bg-success' : 'bg-danger'}`}></div>
                                                                <div className="flex flex-col">
                                                                    <span className="text-xs font-black uppercase text-foreground">
                                                                        {d.decision.replace(/_/g, ' ')}
                                                                    </span>
                                                                    <span className="text-[10px] font-medium text-default-400 truncate max-w-[400px]">
                                                                        "{d.reason}"
                                                                    </span>
                                                                </div>
                                                            </div>
                                                            <div className="flex items-center gap-6">
                                                                <div className="flex flex-col items-end">
                                                                    <span className="text-[10px] font-black text-default-500">{(d.risk_score * 100).toFixed(0)}% RISK</span>
                                                                    <span className="text-[10px] font-medium text-default-400">{new Date(d.timestamp).toLocaleDateString()}</span>
                                                                </div>
                                                                <Button isIconOnly size="sm" variant="light" className="opacity-0 group-hover:opacity-100 transition-all font-black text-primary">
                                                                    →
                                                                </Button>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            ) : (
                                                <div className="p-8 border-2 border-dashed border-default-100 rounded-[2rem] flex items-center justify-center italic text-default-400 text-sm">
                                                    No processing events found for this entity.
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </ModalBody>
                            <ModalFooter>
                                <Button color="default" variant="light" onPress={onClose} className="font-bold h-12 rounded-2xl">
                                    Close Intelligence
                                </Button>
                            </ModalFooter>
                        </>
                    )}
                </ModalContent>
            </Modal>

            {/* Create/Edit Form Modal */}
            <Modal
                isOpen={isFormOpen}
                onOpenChange={onFormOpenChange}
                size="2xl"
                backdrop="blur"
                placement="center"
                scrollBehavior="inside"
                classNames={{
                    base: "bg-white dark:bg-zinc-900 rounded-[2.5rem] border-none",
                    header: "p-8 pb-0",
                    body: "p-8 py-10",
                    footer: "p-8 pt-0 border-none"
                }}
            >
                <ModalContent>
                    {(onClose) => (
                        <>
                            <ModalHeader className="flex flex-col gap-1">
                                <p className="text-2xl font-black tracking-tight">{isEditMode ? 'Modify' : 'Initialize'} Knowledge Node</p>
                                <p className="text-tiny text-default-400 font-mono tracking-widest uppercase">{formState.id}</p>
                            </ModalHeader>
                            <ModalBody className="flex flex-col gap-8">
                                <div className="grid grid-cols-2 gap-6">
                                    <Input
                                        label="Full Legal Name"
                                        labelPlacement="outside"
                                        placeholder="Enter entity name"
                                        value={formState.name}
                                        onValueChange={(v: string) => setFormState({ ...formState, name: v })}
                                        variant="bordered"
                                        size="lg"
                                        classNames={{ input: "font-bold" }}
                                    />
                                    <Input
                                        label="Secure Email"
                                        labelPlacement="outside"
                                        placeholder="entity@mesh.point"
                                        type="email"
                                        value={formState.email}
                                        onValueChange={(v: string) => setFormState({ ...formState, email: v })}
                                        variant="bordered"
                                        size="lg"
                                        classNames={{ input: "font-bold" }}
                                    />
                                </div>
                                <div className="grid grid-cols-2 gap-6">
                                    <Input
                                        label="Mesh Risk Score"
                                        labelPlacement="outside"
                                        type="number"
                                        value={formState.credit_score}
                                        onValueChange={(v: string) => setFormState({ ...formState, credit_score: v })}
                                        variant="bordered"
                                        size="lg"
                                        classNames={{ input: "font-bold" }}
                                    />
                                    <Input
                                        label="Annual Revenue ($)"
                                        labelPlacement="outside"
                                        type="number"
                                        value={formState.income}
                                        onValueChange={(v: string) => setFormState({ ...formState, income: v })}
                                        variant="bordered"
                                        size="lg"
                                        classNames={{ input: "font-bold" }}
                                    />
                                </div>
                                <div className="grid grid-cols-2 gap-6">
                                    <Input
                                        label="DTI Context"
                                        labelPlacement="outside"
                                        type="number"
                                        step="0.01"
                                        value={formState.debt_to_income_ratio}
                                        onValueChange={(v: string) => setFormState({ ...formState, debt_to_income_ratio: v })}
                                        variant="bordered"
                                        size="lg"
                                        classNames={{ input: "font-bold" }}
                                    />
                                    <Input
                                        label="Years in Network"
                                        labelPlacement="outside"
                                        type="number"
                                        value={formState.employment_years}
                                        onValueChange={(v: string) => setFormState({ ...formState, employment_years: v })}
                                        variant="bordered"
                                        size="lg"
                                        classNames={{ input: "font-bold" }}
                                    />
                                </div>
                                <Textarea
                                    label="Registered Address"
                                    labelPlacement="outside"
                                    placeholder="Verify physical grounding point"
                                    value={formState.address}
                                    onValueChange={(v: string) => setFormState({ ...formState, address: v })}
                                    variant="bordered"
                                    size="lg"
                                    classNames={{ input: "font-medium italic" }}
                                />
                            </ModalBody>
                            <ModalFooter className="flex gap-4">
                                <Button color="default" variant="light" onPress={onClose} className="font-bold flex-1 h-12 rounded-2xl">
                                    Abort
                                </Button>
                                <Button
                                    color="primary"
                                    className="font-black rounded-2xl shadow-xl shadow-primary/20 flex-[2] h-12"
                                    onPress={handleSubmit}
                                    isLoading={submitting}
                                >
                                    {isEditMode ? 'Update Model' : 'Initialize Node'}
                                </Button>
                            </ModalFooter>
                        </>
                    )}
                </ModalContent>
            </Modal>
        </div>
    );
}
