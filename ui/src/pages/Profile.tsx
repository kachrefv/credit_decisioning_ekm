import { useEffect, useState } from 'react';
import { Button, Card, CardHeader, CardBody, CardFooter, Avatar, Divider } from "@heroui/react";
import client from '../api/client';

import { Input, Modal, ModalContent, ModalHeader, ModalBody, ModalFooter, useDisclosure } from "@heroui/react";

const Profile = () => {
    const [user, setUser] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const { isOpen, onOpen, onOpenChange } = useDisclosure();

    // Edit Form State
    const [editName, setEditName] = useState("");
    const [editEmail, setEditEmail] = useState("");
    const [editPicture, setEditPicture] = useState("");
    const [editPassword, setEditPassword] = useState(""); // Only send if changed
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        const fetchProfile = async () => {
            try {
                const res = await client.get('/users/me');
                setUser(res.data);
                setEditName(res.data.name || "");
                setEditEmail(res.data.email || "");
                setEditPicture(res.data.picture || "");
            } catch (err) {
                console.error("Failed to fetch profile:", err);
            } finally {
                setLoading(false);
            }
        };
        fetchProfile();
    }, []);

    const handleSave = async (onClose: () => void) => {
        setSaving(true);
        try {
            const payload: any = {
                name: editName,
                email: editEmail,
                picture: editPicture,
            };
            if (editPassword) {
                payload.password = editPassword;
            }

            const res = await client.put('/users/me', payload);
            setUser(res.data);
            onClose();
            setEditPassword(""); // Clear password field after save
        } catch (err) {
            console.error("Failed to update profile:", err);
            // Could add error toast here
        } finally {
            setSaving(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-full">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
            </div>
        );
    }

    if (!user) {
        return <div>Unable to load profile. Please log in again.</div>;
    }

    return (
        <div className="p-8 max-w-4xl mx-auto space-y-6">
            <header className="mb-8 flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-foreground">Account Settings</h1>
                    <p className="text-default-500">Manage your profile and account preferences</p>
                </div>
                <Button color="primary" onPress={onOpen}>
                    Edit Profile
                </Button>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Profile Summary Card */}
                <Card className="col-span-1 md:col-span-1 h-fit">
                    <CardBody className="flex flex-col items-center gap-4 py-8">
                        <Avatar
                            src={user.picture || "https://i.pravatar.cc/150?u=a042581f4e29026704d"}
                            className="w-24 h-24 text-large"
                            isBordered
                            color="primary"
                        />
                        <div className="text-center">
                            <h2 className="text-xl font-bold">{user.name || "User"}</h2>
                            <p className="text-small text-default-500">{user.email}</p>
                        </div>
                    </CardBody>
                </Card>

                {/* Detailed Info Card */}
                <Card className="col-span-1 md:col-span-2">
                    <CardHeader className="flex gap-3">
                        <div className="flex flex-col">
                            <p className="text-md">Profile Information</p>
                            <p className="text-small text-default-500">Your personal account details</p>
                        </div>
                    </CardHeader>
                    <Divider />
                    <CardBody>
                        <div className="space-y-4">
                            <div className="flex justify-between items-center py-2 border-b border-divider">
                                <span className="text-default-500">Email</span>
                                <span>{user.email}</span>
                            </div>
                            <div className="flex justify-between items-center py-2 border-b border-divider">
                                <span className="text-default-500">Name</span>
                                <span>{user.name}</span>
                            </div>
                            <div className="flex justify-between items-center py-2 border-b border-divider">
                                <span className="text-default-500">Picture URL</span>
                                <span className="text-tiny text-default-400 truncate max-w-[200px]">{user.picture || "N/A"}</span>
                            </div>
                        </div>
                    </CardBody>
                    <CardFooter>
                        <p className="text-tiny text-default-400">Settings managed by Credithos Admin</p>
                    </CardFooter>
                </Card>

                {/* Edit Modal */}
                <Modal isOpen={isOpen} onOpenChange={onOpenChange} placement="top-center">
                    <ModalContent className="bg-content1">
                        {(onClose) => (
                            <>
                                <ModalHeader className="flex flex-col gap-1">Edit Profile</ModalHeader>
                                <ModalBody>
                                    <Input
                                        autoFocus
                                        label="Name"
                                        placeholder="Enter your name"
                                        variant="bordered"
                                        value={editName}
                                        onValueChange={setEditName}
                                    />
                                    <Input
                                        label="Email"
                                        placeholder="Enter your email"
                                        variant="bordered"
                                        value={editEmail}
                                        onValueChange={setEditEmail}
                                    />
                                    <Input
                                        label="Profile Picture URL"
                                        placeholder="https://..."
                                        variant="bordered"
                                        value={editPicture}
                                        onValueChange={setEditPicture}
                                    />
                                    <Input
                                        label="New Password"
                                        placeholder="Leave blank to keep current"
                                        type="password"
                                        variant="bordered"
                                        value={editPassword}
                                        onValueChange={setEditPassword}
                                    />
                                </ModalBody>
                                <ModalFooter>
                                    <Button color="danger" variant="flat" onPress={onClose}>
                                        Cancel
                                    </Button>
                                    <Button color="primary" onPress={() => handleSave(onClose)} isLoading={saving}>
                                        Save Changes
                                    </Button>
                                </ModalFooter>
                            </>
                        )}
                    </ModalContent>
                </Modal>
            </div>
        </div>
    );
};

export default Profile;
