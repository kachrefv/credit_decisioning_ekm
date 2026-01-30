import { app, BrowserWindow, ipcMain, shell, protocol, net } from 'electron';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';
import { exec } from 'child_process';
import fs from 'fs';

// Re-defining to avoid ESM issues in trial
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isDev = !app.isPackaged;
let mainWindow: BrowserWindow | null;

// Register 'app' protocol as privileged
if (!isDev) {
    protocol.registerSchemesAsPrivileged([
        { scheme: 'app', privileges: { secure: true, standard: true, supportFetchAPI: true } }
    ]);
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false,
            contextIsolation: true,
            sandbox: false
        },
    });

    if (isDev) {
        mainWindow.loadURL('http://localhost:5173');
        mainWindow.webContents.openDevTools();
    } else {
        // ALWAYS open DevTools for this fixed build
        mainWindow.webContents.openDevTools();
        mainWindow.loadURL('app://application/index.html');
    }

    // Set icon
    mainWindow.setIcon(path.join(__dirname, isDev ? '../public/icon.png' : '../dist/icon.png'));

    // Capture and log loading errors
    mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription, validatedURL) => {
        console.error(`Failed to load URL: ${validatedURL} with error: ${errorDescription} (${errorCode})`);
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

app.whenReady().then(() => {
    if (!isDev) {
        protocol.handle('app', (request) => {
            const url = new URL(request.url);
            const appPath = app.getAppPath();

            // In production, app.getAppPath() usually points to current directory or resources/app.asar
            // Our files are in ../dist relative to dist-electron/main.js
            // But packaged main.js is in resources/app.asar/dist-electron/main.js
            // So dist is at resources/app.asar/dist

            let filePath = decodeURIComponent(url.pathname);
            if (filePath.startsWith('/')) filePath = filePath.substring(1);

            // Map app://application/index.html -> <root>/dist/index.html
            // If the path is empty or just 'application', default to application/index.html
            const relativePath = filePath.replace(/^application\//, '');
            const finalPath = relativePath === '' || relativePath === 'application' ? 'index.html' : relativePath;

            const absolutePath = path.join(appPath, 'dist', finalPath);

            console.log(`[App Protocol] Request: ${request.url} -> ${absolutePath}`);

            return net.fetch(pathToFileURL(absolutePath).toString());
        });
    }

    createWindow();

    // IPC Handlers
    ipcMain.handle('create-desktop-shortcut', async () => {
        if (process.platform !== 'win32') return { success: false, message: 'Only supported on Windows' };

        const exePath = app.getPath('exe');
        const desktopPath = app.getPath('desktop');
        const shortcutPath = path.join(desktopPath, 'Credithos.lnk');

        const psCommand = `
            $WshShell = New-Object -ComObject WScript.Shell
            $Shortcut = $WshShell.CreateShortcut("${shortcutPath}")
            $Shortcut.TargetPath = "${exePath}"
            $Shortcut.Save()
        `;

        return new Promise((resolve) => {
            const { exec } = require('child_process');
            exec(`powershell -Command "${psCommand.replace(/\n/g, '')}"`, (error: any) => {
                if (error) {
                    console.error('Shortcut error:', error);
                    resolve({ success: false, message: error.message });
                } else {
                    resolve({ success: true });
                }
            });
        });
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    }
});
