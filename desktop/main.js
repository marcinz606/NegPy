const { app, BrowserWindow, screen, Tray, Menu } = require('electron');
const path = require('path');
const { spawn, spawnSync } = require('child_process');
const os = require('os');
const fs = require('fs');
let mainWindow;
let splashWindow;
let backendProcess;
let tray = null;
const isPackaged = app.isPackaged;
const port = 8501;
const ICON_PATH = path.join(__dirname, '..', 'media', 'icons', os.platform() === 'win32' ? 'icon.ico' : 'icon.png');
const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
    app.quit();
} else {
    app.on('second-instance', (event, commandLine, workingDirectory) => {
        // Someone tried to run a second instance, we should focus our window.
        if (mainWindow) {
            if (mainWindow.isMinimized()) mainWindow.restore();
            if (!mainWindow.isVisible()) mainWindow.show();
            mainWindow.focus();
        }
    });

    // --- Error Handling ---
    process.on('uncaughtException', (error) => {
        console.error('Uncaught Exception:', error);
    });

    process.on('unhandledRejection', (reason, promise) => {
        console.error('Unhandled Rejection at:', promise, 'reason:', reason);
    });

    function getNegativePyUserDir() {
        // NEGATIVEPY_USER_DIR: platform-specific Documents/NegativePy
        const homeDocs = app.getPath('documents');
        const userDir = path.join(homeDocs, 'NegativePy');
        if (!fs.existsSync(userDir)) {
            fs.mkdirSync(userDir, { recursive: true });
        }
        return userDir;
    }

    function startBackend() {
        // Proactively kill anything on our port (eg previous dangling instance that didn't close properly)
        if (process.platform === 'win32') {
            // Windows port cleanup (optional, taskkill in will-quit is usually enough)
            try {
                spawn('cmd', ['/c', `for /f "tokens=5" %a in ('netstat -aon ^| findstr :${port}') do taskkill /f /pid %a`]);
            } catch (e) { }
        } else {
            try {
                spawn('sh', ['-c', `lsof -ti :${port} | xargs kill -9`]);
            } catch (e) {
                console.log("Port cleanup skipped or failed");
            }
        }

        const userDir = getNegativePyUserDir();
        const env = { ...process.env, NEGATIVEPY_USER_DIR: userDir };

        let pythonExecutable;
        let args = [];

        if (isPackaged) {
            // Path to the bundled binary
            if (os.platform() === 'win32') {
                pythonExecutable = path.join(process.resourcesPath, 'negativepy', 'negativepy.exe');
            } else {
                // macOS and Linux now both produce a single binary file named 'negativepy'
                pythonExecutable = path.join(process.resourcesPath, 'negativepy', 'negativepy');
            }
        } else {
            // Path to local python/streamlit
            // Try to find venv python first
            const venvPython = os.platform() === 'win32'
                ? path.join(__dirname, '..', 'venv', 'Scripts', 'python.exe')
                : path.join(__dirname, '..', 'venv', 'bin', 'python');

            pythonExecutable = fs.existsSync(venvPython) ? venvPython : 'python';
            args = ['-m', 'streamlit', 'run', 'app.py', '--server.port=' + port, '--server.headless=true', '--browser.gatherUsageStats=false'];
        }

        console.log(`Starting backend: ${pythonExecutable} ${args.join(' ')}`);
        console.log(`User directory: ${userDir}`);

        backendProcess = spawn(pythonExecutable, args, {
            env,
            cwd: isPackaged ? process.resourcesPath : path.join(__dirname, '..'),
            detached: process.platform !== 'win32' // Required for process group kill
        });

        backendProcess.stdout.on('data', (data) => {
            console.log(`Backend: ${data}`);
            if (data.toString().includes('URL: http://localhost:' + port)) {
                if (splashWindow) {
                    createMainWindow();
                }
            }
        });

        backendProcess.stderr.on('data', (data) => {
            console.error(`Backend Error: ${data}`);
        });

        backendProcess.on('close', (code) => {
            console.log(`Backend process exited with code ${code}`);
            if (mainWindow) mainWindow.close();
        });
    }

    function createSplashWindow() {
        splashWindow = new BrowserWindow({
            width: 400,
            height: 300,
            frame: false,
            alwaysOnTop: true,
            transparent: true,
            icon: ICON_PATH,
            webPreferences: {
                nodeIntegration: false
            }
        });

        splashWindow.loadFile(path.join(__dirname, 'splash.html'));
    }

    function createTray() {
        const iconPath = ICON_PATH;
        tray = new Tray(iconPath);

        const contextMenu = Menu.buildFromTemplate([
            {
                label: 'Show NegativePy', click: () => {
                    if (mainWindow) {
                        mainWindow.show();
                        mainWindow.focus();
                    }
                }
            },
            { type: 'separator' },
            {
                label: 'Quit', click: () => {
                    app.isQuitting = true;
                    app.quit();
                }
            }
        ]);

        tray.setToolTip('NegativePy');
        tray.setContextMenu(contextMenu);

        tray.on('click', () => {
            if (mainWindow) {
                if (mainWindow.isVisible()) {
                    mainWindow.hide();
                } else {
                    mainWindow.show();
                    mainWindow.focus();
                }
            }
        });
    }

    function createMainWindow() {
        const { width, height } = screen.getPrimaryDisplay().workAreaSize;

        mainWindow = new BrowserWindow({
            width: Math.min(1600, width),
            height: Math.min(1000, height),
            show: false,
            autoHideMenuBar: true,
            icon: ICON_PATH,
            webPreferences: {
                nodeIntegration: false,
                contextIsolation: true
            }
        });

        mainWindow.loadURL(`http://localhost:${port}`);

        mainWindow.once('ready-to-show', () => {
            if (splashWindow) {
                splashWindow.close();
                splashWindow = null;
            }
            mainWindow.show();
        });

        mainWindow.on('close', (event) => {
            if (!app.isQuitting) {
                event.preventDefault();
                mainWindow.hide();
            }
            return false;
        });

        mainWindow.on('closed', () => {
            mainWindow = null;
        });
    }

    app.on('ready', () => {
        createTray();
        createSplashWindow();
        startBackend();

        // Fallback: If it takes too long, just try to show the main window or show error
        setTimeout(() => {
            if (splashWindow && !mainWindow) {
                console.log("Timeout waiting for backend, trying to connect anyway...");
                createMainWindow();
            }
        }, 10000);
    });

    app.on('window-all-closed', () => {
        if (process.platform !== 'darwin') {
            app.quit();
        }
    });

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createMainWindow();
        }
    });

    app.on('before-quit', () => {
        app.isQuitting = true;
        if (backendProcess) {
            if (os.platform() === 'win32') {
                try {
                    spawnSync("taskkill", ["/pid", backendProcess.pid, '/f', '/t'], { timeout: 1000 });
                } catch (e) { }
            }
        }
    });

    app.on('will-quit', () => {
        if (backendProcess) {
            console.log(`Killing backend process ${backendProcess.pid}...`);
            if (os.platform() === 'win32') {
                try {
                    // Use spawnSync with timeout to ensure we don't hang Electron
                    spawnSync("taskkill", ["/pid", backendProcess.pid, '/f', '/t'], { timeout: 1000 });

                    // Fallback: Kill by image name if it's still hanging
                    spawnSync("taskkill", ["/f", "/im", "negativepy.exe"], { timeout: 1000 });

                    // Extra safety: Cleanup port 8501 just in case
                    spawnSync('cmd', ['/c', `for /f "tokens=5" %a in ('netstat -aon ^| findstr :${port}') do taskkill /f /pid %a`], { timeout: 1000 });
                } catch (e) {
                    console.error(`Error killing backend: ${e}`);
                }

                // Force exit the app process itself
                app.exit(0);
            } else {
                // Kill the whole process group
                try {
                    process.kill(-backendProcess.pid, 'SIGKILL');
                } catch (e) {
                    backendProcess.kill('SIGKILL');
                }
            }
            backendProcess = null;
        }
    });
}
