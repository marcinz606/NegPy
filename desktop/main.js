const { app, BrowserWindow, screen } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const os = require('os');
const fs = require('fs');

let mainWindow;
let splashWindow;
let backendProcess;

const isPackaged = app.isPackaged;
const port = 8501;

function getDarkroomUserDir() {
    // DARKROOM_USER_DIR: platform-specific Documents/DarkroomPy
    const homeDocs = app.getPath('documents');
    const userDir = path.join(homeDocs, 'DarkroomPy');
    if (!fs.existsSync(userDir)) {
        fs.mkdirSync(userDir, { recursive: true });
    }
    return userDir;
}

function startBackend() {
    // Proactively kill anything on our port
    if (process.platform === 'win32') {
        // Windows port cleanup (optional, taskkill in will-quit is usually enough)
        try {
            spawn('cmd', ['/c', `for /f "tokens=5" %a in ('netstat -aon ^| findstr :${port}') do taskkill /f /pid %a`]);
        } catch (e) {}
    } else {
        try {
            spawn('sh', ['-c', `lsof -ti :${port} | xargs kill -9`]);
        } catch (e) {
            console.log("Port cleanup skipped or failed");
        }
    }

    const userDir = getDarkroomUserDir();
    const env = { ...process.env, DARKROOM_USER_DIR: userDir };

    let pythonExecutable;
    let args = [];

    if (isPackaged) {
        // Path to the bundled binary
        if (os.platform() === 'win32') {
            pythonExecutable = path.join(process.resourcesPath, 'backend', 'backend.exe');
        } else if (os.platform() === 'darwin') {
            // macOS PyInstaller --windowed creates a .app bundle
            pythonExecutable = path.join(process.resourcesPath, 'backend', 'backend.app', 'Contents', 'MacOS', 'backend');
            // Fallback for non-windowed or if it was moved differently
            if (!fs.existsSync(pythonExecutable)) {
                pythonExecutable = path.join(process.resourcesPath, 'backend', 'backend');
            }
        } else {
            pythonExecutable = path.join(process.resourcesPath, 'backend', 'backend');
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
        icon: path.join(__dirname, '..', 'media', os.platform() === 'win32' ? 'icon.ico' : 'icon.png'),
        webPreferences: {
            nodeIntegration: false
        }
    });

    splashWindow.loadFile(path.join(__dirname, 'splash.html'));
}

function createMainWindow() {
    const { width, height } = screen.getPrimaryDisplay().workAreaSize;

    mainWindow = new BrowserWindow({
        width: Math.min(1600, width),
        height: Math.min(1000, height),
        show: false,
        autoHideMenuBar: true,
        icon: path.join(__dirname, '..', 'media', os.platform() === 'win32' ? 'icon.ico' : 'icon.png'),
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

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

app.on('ready', () => {
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

app.on('will-quit', () => {
    if (backendProcess) {
        if (os.platform() === 'win32') {
            spawn("taskkill", ["/pid", backendProcess.pid, '/f', '/t']);
        } else {
            // Kill the whole process group
            try {
                process.kill(-backendProcess.pid, 'SIGKILL');
            } catch (e) {
                backendProcess.kill('SIGKILL');
            }
        }
    }
});
