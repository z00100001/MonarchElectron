const { app, BrowserWindow } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const http = require("http");

let streamlitProcess;
let mainWindow;

function waitForStreamlit(url, timeout = 15000) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const check = () => {
      http.get(url, (res) => {
        if (res.statusCode === 200) {
          resolve(true);
        } else {
          retry();
        }
      }).on("error", retry);
    };

    const retry = () => {
      if (Date.now() - start > timeout) {
        reject(new Error("Streamlit failed to start in time"));
      } else {
        setTimeout(check, 500);
      }
    };

    check();
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    title: "Monarch",
    icon: path.join(__dirname, "Monarch.icns"),
    webPreferences: {
      nodeIntegration: false,
    },
  });

  mainWindow.loadURL("http://localhost:8501");

  mainWindow.on("closed", function () {
    mainWindow = null;
    if (streamlitProcess) streamlitProcess.kill();
  });
}

app.whenReady().then(async () => {
  const pythonPath = "/Users/tylerclanton/Desktop/MonarchElectron/.venv/bin/python";
  const appPath = path.join(__dirname, "app.py");

  streamlitProcess = spawn(pythonPath, [
    "-m", "streamlit", "run", appPath,
    "--server.headless=true",
    "--server.port=8501",
    "--browser.serverAddress=localhost",
    "--server.enableXsrfProtection=false",
    "--browser.gatherUsageStats=false"
  ]);

  streamlitProcess.stdout.on("data", (data) => {
    console.log(`stdout: ${data}`);
  });

  streamlitProcess.stderr.on("data", (data) => {
    console.error(`stderr: ${data}`);
  });

  streamlitProcess.on("close", (code) => {
    console.log(`Streamlit exited with code ${code}`);
  });

  try {
    console.log("🔄 Waiting for Streamlit to launch...");
    await waitForStreamlit("http://localhost:8501");
    console.log("✅ Streamlit is live! Launching Electron...");
    createWindow();
  } catch (e) {
    console.error("❌ Failed to connect to Streamlit:", e);
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});