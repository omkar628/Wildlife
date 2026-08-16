/**
 * Electron desktop shell.
 * Loads the Vite UI in development and exposes a native folder picker.
 * Does not package an installer or .exe.
 */
const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const path = require("path");

const VITE_URL = process.env.VITE_DEV_SERVER_URL || "http://127.0.0.1:5173";

/** @type {BrowserWindow | null} */
let mainWindow = null;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForVite() {
  const deadline = Date.now() + 180000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(VITE_URL, { method: "GET" });
      if (response.ok) {
        return;
      }
    } catch {
      // Vite is not up yet.
    }
    await sleep(400);
  }
  throw new Error(`Timed out waiting for Vite at ${VITE_URL}`);
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1320,
    height: 860,
    minWidth: 980,
    minHeight: 640,
    title: "Wildlife Intelligence",
    backgroundColor: "#0b100e",
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.once("ready-to-show", () => {
    if (mainWindow) {
      mainWindow.show();
    }
  });

  mainWindow.webContents.on("did-fail-load", (_event, code, description, url) => {
    console.error(`Failed to load ${url}: ${code} ${description}`);
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

async function openApp() {
  createWindow();
  await waitForVite();
  if (!mainWindow) {
    return;
  }
  await mainWindow.loadURL(VITE_URL);
}

ipcMain.handle("desktop:select-folder", async (event) => {
  const parent = BrowserWindow.fromWebContents(event.sender) ?? mainWindow ?? undefined;
  const result = await dialog.showOpenDialog(parent, {
    title: "Select Camera Trap Folder",
    buttonLabel: "Select Folder",
    properties: ["openDirectory"],
  });
  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }
  return result.filePaths[0];
});

app.whenReady().then(() => {
  openApp().catch((error) => {
    console.error(error);
    app.quit();
  });
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      openApp().catch((error) => console.error(error));
    }
  });
});

app.on("window-all-closed", () => {
  app.quit();
});
