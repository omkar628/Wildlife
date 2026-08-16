const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktop", {
  isElectron: true,
  selectFolder: () => ipcRenderer.invoke("desktop:select-folder"),
});
