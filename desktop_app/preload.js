const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('pipeline', {
  analyze: (frame) => ipcRenderer.invoke('analyze', frame),
});
