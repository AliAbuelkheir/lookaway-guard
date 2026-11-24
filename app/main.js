const path = require('path');
const { app, BrowserWindow, ipcMain, session } = require('electron');
const { loadModels, analyze } = require('./pipeline');

app.whenReady().then(async () => {
  // Allow the camera, nothing else
  session.defaultSession.setPermissionRequestHandler((_wc, permission, done) => done(permission === 'media'));
  session.defaultSession.setPermissionCheckHandler((_wc, permission) => permission === 'media');

  await loadModels(path.join(__dirname, 'models'));
  ipcMain.handle('analyze', (_event, frame) => analyze(frame));

  const win = new BrowserWindow({
    width: 1180, height: 760, minWidth: 900, minHeight: 600, backgroundColor: '#14171a', title: 'LookAway Guard',
    webPreferences: { preload: path.join(__dirname, 'preload.js') },
  });
  win.removeMenu();
  win.loadFile('index.html');
});

app.on('window-all-closed', () => app.quit());
