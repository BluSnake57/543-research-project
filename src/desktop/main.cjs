'use strict';
const {app, BrowserWindow, ipcMain, dialog, shell} = require('electron');
const {spawn} = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const crypto = require('node:crypto');
const {validateRequest, dataDirectory} = require('./guard.cjs');
let worker, window, endpoint, dataRoot, quitting = false;
const token = crypto.randomBytes(32).toString('hex');
const repository = path.resolve(__dirname, '../..');
const development = !app.isPackaged;

async function request(method, route, body) {
  validateRequest(method, route);
  if (!endpoint) throw Error('Backend is not ready');
  const response = await fetch(endpoint + route, {method, headers: {
    Authorization: `Bearer ${token}`, 'Content-Type': 'application/json'
  }, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(15000)});
  const result = await response.json();
  if (!response.ok) throw Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail || result));
  return result;
}

async function startWorker() {
  const resourceRoot = development ? repository : process.resourcesPath;
  dataRoot = path.resolve(dataDirectory(process.platform, process.env, app.getPath('home'), app.getPath('appData')));
  fs.mkdirSync(dataRoot, {recursive: true});
  const executable = development
    ? (process.env.VERITAS_PYTHON || path.join(repository, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python'))
    : path.join(resourceRoot, 'worker', process.platform === 'win32' ? 'veritas-worker.exe' : 'veritas-worker');
  if (!fs.existsSync(executable)) throw Error('Backend executable missing. Install the Python backend or build the packaged worker; see README.md.');
  const args = development ? ['-m', 'veritas.cli'] : [];
  args.push('--data-dir', dataRoot, 'worker', '--manifest', path.join(resourceRoot, 'runtime-manifest.json'),
    '--skills', path.join(resourceRoot, 'skills'), '--runtime',
    development ? path.join(repository, 'build', 'worker', 'llama') : path.join(resourceRoot, 'worker', 'llama'));
  worker = spawn(executable, args, {env: {...process.env, VERITAS_WORKER_TOKEN: token,
    PYTHONPATH: development ? path.join(repository, 'src/backend') : ''}, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true});
  const log = fs.createWriteStream(path.join(dataRoot, 'worker.log'), {flags: 'a'});
  worker.stderr.pipe(log);
  await new Promise((resolve, reject) => {
    let buffer = '';
    const timeout = setTimeout(() => reject(Error('Backend startup timed out; inspect worker.log')), 90000);
    worker.once('error', e => {clearTimeout(timeout); reject(e);});
    worker.once('exit', code => {clearTimeout(timeout); reject(Error(`Backend exited (${code}); inspect worker.log`));});
    worker.stdout.on('data', chunk => {
      buffer += chunk.toString();
      while (buffer.includes('\n')) {
        const position = buffer.indexOf('\n'); const line = buffer.slice(0, position); buffer = buffer.slice(position+1);
        try {
          const message = JSON.parse(line);
          if (Number.isInteger(message.port) && message.port > 0 && message.port < 65536) {
            endpoint = `http://127.0.0.1:${message.port}`; clearTimeout(timeout); resolve();
          }
        } catch { /* Library startup messages are not readiness records. */ }
      }
    });
  });
  for (let attempt = 0; attempt < 60; attempt++) {
    try {await request('GET', '/v1/health'); return;} catch {await new Promise(r => setTimeout(r, 250));}
  }
  throw Error('Backend did not become healthy');
}

function registerIPC() {
  function trusted(event) {
    if (!window || event.sender !== window.webContents || event.senderFrame !== window.webContents.mainFrame) throw Error('Untrusted IPC sender');
  }
  ipcMain.handle('veritas:request', (event, method, route, body) => {trusted(event); return request(method, route, body);});
  ipcMain.handle('veritas:pick', async (event, kind) => {
    trusted(event);
    const options = {properties: [kind === 'directory' ? 'openDirectory' : 'openFile']};
    if (kind === 'posts') options.filters = [{name: 'Posts', extensions: ['json', 'jsonl']}];
    if (kind === 'gguf') options.filters = [{name: 'GGUF model', extensions: ['gguf']}];
    const result = await dialog.showOpenDialog(window, options);
    return result.canceled ? null : result.filePaths[0];
  });
  ipcMain.handle('veritas:open-data', event => {trusted(event); return shell.openPath(dataRoot);});
}

if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => {if (window) {window.show(); window.focus();}});
  app.whenReady().then(async () => {
    try {
      await startWorker(); registerIPC();
      window = new BrowserWindow({width: 1320, height: 900, minWidth: 960, minHeight: 640,
        title: 'Veritas', backgroundColor: '#10151d', webPreferences: {
          preload: path.join(__dirname, 'preload.cjs'), nodeIntegration: false,
          contextIsolation: true, sandbox: true, webSecurity: true
        }});
      window.webContents.setWindowOpenHandler(() => ({action: 'deny'}));
      window.webContents.on('will-navigate', (event, url) => {
        if (!(development && url.startsWith('http://127.0.0.1:5173/'))) event.preventDefault();
      });
      if (development && process.env.VERITAS_DEV_URL) await window.loadURL('http://127.0.0.1:5173');
      else await window.loadFile(path.join(repository, 'dist/renderer/index.html'));
      worker.on('exit', () => {if (!quitting) dialog.showErrorBox('Veritas backend stopped', 'Restart the app. Diagnostic details are in worker.log in the application data directory.');});
    } catch (error) {dialog.showErrorBox('Unable to start Veritas', String(error)); app.quit();}
  });
  app.on('window-all-closed', () => app.quit());
  app.on('before-quit', event => {
    if (quitting || !worker || worker.exitCode !== null) return;
    event.preventDefault(); quitting = true;
    const shutdown = endpoint ? request('POST', '/v1/shutdown', {}).catch(() => {}) : Promise.resolve();
    shutdown.finally(() => {worker.kill(); setTimeout(() => app.exit(0), 1500);});
  });
}
