'use strict';
const {contextBridge, ipcRenderer} = require('electron');
contextBridge.exposeInMainWorld('veritas', {
  request: (method, route, body) => ipcRenderer.invoke('veritas:request', method, route, body),
  pick: kind => ipcRenderer.invoke('veritas:pick', kind),
  openData: () => ipcRenderer.invoke('veritas:open-data')
});
