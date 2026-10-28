'use strict';
function validateRequest(method, route) {
  if (!['GET', 'POST', 'PATCH', 'DELETE'].includes(method)) throw Error('Unsupported method');
  if (typeof route !== 'string' || !/^\/v1\/[A-Za-z0-9_/?=&.%:+-]+$/.test(route)) {
    throw Error('Invalid local API route');
  }
  const pathname = route.split('?')[0];
  if (pathname.includes('..') || /%2f|%5c|%2e/i.test(pathname)) throw Error('Invalid local API route');
  return route;
}
function dataDirectory(platform, env, home, appData) {
  const path = require('node:path');
  const join = platform === 'win32' ? path.win32.join : path.posix.join;
  if (env.VERITAS_DATA_DIR) return env.VERITAS_DATA_DIR;
  if (platform === 'win32') return join(env.LOCALAPPDATA || appData, 'Veritas');
  if (platform === 'darwin') return join(home, 'Library/Application Support/Veritas');
  return join(env.XDG_DATA_HOME || join(home, '.local/share'), 'veritas');
}
module.exports = {validateRequest, dataDirectory};
