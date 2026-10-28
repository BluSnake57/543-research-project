import {spawn} from 'node:child_process';
import './build-desktop.mjs';
const processes = [];
function run(command, args, env = {}) {
  const child = spawn(command, args, {stdio: 'inherit', env: {...process.env, ...env}, shell: process.platform === 'win32'});
  processes.push(child); child.on('error', error => {console.error(error); shutdown(1);}); return child;
}
function shutdown(code = 0) {for (const child of processes) child.kill(); process.exit(code);}
process.on('SIGINT', () => shutdown()); process.on('SIGTERM', () => shutdown());
run('npm', ['exec', 'vite']);
for (let attempt = 0; attempt < 80; attempt++) {
  try {const response = await fetch('http://127.0.0.1:5173'); if (response.ok) break;} catch {}
  await new Promise(resolve => setTimeout(resolve, 250));
  if (attempt === 79) shutdown(1);
}
run('npm', ['exec', 'electron', '.'], {VERITAS_DEV_URL: '1'}).on('exit', code => shutdown(code ?? 0));
