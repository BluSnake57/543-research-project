import {mkdir, copyFile} from 'node:fs/promises';
await mkdir('dist/desktop', {recursive: true});
for (const file of ['main.cjs', 'preload.cjs', 'guard.cjs']) await copyFile('src/desktop/'+file, 'dist/desktop/'+file);
