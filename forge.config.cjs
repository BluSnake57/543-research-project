const fs = require('node:fs');
module.exports = {
  packagerConfig: {
    asar: true, executableName: 'Veritas',
    extraResource: ['build/worker', 'skills', 'build/runtime-manifest.json'],
    osxSign: process.env.APPLE_IDENTITY ? {identity: process.env.APPLE_IDENTITY} : undefined,
    osxNotarize: process.env.APPLE_ID ? {appleId: process.env.APPLE_ID, appleIdPassword: process.env.APPLE_PASSWORD, teamId: process.env.APPLE_TEAM_ID} : undefined,
    ignore: [/^\/src/, /^\/tests/, /^\/\.venv/, /^\/resources/, /^\/docs/, /^\/build/]
  },
  hooks: {prePackage: async () => {if (!fs.existsSync('build/worker')) throw Error('Run python scripts/package_worker.py first');}},
  makers: [
    {name: '@electron-forge/maker-squirrel', config: {name: 'Veritas'}},
    {name: '@electron-forge/maker-zip', platforms: ['darwin', 'linux']},
    {name: '@electron-forge/maker-deb', config: {}},
    {name: '@electron-forge/maker-rpm', config: {}}
  ]
};
