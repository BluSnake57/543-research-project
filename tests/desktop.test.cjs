const {test} = require('node:test');
const assert = require('node:assert/strict');
const {validateRequest} = require('../src/desktop/guard.cjs');
const fs = require('node:fs');

test('renderer can request versioned local endpoints', () => {
  assert.equal(validateRequest('GET', '/v1/conversations?q=Mars%20moons'), '/v1/conversations?q=Mars%20moons');
  assert.equal(validateRequest('POST', '/v1/graphs/abc/accept'), '/v1/graphs/abc/accept');
});
test('renderer cannot turn the worker bridge into a network proxy', () => {
  for (const value of ['https://example.com', '//example.com', '/v1/../private', '/v1/%2e%2e/private', '/v1/%2fother', '/v1/health#fragment']) {
    assert.throws(() => validateRequest('GET', value));
  }
  assert.throws(() => validateRequest('CONNECT', '/v1/health'));
});
test('preload exposes no raw IPC or Node objects', () => {
  const source = fs.readFileSync('src/desktop/preload.cjs', 'utf8');
  assert.ok(source.includes('contextBridge.exposeInMainWorld'));
  assert.ok(!source.includes('send:'));
  const main = fs.readFileSync('src/desktop/main.cjs', 'utf8');
  assert.ok(main.includes('nodeIntegration: false'));
  assert.ok(main.includes('contextIsolation: true'));
  assert.ok(main.includes('sandbox: true'));
});
