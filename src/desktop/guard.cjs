'use strict';
function validateRequest(method, route) {
  if (!['GET', 'POST', 'PATCH', 'DELETE'].includes(method)) throw Error('Unsupported method');
  if (typeof route !== 'string' || !/^\/v1\/[A-Za-z0-9_/?=&.%:+-]+$/.test(route) || route.includes('..') || /%2f|%5c|%2e/i.test(route)) {
    throw Error('Invalid local API route');
  }
  return route;
}
module.exports = {validateRequest};
