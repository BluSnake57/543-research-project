export type Json = any;
declare global {
  interface Window {veritas: {
    request(method: string, route: string, body?: unknown): Promise<Json>;
    pick(kind: 'directory' | 'posts' | 'gguf' | 'executable'): Promise<string | null>;
    openData(): Promise<void>;
  }}
}
export const api = (method: string, route: string, body?: unknown): Promise<Json> => window.veritas.request(method, route, body);
export const pick = window.veritas.pick;
