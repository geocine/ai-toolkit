import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `npm run dev` serves the studio on :5174. A server added with the base URL
// "/aitk" goes through this proxy to AITK_API (default http://127.0.0.1:8675),
// so local development needs no CORS setup on the server.
const target = process.env.AITK_API ?? 'http://127.0.0.1:8675';

export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    port: 5174,
    proxy: {
      '/aitk': { target, changeOrigin: true, rewrite: p => p.replace(/^\/aitk/, '') },
    },
  },
});
