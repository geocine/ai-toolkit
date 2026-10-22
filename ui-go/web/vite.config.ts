import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `npm run dev` serves the UI on :5173 and forwards the server routes (/api,
// /v1, /mcp) to the Go server, so it also works against a remote one.
const target = process.env.AITK_API ?? 'http://127.0.0.1:8675';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': target,
      '/v1': target,
      '/mcp': target,
    },
  },
});
