import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `npm run dev` serves the UI on :5173 and forwards /api to the Go server.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': process.env.AITK_API ?? 'http://127.0.0.1:8675',
    },
  },
});
