import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

// The FastAPI app (uvicorn src.api.app:app --port 8000) serves both the chat and the
// read-only explore endpoints; in development Vite forwards /api to it.
const apiTarget = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [
    vue({
      template: {
        // <md-*> tags are Material Web custom elements, not Vue components.
        compilerOptions: { isCustomElement: (tag) => tag.startsWith('md-') },
      },
    }),
  ],
  server: {
    port: 5173,
    proxy: { '/api': apiTarget, '/health': apiTarget },
  },
})
