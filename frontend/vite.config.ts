import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: '0.0.0.0',  // Accept connections from any host (needed for Docker)
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.NODE_ENV === 'production' ? 'http://backend:8000' : 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      }
    }
  }
})

