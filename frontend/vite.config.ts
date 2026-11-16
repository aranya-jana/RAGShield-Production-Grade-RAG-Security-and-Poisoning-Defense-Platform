import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  // GitHub Pages serves this repository beneath its repository-name path.
  base: '/RAGShield-Production-Grade-RAG-Security-and-Poisoning-Defense-Platform/',
  plugins: [
    react(),
    tailwindcss(),
  ],
})
