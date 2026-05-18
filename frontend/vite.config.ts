import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy /api → Flask (évite les problèmes CORS en dev)
    proxy: {
      "/api": {
        target: "http://localhost:5005",
        changeOrigin: true,
      },
    },
  },
});