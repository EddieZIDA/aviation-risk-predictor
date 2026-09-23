import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy /api → Flask (évite les problèmes CORS en dev)
    proxy: {
      "/api": {
        // 127.0.0.1 et non « localhost » : Node résout localhost en IPv6 (::1)
        // alors que Flask écoute par défaut sur 127.0.0.1 (FLASK_HOST).
        target: process.env.VITE_PROXY_TARGET ?? "http://127.0.0.1:5005",
        changeOrigin: true,
      },
    },
  },
});