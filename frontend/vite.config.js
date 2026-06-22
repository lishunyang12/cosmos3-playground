import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build straight into the Python package so the backend serves the SPA.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../cosmos3_playground/static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8800",
    },
  },
});
