import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Split React / router into a stable vendor chunk so it can be cached
        // independently from page code.  Page chunks are split automatically by
        // React.lazy() dynamic imports in App.tsx.
        manualChunks: (id) => {
          if (
            id.includes("node_modules/react/") ||
            id.includes("node_modules/react-dom/") ||
            id.includes("node_modules/react-router") ||
            id.includes("node_modules/scheduler/")
          ) {
            return "vendor-react";
          }
        },
      },
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      // Forward /health directly to FastAPI root
      "/health": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      // Forward /api/v1/* to FastAPI (path kept as-is; FastAPI prefix is /api/v1)
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
