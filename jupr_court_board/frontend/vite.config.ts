import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",                 // <-- critical for Streamlit components
  plugins: [react()],
  build: {
    outDir: "build",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    strictPort: true,
  },
});
