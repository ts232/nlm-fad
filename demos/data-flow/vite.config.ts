import { defineConfig } from "vite";

// Pinned, unique port. 1997 — the FAD "data objects over a zone" era (apps/2d).
export default defineConfig({
  server: { port: 1997, strictPort: true },
  preview: { port: 1997, strictPort: true },
});
