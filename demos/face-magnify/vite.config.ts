import { defineConfig } from "vite";
// Pinned port 1998 — Keahey, "The Generalized Detail-In-Context Problem" (InfoVis 1998).
export default defineConfig({ server: { port: 1998, strictPort: true }, preview: { port: 1998, strictPort: true } });
