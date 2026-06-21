import { defineConfig } from "vite";
// Pinned port 1998 — Keahey, "The Generalized Detail-In-Context Problem" (InfoVis 1998).
export default defineConfig({
  server: { port: 1998, strictPort: true },
  preview: { port: 1998, strictPort: true },
  // Cheap source-hardening for published builds: no source maps, strip all
  // comments (incl. FAD provenance cites) from the bundle. See fisheye-2d's
  // config / NLM-ORIENTATION §IP for the rationale.
  build: { sourcemap: false },
  esbuild: { legalComments: "none" },
});
