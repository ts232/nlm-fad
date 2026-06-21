import { defineConfig } from "vite";

// Pinned, unique port. 1997 — the FAD "data objects over a zone" era (apps/2d).
export default defineConfig({
  server: { port: 1997, strictPort: true },
  preview: { port: 1997, strictPort: true },
  // Cheap source-hardening for published builds: no source maps, strip all
  // comments (incl. FAD provenance cites) from the bundle. See fisheye-2d's
  // config / NLM-ORIENTATION §IP for the rationale.
  build: { sourcemap: false },
  esbuild: { legalComments: "none" },
});
