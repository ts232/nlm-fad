import { defineConfig } from "vite";

// Pinned, unique port. 1996 = the year of Keahey & Robertson, "Techniques for
// Nonlinear Magnification Transformations" (IEEE InfoVis), the paper this
// area-based diffusion solver comes from. strictPort = always this port.
export default defineConfig({
  server: { port: 1996, strictPort: true },
  preview: { port: 1996, strictPort: true },
  // Cheap source-hardening for published builds: no source maps, strip all
  // comments (incl. FAD provenance cites) from the bundle. See fisheye-2d's
  // config / NLM-ORIENTATION §IP for the rationale.
  build: { sourcemap: false },
  esbuild: { legalComments: "none" },
});
