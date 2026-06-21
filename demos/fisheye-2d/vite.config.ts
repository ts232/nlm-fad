import { defineConfig } from "vite";

// Pinned, unique port so this demo never fights Entoptica (5173) or anything
// else for a port. 1995 = the year of IUCS-TR455, the original Nonlinear
// Magnification tech report. strictPort = always this port, predictable URL.
export default defineConfig({
  server: {
    port: 1995,
    strictPort: true,
  },
  preview: {
    port: 1995,
    strictPort: true,
  },
  // Cheap source-hardening for published builds: never ship source maps (they
  // reconstruct the original commented source), and drop every comment from the
  // bundle — including the FAD file:line provenance cites in warp.ts, which stay
  // useful in-source but shouldn't travel into a public build. Minification
  // (esbuild, the prod default) does the rest. Not real protection — a speed
  // bump against casual copying; see NLM-ORIENTATION §IP.
  build: { sourcemap: false },
  esbuild: { legalComments: "none" },
});
