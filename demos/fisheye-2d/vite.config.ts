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
});
