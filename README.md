---
title: Nonlinear Magnification (FAD) — port
type: moc
project: nlm
status: living
created: 2026-06-20
updated: 2026-06-20
tags: [nlm, magnification, fisheye, focus-context, porting, dataviz]
---

# Nonlinear Magnification (FAD) — port

A modern (TypeScript / WebGL2) re-implementation of **FAD**, T. Alan Keahey's
nonlinear-magnification toolkit from his Indiana University PhD work
(Magnivista, 1995–2005). The original is C/C++ for SGI/GLUT; this repo rebuilds
it as standalone web demos, with eventual fold-in to [[Entoptica]] as
composable build-ops.

Its own git repo (separate from Entoptica). The original FAD C/C++ source is
*licensed Magnivista code*; it lives in iCloud and is **symlinked** here as
`FAD/`, not committed.

## Contents

- **[`NLM-ORIENTATION.md`](NLM-ORIENTATION.md)** — the map: what's in the FAD
  tree, the math (closed-form pipeline + area-based diffusion), a numerical-
  methods primer (error diffusion / relaxation / Poisson), related work, and the
  porting plan. Start here.
- **[`demos/fisheye-2d/`](demos/fisheye-2d/README.md)** — Engine A: the
  closed-form radial fisheye on a deformable, optionally textured grid. Portable
  math core: `demos/fisheye-2d/src/warp.ts`. Port 1995.
- **[`demos/diffusion-mag/`](demos/diffusion-mag/README.md)** — Engine B: paint a
  magnification field and watch a mesh relax to satisfy it (area-based diffusion
  solver). Portable core: `demos/diffusion-mag/src/diffuse.ts`. Port 1996.
- **[`demos/data-flow/`](demos/data-flow/README.md)** — Engine B, **data-driven**:
  planes fly over a zone, their density/proximity *becomes* the field, the mesh
  follows the traffic. Reuses the solver; only the field source changes. Port 1997.
- **[`demos/face-magnify/`](demos/face-magnify/README.md)** — Engine B, **image
  features**: a face is detected (face-api 68 landmarks), the landmarks are the
  data points, and the eyes/nose/mouth magnify. Same solver; the field source is
  now features extracted from an image. Port 1998.

## Run a demo

```bash
cd demos/fisheye-2d    && npm install && npm run dev   # → http://localhost:1995
cd demos/diffusion-mag && npm install && npm run dev   # → http://localhost:1996
cd demos/data-flow     && npm install && npm run dev   # → http://localhost:1997
cd demos/face-magnify  && npm install && npm run dev   # → http://localhost:1998
```

Each demo has its own pinned port (the year of the relevant paper), so they
never collide and can run side by side.

## Status

Both engines are prototyped as standalone demos. Next candidates: multi-scale
(stride) convergence for the solver, data-driven field sources (`MeshMag`), 3D,
and defining the `warp.ts`/`diffuse.ts` → Entoptica `src/core` build-op
interface. See `NLM-ORIENTATION.md` §8–9.
