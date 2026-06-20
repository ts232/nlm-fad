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
- **[`demos/fisheye-2d/`](demos/fisheye-2d/README.md)** — first demo: the
  closed-form radial fisheye on a deformable, optionally textured grid. The
  portable math core is `demos/fisheye-2d/src/warp.ts`.

## Run a demo

```bash
cd demos/fisheye-2d && npm install && npm run dev   # → http://localhost:1995
```

## Status

Engine A (closed-form pipeline) is the working demo. Engine B (the area-based
diffusion solver) is next. See `NLM-ORIENTATION.md` §8–9 for the plan and open
questions.
