# CLAUDE.md — Nonlinear Magnification (FAD) port

Guidance for Claude Code working in `Experiments/nlm/`. This sits under the Entoptica umbrella `CLAUDE.md` but the rules here take precedence for this directory.

## What this is

A modern **TypeScript / WebGL2** re-implementation of **FAD**, T. Alan Keahey's nonlinear-magnification toolkit (Indiana University PhD; Magnivista, 1995–2005; original is C/C++ for SGI/GLUT). **The user is the original author** *and* a data-visualization PhD — engage at research level; explain unfamiliar numerical-analysis names (Floyd–Steinberg, Gauss–Seidel) but never condescend about his own work. Goal: standalone web demos now, eventual fold-in to Entoptica `src/core` as composable build-ops. Full reference: **`NLM-ORIENTATION.md`** (repo map, math, numerical-methods primer, related work, per-demo notes §8).

## Repo rules

- **This is its own git repo** (`Experiments/nlm/.git`, branch `main`), separate from Entoptica. **Do NOT commit nlm into the Entoptica repo** — Entoptica `.gitignore`s `Experiments/nlm/`.
- **`FAD/` is a symlink** to the original **licensed Magnivista C/C++** source in iCloud. It is gitignored — never commit it, never redistribute it. Quoting small snippets in docs is fine (he owns it).
- Commit directly when work is done (no need to confirm the message). End commit messages with the `Co-Authored-By` trailer.

## Architecture — the one idea

**One solver, swappable field source.** Two engines from FAD:

- **Engine A — closed-form pipeline** (`trans2D`): analytic radial/ortho fisheye kernels (Warp + Domain + mux + filter). Shader-friendly. → `demos/fisheye-2d/src/warp.ts`.
- **Engine B — area-based diffusion** (`mag2D`/`mag3D` `MeshDiff`): specify a magnification *field* `z` per node; iteratively displace mesh vertices so each node's local area-ratio → `z` (error-diffusion + damped Gauss–Seidel; the "integral-equation" relaxation the user described). → `src/diffuse.ts` + `src/field.ts`.

The Engine B demos differ **only in where the field comes from**: paint it, derive it from moving data, or extract it from image features. Keep that seam clean.

## Demos (each standalone, pinned thematic port)

| demo | port | what | field source |
|---|---|---|---|
| `fisheye-2d` | 1995 | closed-form fisheye on a grid/image | — (Engine A) |
| `diffusion-mag` | 1996 | paint a magnification field, watch it solve | brush |
| `data-flow` | 1997 | planes over a zone → density/proximity | moving data points |
| `face-magnify` | 1998 | magnify a face's features | face-api 68 landmarks |

Ports are pinned (`strictPort`) by relevant paper year (1995 TR455 · 1996 InfoVis · 1997 apps/2d · 1998 detail-in-context). The user is "sick of port management" — **never let demos collide; always pin a new unique port.**

## Conventions (what works for this user)

- **One standalone prototype per control space.** When the *interaction model* differs (aim-a-lens vs paint vs animated-data vs image-features), make a new `demos/<name>/` even though they share core code — don't pile modes into one app.
- **Copy the portable cores per demo** (`diffuse.ts`, `field.ts`); keep the copies in sync. Real dedup happens at Entoptica fold-in. `warp.ts`/`diffuse.ts`/`field.ts` are the files that graduate into `src/core`.
- **Verify findings empirically before applying them.** The TS cores run headless via `node --experimental-strip-types some_test.ts` (Node 23). Used it to benchmark convergence and to *disprove* the obvious-looking optimization.
- **Faithful first, then honest correction.** Port FAD's actual algorithm (cite file:line); when it's too weak/wrong for modern use, fix it and say why in a comment. If you mis-state a fact, own it and fix the docs.
- Monochrome/muted UI, one purposeful accent (matches the user's global prefs).

## Run / verify

```bash
cd demos/<name> && npm install && npm run dev    # pinned port, see table
# headless core test:
node --experimental-strip-types path/to/test.ts
```

The browser is needed to *see* WebGL output (and to run face-api, which is **TensorFlow.js on the WebGL backend — NOT WebAssembly**); the math cores are verifiable headless.

## Gotchas / hard-won facts

- **Diffusion speed:** the dominant lever is the per-step clamp `clampEps` (0.1→0.25 ≈ 2.5×), not `refineCoeff`. **Multi-scale `stride` does NOT help** — this damped/clamped nonlinear relaxation is rate-limited, not propagation-limited (benchmarked; kept on `diffuseStep` for completeness, demos run stride 1).
- **Field choice:** for point data (`data-flow`), `proximity` (bounded halo per point, scaled by per-point weight) is the stable default; `density` is a kernel splat that *sums* → clusters over-deform ("ugly spiral"), available but spikier. For *feature* data (`face-magnify`) both are wrong — use `featureField.ts` (shape-distance kernels + soft union), see below.
- **A pinned boundary conserves area, so mean(z) MUST be 1.** Any field whose mean is above 1 is asking for more area than the frame has; the relaxation cannot converge and instead drifts back up after bottoming out (measured: proximity on landmarks, mean z 1.132, RMS 0.118 at ~600 sweeps → 0.150 by 1200). Project the field onto mean(z) = 1 by **subtracting a constant** — the least-squares projection, preserves every feature difference, and is exactly the detail-in-context trade. `featureField.ts` `balanceField()`; generic, belongs in `field.ts` at fold-in.
- **Never pick a per-point weight by nearest neighbour.** `computeProximityField` does (`field.ts:106`) and it makes `z` jump across every Voronoi seam between differently-weighted points (0.625 between adjacent nodes). Combine smooth per-*feature* kernels with a soft union `(Σ eᵖ)^(1/p)` instead — C¹, and p is a real control (1 = additive, large = separate).
- **Put the units on the control — and make them true.** The old face sliders were a unitless 0–12 that reached the mesh through `tanh(radius) ≈ 0.2165`, so "6" meant 2.28×. Sliders now read `1.55×` in 0.01 steps. But blur + balance then shaved 10–15% off each feature's excess (1.55× delivered 1.47×), so `featureField.ts` **calibrates** per-feature gains until the finished field hits the target on each feature's core (ring interior / curve band). A ring's core must be its interior only — including the rim band under-reads and overshoots (measured 1.596× for 1.55×).
- **Compare methods at equal targets.** The first face write-up claimed a "40×" residual win by comparing the old 2.28× defaults against new 1.55× targets; at identical targets it is ~4× (and balance matters most at *strong* settings). `test/diagnose.ts` runs the fair comparison, including delivered magnification inside the eye/mouth rings.
- **Don't reset the mesh when a target changes.** Retarget the field and let the running relaxation walk there; resetting on every slider nudge is what made the face demo feel uncontrollable.
- **Per-point weight** (`field.ts` `Planes.w`) carries per-feature strength in `face-magnify` (proximity scales by the nearest point's weight). Backward-compatible — unset → 1.
- **face-api** assets are vendored under `face-magnify/public/{vendor,models}` (MIT), mirroring Entoptica's `src/core/field/faceApi.ts`. Landmarks are typed (`getMouth/getLeftEye/getNose/getJawOutline/…`) → per-feature controls.

## Next candidates

Per-feature radius; other feature sources (edges/saliency/corners — same template); 3D (`trans3D`/`mag3D`); defining the `warp.ts`/`diffuse.ts`/`field.ts` → Entoptica `src/core` build-op interface. See `NLM-ORIENTATION.md` §9.
