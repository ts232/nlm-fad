# Magnify a face from its landmarks — FAD demo

Image features → magnification. A face is detected, its **68 facial landmarks**
outline each feature, and the diffusion solver (Engine B) magnifies those
features — so the **eyes, nose and mouth enlarge** while the periphery
compresses to pay for them. The data-driven magnification of
[`../data-flow`](../data-flow/README.md), but the "data" is now *features
extracted from an image* (the thing this whole port was building toward).

## Run

```bash
npm install
npm run dev      # → http://localhost:1998  (pinned port)
```

## How it works

- **Landmarks** (`src/faceApi.ts`) — loads **face-api.js@0.22.2** + TinyFaceDetector
  / 68-landmark weights from local `/public` assets (vendored, no CDN), mirroring
  Entoptica's `src/core/field/faceApi.ts`. Runs on **TensorFlow.js's WebGL (GPU)
  backend** — not WebAssembly. `detectAllFaces(img).withFaceLandmarks()` → 68
  points per face, **grouped by feature** and normalized to mesh space `[-1,1]`.
- **Feature shapes → field** (`src/featureField.ts`, the default) — the 68
  landmarks are typed (face-api's `getMouth/getLeftEye/getNose/getJawOutline/…`),
  so each group is a **shape**, not a scatter of points: a closed ring for the
  eyes and mouth, an open curve for the brows, nose and jaw. The kernel is the
  distance to that outline (0 inside a ring) put through a C¹ `smoothstep`
  skirt, so **an eye magnifies as an eye**. Features combine by a **soft union**
  `(Σ eᵖ)^(1/p)` — `p = 1` adds where they overlap, large `p` keeps each
  separate — and the finished field gets an optional binomial blur.
- **Per-feature magnification** — one slider per group in honest units: `1.55×`
  means "this feature should end up 1.55× its area". `1.00×` = leave it alone.
- **Balance area (detail-in-context)** — a pinned boundary conserves total area,
  so the mean of `z` **must** be 1; the field is projected onto that constraint
  by subtracting a constant (the least-squares projection, which preserves every
  difference between features). Magnifying the features is *paid for* by
  compressing the periphery. See "Why this replaced the point-cloud field" below.
- **Legacy point-cloud fields** — the original `field.ts` shared with
  `data-flow` is still selectable: **proximity** (a halo per point, `z = 1 +
  tanh(radius − dist)·weight`) and **density** (a kernel splat that *sums*, so
  clusters over-deform). The magnification sliders drive these too — a target of
  `M×` maps to the weight `(M − 1)/tanh(radius)` — so the control means the same
  thing under all three methods.
- **Solve + render** — `diffuse.ts` (Engine B) relaxes the mesh to the field;
  the face image rides the warp, with the landmark points and mesh overlaid.

So the *only* thing different from the planes demo is the field **source**.
`diffuse.ts` is untouched, and so is the `field.ts` shared with `data-flow` —
the new work lives entirely in `featureField.ts`, on the same Engine-B seam.

## Why this replaced the point-cloud field

The first build magnified from the landmark *points*, and it made blobby,
uncontrollable faces. `test/diagnose.ts` measures why, headless, against a
committed fixture of real landmarks (`test/landmarks.sample.json`):

| field | mean z | max Δz between neighbours | RMS error after 1200 sweeps |
|---|---|---|---|
| proximity, as shipped | 1.132 | 0.625 | **0.1495** — best was 0.1178 at ~600, then *drifts back up* |
| density, as shipped | 1.958 | 2.482 | 1.2984 |
| feature shapes | 1.015 | 0.122 | 0.0089 (monotone) |
| feature shapes + balance | **1.000** | 0.122 | **0.0037** (monotone) |

Four separate faults, all fixed above:

1. **The target was infeasible.** `pinBoundary` conserves total area, so mean `z`
   must be 1; the point cloud asked for 1.132 (density: 1.958). The relaxation
   cannot satisfy that, so it never converges — it reaches its best residual
   around 600 sweeps and then *drifts 27% back up*. That slow crawl, not mesh
   folding (0 folded cells in every case), is what read as "blobby".
2. **The field was discontinuous.** `computeProximityField` uses the weight of
   the **nearest** point (`field.ts:106`), so `z` jumps across every Voronoi seam
   between features of different strength — 0.625 between adjacent nodes.
3. **Cones, not features.** `1 + tanh(maxDist − minDist)·v` is near-linear over
   these radii, so each landmark contributes a cone: a gradient discontinuity at
   the apex and a kink at the rim. 68 cones unioned by nearest-point is a lumpy
   surface that has nothing to do with the shape of an eye.
4. **The units were opaque.** `tanh(0.22) ≈ 0.2165`, so the old "strength 6"
   silently meant 2.28×. The slider number was uninterpretable, which is most of
   why it was hard to aim.

Two smaller behavioural fixes came with it: moving a slider no longer resets the
mesh (the old build restarted the solve from a flat grid on every nudge, so
nothing felt continuous), and the heat overlay is now two-sided — warm for
magnified, cool for the compression that pays for it.

Verify any of this with:

```bash
node --experimental-strip-types test/diagnose.ts
```

## Controls

- **Field** — feature shapes / proximity / density; master strength, falloff
  width, feature blend, field smoothing, balance area (shapes), influence
  radius (legacy methods).
- **Per-feature magnification** — eyes / brows / nose / mouth / jaw, `1.00×`–`2.50×`
  in 0.01 steps.
- **Solver** — refine coeff, step cap, iterations/frame; Play/Pause, Reset mesh,
  and a live **RMS magnification error** readout — the honest read on whether the
  target is achievable. It should fall and stay down.
- **Display** — face image / field heat overlay / mesh / landmarks.
- **Load a face** — pick a file, or **drop any front-facing face** onto the canvas.

## Notes

- Detection runs in-browser on load (~1–2 s the first time: model fetch + TF.js WebGL init).
  Status is shown in the panel ("N landmarks", "no face found", etc.).
- Best on a reasonably front-facing, well-lit face (TinyFaceDetector). Profile /
  tiny / occluded faces may not detect.
- The vendored face-api assets (`public/vendor`, `public/models`) are MIT
  (face-api.js) and committed so the demo is self-contained. `sample-face.png`
  is the same test face Entoptica uses.
- This is the template for any feature→field source: swap the landmark detector
  for edges / saliency / corners and nothing downstream changes.
- `test/landmarks.sample.json` is the real 68-point detection for
  `sample-face.png`, dumped once from the browser so the field maths is testable
  headless. Detection itself is still browser-only.
- Next candidates: a **per-feature falloff width** (an eye and a jawline want
  different skirts from one global slider), and anisotropic kernels that follow
  each feature's principal axis.
