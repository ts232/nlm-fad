# Data-driven magnification (traffic over a zone) — FAD demo

The data-driven mode of FAD: the magnification field isn't painted, it's
**derived from data**. Planes fly across a zone; their **density** (or
**proximity**) becomes the magnification field; the diffusion solver continuously
relaxes the mesh to follow the traffic; the planes ride the warp, so clusters
visibly spread apart. This is the TypeScript port of FAD's `apps/2d` +
`MeshMag::ComputeDensityToZ / ComputeProximityToZ` (Keahey, ~1997).

It reuses the Engine B solver (`diffuse.ts`) **verbatim** — only the field
*source* changes. That's the point: the same machine that took a hand-painted
field in [`../diffusion-mag`](../diffusion-mag/README.md) takes a data-derived
one here, and will take an image-feature-derived one next.

## Run

```bash
npm install
npm run dev      # → http://localhost:1997  (pinned port)
```

## What it does

- **Planes** (`field.ts`): N points with position + velocity, bouncing in the
  [-1,1] zone (count + speed adjustable; deterministic seed, "Re-seed" to vary).
- **Field from the data**, recomputed every frame:
  - **Density** — accumulate each plane into its grid cell (+ 2×2 quad).
    Magnify where the traffic is. (FAD divides the budget by N; we use a fixed
    reference so it reads well at any plane count — `volume` is the knob.)
  - **Proximity** — each node's `z = 1 + tanh(haloRadius − dist_to_nearest)·volume`.
    A magnifying halo around each plane.
- **Solve + render**: the field drives `diffuseStep`; the deforming mesh is drawn
  with the zone map under it, an optional field heat overlay, and the planes
  bilinear-mapped onto the warp (`warpLookup`) so they sit on the magnified mesh.

## Controls

- **Data** — field method (density / proximity), # planes, speed, magnification
  (volume), halo radius (proximity only).
- **Solver** — refine coeff, step cap, iterations/frame (fewer than the paint
  demo, since the field moves every frame and the solver chases it).
- **Play / Pause**, **Re-seed** (new plane layout), **Reset mesh**.
- **Display** — zone map / field heat overlay / mesh / planes.

## Notes

- The solver *chases a moving target*: the field changes each frame as planes
  move, so the mesh never fully settles — it tracks. That continuous tracking is
  the effect. RMS error stays nonzero while planes move (expected).
- Density magnification is naturally gentle unless traffic clusters; raise
  `volume`, lower the plane count, or switch to **proximity** for a stronger,
  always-on effect.
- This is the architecture image-feature magnification will plug into: replace
  `computeDensityField` with `computeSaliencyField(image)` and everything else is
  unchanged.
