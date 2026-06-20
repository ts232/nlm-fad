# Diffusion magnification (paint a field) — FAD demo

The **data-driven** half of FAD: instead of aiming a parametric lens, you **paint
a magnification field** and an iterative solver **relaxes a mesh** until each
region's local area matches what you asked for. This is the TypeScript port of
FAD's `MeshDiff` / `MeshMag` (Keahey, IEEE InfoVis 1996) — Engine B in
[`../../NLM-ORIENTATION.md`](../../NLM-ORIENTATION.md).

## Run

```bash
npm install
npm run dev      # → http://localhost:1996  (pinned port; 1996 = the InfoVis paper year)
```

## What it does

- **Paint** magnification onto the mesh with a soft brush (warm = enlarge, cool =
  shrink, plus an erase mode). The painted scalar field is the *target*.
- The **solver runs live**: each frame it performs N relaxation sweeps, and you
  watch the mesh deform toward the field — grid lines bend, the textured image
  magnifies where you painted, the heat overlay shows the field.
- A **RMS magnification error** readout shows convergence.

## How the solver works (`src/diffuse.ts`)

Faithful port of `MeshDiff::EnergyDiffuse` + `MoveNeighbour`:

1. At each node, measure the achieved magnification = local **area ratio**
   (current quad area / rest area).
2. **Error** = target / achieved − 1 (>0 under-magnified, <0 over).
3. Damp by `refineCoeff` (~0.3), clamp, then **diffuse the error to the four
   neighbours**, distance-weighted — physically **pushing them apart** (under) or
   **pulling them together** (over).
4. Serpentine sweep direction each pass; optional pinned boundary holds the frame.

This is error diffusion (Floyd–Steinberg in spirit) on mesh geometry, equivalently
a damped nonlinear Gauss–Seidel relaxation of a prescribed-Jacobian (Poisson)
problem. See `NLM-ORIENTATION.md` §5 for the math.

## Controls

- **Brush** — mode (magnify / minify / erase), size, strength, max magnification.
- **Solver** — refine coeff (rate), **step cap** (the per-step displacement clamp
  — the dominant speed↔stability lever), iterations per frame, error weighting
  (by-mag / uniform / inverse), pin-boundary.
- **Run / Pause**, **Step ×iters** (one burst while paused), **Reset mesh**
  (positions home, keep field), **Clear field** (z → 1).
- **Display** — image / field heat overlay / mesh, and the texture source.

## Notes & known behaviour

- **Convergence is "soft" and bounded.** Small/modest fields converge to their
  target almost exactly; a *large* high-magnification region can't fully resolve
  because the pinned frame conserves total area — you can't fit a 3× region that
  large without infinite compression elsewhere. That plateau is correct physics,
  not a bug.
- **On speed.** The dominant lever is the **step cap** (`clampEps`): raising it
  0.1 → 0.25 converges ~2.5× faster while still settling on target. `refineCoeff`
  and sweeps/frame help linearly. Multi-scale (`stride`) was implemented and
  benchmarked but does **not** help — this damped/clamped nonlinear relaxation is
  rate-limited, not propagation-limited, so coarse passes add overhead without
  payoff. The `stride` parameter is kept in `diffuse.ts` for completeness; the
  demo runs stride 1. (Push the sliders too hard — high refine + high step cap —
  and it will overshoot/oscillate; the defaults leave headroom.)
- The field is **Lagrangian** — it's attached to nodes, so it deforms with the
  mesh (a magnified region's heat blob grows with it).
- Solver runs **on the CPU** (`diffuse.ts`, the portable core, no WebGL/DOM
  dependency) — faithful to the original and fine at this grid size. A compute-
  shader / WASM version is a later option if larger meshes are wanted.

## Not yet (future)

Non-square aspect, data-driven field sources (density / proximity / image-
gradient, à la FAD `MeshMag`), and an "achieved vs target" error visualization.
