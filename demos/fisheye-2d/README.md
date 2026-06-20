# Closed-form fisheye (2D) — FAD demo

A standalone, dependency-light demo of the **closed-form nonlinear-magnification**
half of the FAD toolkit: a single radial fisheye focus over a round domain,
applied to a deformable grid that can optionally be textured with an image.

It is the first porting beachhead for moving FAD from C/C++ to the Entoptica
stack (TypeScript + WebGL2 + Vite). See [`../../NLM-ORIENTATION.md`](../../NLM-ORIENTATION.md)
for the full toolkit map, the math, and provenance.

## Run

```bash
npm install
npm run dev      # → http://localhost:1995  (pinned port; never conflicts with Entoptica's 5173)
```

Port **1995** (the year of IUCS-TR455, the original Nonlinear Magnification tech
report) is pinned in `vite.config.ts` with `strictPort`, so the URL is identical
every run.

## What it shows

- A regular **point lattice** over `[-1,1]²` warped by a radial fisheye, drawn as
  curved **grid lines** and/or a **textured mesh** — exactly FAD's model (warp a
  `PointGrid`, texture-map it).
- **Focus** follows the cursor. **β** sets magnification strength, **γ** the
  domain radius (the lens fades to identity at its rim), **filter weight** blends
  identity → full transform.
- Three radial kernels: `fisheye` `(1+β)r/(βr+1)`, `hyper` `tanh(βr)`,
  `logistic`. All are boundary-normalized so the lens is seamless at its edge.
- Textures: the iconic **DC Metro map** (the original FAD demo image), procedural
  **test patterns** (square / wide / tall — to confirm aspect handling), or
  drag-and-drop / pick your own image.

## Images — sizes, shapes, limits

- **Any aspect ratio.** The canvas adopts each image's true shape; the warp is
  aspect-corrected so the lens stays a perfect circle on a non-square canvas.
- **Any size up to `gl.MAX_TEXTURE_SIZE`** (typically 8192 on integrated GPUs,
  16384 on Apple Silicon / modern discrete GPUs — logged to the console on load).
  Larger images are automatically downscaled to fit, preserving aspect.
- **No power-of-two requirement** (WebGL2 supports NPOT textures with mipmaps).
- **Trilinear mipmaps + anisotropic filtering** on upload: a high-resolution
  source minified into a small viewport stays clean in the context view, while
  the magnified focus pulls genuine detail (not upscaled pixels). This is the
  payoff of large textures — e.g. a 45 MP camera image (8192×5464) shown ~400px
  wide has ~20× of real detail to reveal under the lens.

## Layout

- `src/warp.ts` — the portable math core (kernels, `warpPoint`, lattice). **This
  is the reusable library seed**; it has no WebGL/DOM dependency and is the file
  that will graduate into Entoptica.
- `src/main.ts` — WebGL2 renderer + interaction (recomputes the warped lattice on
  the CPU each change, uploads positions, draws textured tris + grid lines).
- `index.html` — monochrome control panel.
- `public/dcMetro.png` — converted from `FAD/dev/fad/demo/image/dcMetro.ppm`.

## Notes

- The warp runs **per-vertex on the CPU** via `warp.ts` (≈26k vertices, trivial),
  which keeps `warp.ts` the single source of truth. Moving it into a GLSL vertex
  shader is a drop-in optimization when needed.
- Mesh-based warping means the warp is piecewise-linear within each cell, so the
  **Mesh resolution** control is a real knob: coarse meshes show visible faceting
  of the warp (and the texture that rides on it); fine meshes are smooth. The grid
  overlay auto-subsamples to ~24 lines so it stays readable at any resolution.
  (A fragment-shader inverse map would remove the faceting entirely — a later
  option if wanted.)
