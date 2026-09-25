# NLM-FAD — Nonlinear Magnification, thirty years on

A TypeScript / WebGL2 port of **FAD**, the nonlinear-magnification toolkit I wrote
for my PhD at Indiana University in the mid-1990s (Keahey & Robertson, IEEE
InfoVis 1996 and 1997; Keahey, InfoVis 1998). The original was ~29,000 lines of
C++ for SGI / GLUT and later a commercial SDK (Magnivista, 1995–2005). This repo
rebuilds the two engines at its core as small, dependency-free TypeScript
modules, with four standalone browser demos.

**Live demos:** <https://holisticsofa.ai/nlm-fad/>

| Demo | What it shows | Engine · field source |
|---|---|---|
| [Closed-form fisheye](demos/fisheye-2d/) | Aim a radial lens at a textured grid — three kernels, an optional legible linear centre | A |
| [Paint a field](demos/diffusion-mag/) | Paint where you want magnification and watch a mesh relax to deliver it | B · brush |
| [Traffic over a zone](demos/data-flow/) | Moving data points *become* the field; the mesh follows the traffic | B · data |
| [Magnify a face](demos/face-magnify/) | 68 facial landmarks outline the features; eyes, nose and mouth magnify | B · image features |

## The idea

Focus+context ("fisheye") displays magnify the part you're looking at while
keeping the rest on screen, compressed. Most implementations pick a **lens** — a
fixed function of distance from the focus. FAD separated two things:

- **Engine A — closed-form transformations** (`warp.ts`, ~180 lines). Analytic
  radial kernels on a bounded domain, composed pipeline-style:
  `point → round domain → radial kernel → filter blend → point`. Pure per-point
  functions — fast, shader-friendly, the classic fisheye family.
- **Engine B — magnification fields** (`diffuse.ts` + `field.ts`, ~360 lines).
  Instead of choosing the transformation, **specify how much magnification you
  want, where** — a scalar field over the plane — and *solve* for a mesh that
  delivers it. The solver is an error-diffusion / damped Gauss–Seidel
  relaxation: measure each node's area-ratio error against its target, damp and
  clamp it, push the correction to the neighbours, repeat. In modern vocabulary
  it is a prescribed-Jacobian (equidistribution) problem — the same shape as
  moving-mesh methods and optimal-transport meshing, arrived at from the
  visualization side.

The three Engine-B demos share **one solver** and differ only in **where the
field comes from**: a brush, moving data, or features extracted from an image.
That seam is the whole architecture, and it's the thing the port was built to
show. [`NLM-ORIENTATION.md`](NLM-ORIENTATION.md) has the math, a short
numerical-methods primer, the related work, and the porting notes.

## Run locally

Each demo is a self-contained Vite app with its own pinned port (the year of the
relevant paper), so they never collide and can run side by side.

```bash
cd demos/fisheye-2d    && npm install && npm run dev   # → http://localhost:1995
cd demos/diffusion-mag && npm install && npm run dev   # → http://localhost:1996
cd demos/data-flow     && npm install && npm run dev   # → http://localhost:1997
cd demos/face-magnify  && npm install && npm run dev   # → http://localhost:1998
```

The math cores run headless too — `node --experimental-strip-types some_test.ts`
— which is how the solver findings below were measured.

## Build the site

```bash
scripts/build-site.sh            # → _site/  (base path /nlm-fad/, for GitHub Pages)
SITE_BASE=/ scripts/build-site.sh   # any other base path
```

The GitHub Actions workflow in `.github/workflows/pages.yml` runs the same
script and deploys `_site/` to GitHub Pages.

## What porting it taught

- **The solver's speed lever is the per-step clamp, not the damping.** Raising
  `clampEps` from 0.1 to 0.25 is ≈2.5× faster; the demos run ~2.5–4× faster than
  the original defaults with no visible loss.
- **Multi-scale (coarse-to-fine) relaxation does *not* help here.** Benchmarked
  and disproved: this damped, clamped, nonlinear relaxation is rate-limited, not
  propagation-limited. The `stride` parameter stays on `diffuseStep` for
  completeness; the demos run stride 1.
- **Proximity beats density as a data-driven field.** A bounded halo per point,
  scaled by a per-point weight, is stable. A kernel splat that *sums* makes
  clusters over-deform.
- **A pinned boundary conserves area, so the field's mean must be 1.** Ask for
  more magnification than the frame can pay for and the relaxation does not
  converge — it bottoms out and then drifts back up. Projecting the field onto
  mean `z` = 1 (subtract a constant: the least-squares projection, which keeps
  every difference between features) is both the fix and the area-conservation half
  of detail-in-context (magnified regions paid for by compressing the periphery). At strong settings it is the biggest single lever on the
  face demo (settled residual 0.036 → 0.023, and no drift); see
  [`demos/face-magnify/`](demos/face-magnify/README.md).
- **Faithful first, then honest correction.** Each core cites the original FAD
  routine it ports; where the 1990s algorithm was too weak for modern use, the
  fix is made and the reason is in a comment.

## Papers

- T. A. Keahey and E. L. Robertson. **Techniques for Non-Linear Magnification
  Transformations.** *Proc. IEEE Symposium on Information Visualization (InfoVis
  '96)*, pp. 38–45. Also Indiana University CS Technical Report 455.
- T. A. Keahey and E. L. Robertson. **Nonlinear Magnification Fields.** *Proc.
  IEEE InfoVis '97*, pp. 51–58.
- T. A. Keahey. **The Generalized Detail-In-Context Problem.** *Proc. IEEE
  InfoVis '98*, pp. 44–51.
- T. A. Keahey. *Nonlinear Magnification.* PhD dissertation, Indiana University,
  1998.

To cite this software, see [`CITATION.cff`](CITATION.cff) (GitHub renders it as
"Cite this repository").

## Provenance and license

The TypeScript in this repository is © 2026 T. Alan Keahey and released under
the [MIT License](LICENSE). The original FAD C++ source (Magnivista) is **not**
included; the algorithms it implements are the ones described in the papers
above. `face-api.js` (MIT) and its model weights are vendored under
`demos/face-magnify/public/` so the face demo runs without a CDN.

## Not yet — deliberately

Packaging the cores as an installable library (they are currently copied per
demo, by design, so each demo stays standalone); a headless test suite beyond
the one in [`demos/face-magnify/test/`](demos/face-magnify/test/); the 3D
engines (`trans3D` / `mag3D`); a real convergence test and multigrid for the
solver. Those come later. The point of this release is the two engines and the
seam between them, running where anyone can touch them.
