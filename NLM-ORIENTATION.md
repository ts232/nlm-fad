---
title: FAD / Nonlinear Magnification — Orientation
type: research
project: entoptica
status: draft
created: 2026-06-20
updated: 2026-06-20
tags: [nlm, magnification, fisheye, focus-context, porting, dataviz, numerical-methods]
---

# FAD / Nonlinear Magnification — Orientation

An orientation to the **FAD toolkit** — the C/C++ implementation of nonlinear magnification from T. Alan Keahey's Indiana University PhD work (1995–1997, commercialized as Magnivista 1997–2005) — written to plan a port to the [[Entoptica]] stack (TypeScript + WebGL2 + Vite). It maps what is in the tree, extracts the math, gives a **primer on the numerical methods** the data-driven solver is an instance of, places the work in its **related-work lineage**, and records the first porting beachhead. Companion to the running demo in [`demos/fisheye-2d/`](demos/fisheye-2d/README.md). See also the engine vision in [[THEORY]] and [[FIELD-THEORY]], indexed from [[HOME]].

> status: `draft` — generated from a code orientation pass, not yet human-reviewed. The math citations were spot-checked against source; the related-work citations are from memory and should be verified before any are cited in a paper.

> **Built so far (checkpoint 2026-06-20).** Both engines are working across **four standalone demos** — `fisheye-2d` (closed-form, port 1995), `diffusion-mag` (paint-a-field, 1996), `data-flow` (data-driven planes, 1997), `face-magnify` (face-landmark features, 1998). The arc is one solver + one field module with a **swappable field source** (paint → data → image features). Per-demo write-ups in §8; project conventions in `CLAUDE.md`. The §5 numerical-methods primer and §6 related work are the durable reference; §8–9 track the build and open questions.

---

## 1. What FAD is, and where it lives

`Experiments/nlm/FAD` is a symlink into iCloud (`~/Library/Mobile Documents/.../Magnivista/src/FAD`). It is the **Magnivista FAD SDK** — a "RISC-style" engine for nonlinear magnification in which complex magnification transformations are composed from small, modular warps. The method was first described in IUCS-TR455 (March 1995) and published as Keahey & Robertson, *Techniques for Nonlinear Magnification Transformations*, IEEE InfoVis 1996 (pp. 38–45). Headers are stamped *"04/97 — Alan Keahey (tkeahey@cs.indiana.edu)"*.

The library core is **~29,000 lines** of headers + source (C++), historically coupled to OpenGL/GLUT for display. Repo map of the parts that matter:

| Path | What it is | Lines (src) |
|---|---|---|
| `dev/fad/include` + `lib` | the live library source tree | ~29k |
| `dev/fad/lib/trans{1D,2D,3D}` | **closed-form pipeline magnification** (the analytic warps) | ~8.8k |
| `dev/fad/lib/mag2D`, `mag3D` | **area-based diffusion solver** (data-driven warping) | ~2.2k |
| `dev/fad/lib/struct`, `basic`, `image`, `graphic` | data-structure + math + I/O substrate | ~5.3k |
| `dev/fad/demo`, `apps` | GLUT example programs (`basic`, `image`, `interact`, `2d/3d/theme/texture`) | — |
| `FAD-CG-MagGUI/MagGUI` | later Windows FLTK + OpenGL + NVIDIA Cg GUI (GPU magnification) | — |
| **`macos-port/xcode/FAD_2_src`** | **2011 headless port — all graphics stripped** (see §7) | ~1.5k |
| `dist/sdk-dist-1.1.{2,5}` | packaged Windows SDK + demo binaries | — |
| `doc/fad-doc.html` | the SDK programming manual (pipeline diagram, API) | — |

There are **two distinct magnification engines** here, and the distinction is the key to the port.

---

## 2. Engine A — closed-form pipeline magnification (`trans2D`)

The "RISC" engine from the manual: a per-point analytic function built by composing simple parts. It is **stateless and embarrassingly parallel** — the same math runs per-pixel in a shader.

**Five radial kernels** (`include/trans2D/Warp.h:141–172`, verified):

| kernel | formula | note |
|---|---|---|
| `hyperWarp(x,β)` | `tanh(x·β)` | the FAD default ("continuous radial") |
| `fisheye(x,β)` | `(1+β)·x / (β·x + 1)` | Keahey's rational fisheye; `=1` at `x=1` (rim-continuous) |
| `logisticHyper(x,β)` | `2/(1+e^(−2βx)) − 1` | sigmoid, steeper centre |
| `invHyper(x,β)` | `atan(x)/β` | inverse, for domain inversion |
| `tanhSin(x,β)` | `tanh(x·β) − sin(tanh(x·β))` | specialty |

**Warp** applies a kernel to a coordinate. The radial case is the whole idea in three lines (`lib/trans2D/WarpContRadial.cpp:38–47`): `r = ‖p‖; r' = hyperWarp(r, βₓ); p · (r'/r)`. Variants cover ortho (per-axis), vertical, horizontal, bi-radial, plus *Fisheye*, *Logistic*, and *Flat* (constant-magnification inner core) flavours.

**Domain** is the region of effect — `infinite / round / rect / vertical` — parameterized by `gamma`. It maps world → canonical `[−1,1]²` (so the warp always works in a unit frame), and back. Round domain: in-domain iff `‖p − centre‖ < γ`.

**Transform = Warp + Domain.** A `Transform` does `domain.toDomain → warp → domain.fromDomain` (`lib/trans2D/Transform.cpp:68–99`).

**TransformList** stacks multiple foci and combines them via a **mux mode**: `topOnly`, `average`, `scaledAverage` (inverse-distance weighted), `clipRay` (nearest centre + ray-clip so a point ends up governed by its own focus), `overlap` (sequential composition).

**TransformPipeline** wraps the stack with two global controls: a **filter weight** `w ∈ [0,1]` that blends `out = (1−w)·p + w·warped` (`0` = identity, `1` = full transform — this is "legibility as a dial" in [[FIELD-THEORY]] terms), and a **dest-range** policy (`none / clamp / constrain / maximal`) that keeps transformed output inside the view frame.

End-to-end: `point → [coord-frame map] → mux of Transforms (each: domain→warp→domain) → filter blend → fit-range → point`. **This engine is what the demo ports (§8).**

> A second, **piecewise-linear** family (`WarpStep*`) exists: precompute the warp on a regular grid (default 96×96) and interpolate, for fast preview. The manual flags its API as unstable; the 2011 port deferred it. Skip it for now.

---

## 3. Engine B — area-based diffusion (`mag2D` / `mag3D`)

The data-driven inverse, and the part you described as *"arbitrary warping based on diffusion models for solving integral equations."* Instead of choosing a lens function, you **specify a desired magnification field** and the solver **finds the mesh displacement that realizes it**. There is no explicit integral solver in the code — the "integral equation" is the informal description of what the area-based diffusion approximates (the thesis carries the formal calculus). §5 unpacks exactly what numerical method this is.

**Specifying the field.** `MeshMag` (`include/mag2D/MeshMag.h`) is a grid carrying a scalar magnification `z` per node. It can be built from magnification primitives (`MagPrimCircleInc::ApplyToMesh` adds an increment inside a circle, clamped to a max), from data **density** (points-per-cell), or from **proximity** (a `tanh` falloff around features). So the "desired magnification" is an arbitrary scalar field painted onto the grid.

**The solver.** `MeshDiff : Mesh` (`include/mag2D/MeshDiff.h`, read in full) drives an iterative routine `Diffuse(flags, iters, stride)` → `EnergyDiffuse()` (`lib/mag2D/MeshDiff.cpp`). One sweep, per node:

1. **Measure the achieved magnification** as the local *area ratio*: `Area(row,col,stride)` of the deformed quad vs. a reference cell area `AreaPrimeInv`.
2. **Error = mismatch with the target:** `Error = (z_desired / (Area · AreaPrimeInv)) − 1` (a normalized form `√z/√area − 1` is used in `ComputeRMSE`). Error `> 0` ⇒ under-magnified here; `< 0` ⇒ over-magnified.
3. **Damp and clamp:** `Error *= refineCoeff` (≈0.3) then `clamp(±0.1)` — optionally weight by magnification (`weightByMag`).
4. **Distribute to the 4 neighbours, distance-weighted**, then physically **displace** each neighbour (`MoveNeighbour`, inline at `MeshDiff.h:111–158`): if under-magnified, **push neighbours apart** (referencing the next neighbour out); if over-magnified, **pull them in**. An `ortho` flag restricts motion to an axis; `FlagGrid` values pin boundary motion.

Controls: `iters`, `stride` (coarse→fine multi-scale), `refineCoeff` (damping), `zClip`/`clipPlane` (ignore small errors / low-mag regions), `weightByMag`, alternating sweep direction (`toggle`) to cancel raster bias. `ComputeRMSE` gives a convergence metric but there is **no automatic stopping** — you run a fixed iteration budget.

**3D.** `DiffField3D` / `MagField3D` (`mag3D`) is the volumetric analog: a 6-neighbour stencil, a **volume ratio** `(De+Dw)(Dn+Ds)(Df+Db)/cell` instead of area, no weighting/ortho refinements. Same algorithm, one dimension up.

---

## 4. The substrate, and the hidden asset

**The math substrate is tiny.** `Point2f`/`Point3f` (vector ops), `Grid`/`Grid3D` (row-major index metadata), `PointGrid`/`PointGrid3D`, `Mesh` (a PointGrid3D + z-statistics), `List`, and a handful of math utilities (`Clamp`, `Sign`, …). **~1,100 lines, zero platform dependency** once rendering is stripped. Everything else — `Field`/`ScalarField3D`/`TriMesh`, PPM image I/O, the `graphic/` GL helpers — is either niche or replaced wholesale on the web.

**The asset (§7 in full):** `macos-port/xcode/FAD_2_src/` is a **2011 macOS/iOS port that already stripped all OpenGL** and renamed everything `FAD*`. Its README: *"All graphics commands are being removed during the port."* It carried forward the **entire closed-form pipeline + substrate + `MeshMag`**, and deferred only the `WarpStep*` and the `mag` diffusion solver (`FAD_2_src_deferred/`). The hard part of any port — decoupling the math from the renderer — **is already done for Engine A.**

---

## 5. Numerical methods — a primer

The data-driven solver (§3) borrows its structure from two classical families: **error diffusion** (from image halftoning) and **relaxation methods for elliptic PDEs** (from numerical analysis). This section grounds both, then shows precisely how FAD's area-diffusion is an instance — and where the "integral equation" language is exactly right.

### 5.1 Error diffusion (Floyd–Steinberg, 1976)

Error diffusion was invented for **dithering**: rendering a continuous-tone image on a 1-bit (black/white) device while preserving *local average* brightness. Process pixels in a raster sweep. At each pixel you quantize it to the nearest available value; the **quantization error** `e = original − quantized` is then *pushed forward* onto not-yet-visited neighbours with fixed weights, so the brightness you "lost" reappears nearby:

```
            (current)   7/16
     3/16     5/16      1/16     ← Floyd–Steinberg weights (sum = 1)
```

Three properties define the paradigm, and all three reappear in FAD:

- **Conservation.** The error is *moved*, never destroyed — weights sum to 1. The total quantity is preserved; only its spatial distribution changes.
- **Sequential local sweep.** Each node is visited once per pass; corrections use the current state of neighbours.
- **Serpentine scanning.** Alternating the sweep direction each pass (boustrophedon) cancels the directional bias a fixed raster order would bake in.

FAD swaps *tone* for *geometry*. The residual it diffuses is not a quantization error but an **area-magnification error** (achieved vs. desired local area ratio, §3 step 2). "Diffusing" it means **displacing neighbour vertices** rather than nudging a tone value: push apart where area is too small, pull in where too large, distance-weighted (`MoveNeighbour`). The alternating `toggle` sweep is Floyd–Steinberg's serpentine scan. So FAD is *error diffusion by structure* — the sweep-and-redistribute paradigm — applied to a deforming mesh, not the literal 1976 dithering weights.

### 5.2 Relaxation methods (Jacobi, Gauss–Seidel, SOR)

The other lineage is the iterative solution of large sparse linear systems, especially **discretized elliptic PDEs**. The model problem is the **Poisson equation** `∇²u = f`. On a regular grid with spacing `h`, the standard 5-point Laplacian stencil makes each unknown the neighbour-average minus a source:

```
u(i,j) = ¼ · [ u(i+1,j) + u(i−1,j) + u(i,j+1) + u(i,j−1) − h²·f(i,j) ]
```

You don't invert the giant matrix; you **relax** — repeatedly overwrite each node with the value that locally satisfies the stencil:

- **Jacobi** updates every node from the *previous* sweep's values (fully parallel, slow).
- **Gauss–Seidel** updates *in place*, using neighbours already refreshed in the current sweep (≈2× faster, inherently sequential — like a raster error-diffusion sweep).
- **SOR** (successive over-relaxation) *overshoots* each correction by `ω ∈ (1,2)` to accelerate convergence.

The quantity that decays over sweeps is the **residual** — how far the discrete equation is from being satisfied. FAD's `ComputeRMSE` is exactly such a residual norm over the magnification constraint. FAD's `refineCoeff ≈ 0.3` is the opposite of SOR's overshoot: it is **under-relaxation / damping** — trading speed for stability, because the constraint is *nonlinear* (area is a nonlinear function of vertex positions) and an undamped step diverges. The hard `±0.1` clamp is the tell-tale of a marginally-stable explicit nonlinear relaxation.

### 5.3 The continuous problem, and why "integral equation" fits

Strip away the discretization and FAD solves a **prescribed-Jacobian** problem. Let `φ: Ω → Ω` be the deformation that takes the home grid to the magnified layout. By the multivariable change-of-variables theorem, the factor by which `φ` scales area at a point is the **Jacobian determinant** `|Dφ|`. "Magnification field `m(x,y)`" *is* a prescription on that determinant — this is the Calc III core of the thesis (area element, Jacobian, change of variables, the divergence/Green's theorems):

$$ |D\varphi(x)| \;=\; m(x), \qquad \varphi:\Omega\to\Omega. $$

In the small-distortion, irrotational regime you write `φ = id + ∇u` (deformation as the gradient of a scalar potential), and the determinant condition linearizes to a **Poisson equation** for that potential:

$$ \nabla^2 u \;=\; m - 1 \quad(\text{source} = \text{magnification surplus}). $$

Now the bridge to your phrasing. A Poisson problem has an equivalent **integral form** through its **Green's function** `G` (the fundamental solution — in 2D, the logarithmic kernel):

$$ u(x) \;=\; \int_\Omega G(x,y)\,\big(m(y)-1\big)\,dy. $$

So **"solve the integral equation"** and **"relax the Poisson PDE"** are two faces of one problem. FAD never forms `G` or the matrix — its iterative area-diffusion is the **matrix-free, nonlinear way to evaluate that solution directly on the mesh**. Each sweep is one relaxation step toward the deformation whose Jacobian equidistributes the requested magnification. Your informal "diffusion models for solving integral equations" is, precisely, *an explicit nonlinear relaxation scheme for the Green's-function solution of the prescribed-Jacobian (Poisson) problem.* It approximates — the area-ratio is a discrete proxy for `|Dφ|`, and the push/pull is a first-order proxy for `∇u` — but the lineage is exact.

### 5.4 How the pieces map

| FAD term (`mag2D`) | classical name | role |
|---|---|---|
| desired magnification `z` field | source term / prescribed Jacobian `m` | the right-hand side |
| achieved area ratio `Area·AreaPrimeInv` | discrete `|Dφ|` | current state |
| `Error = z/area − 1` | local **residual** of `∇²u = m−1` | what's minimized |
| distribute error to 4 neighbours + displace | **relaxation sweep** (Gauss–Seidel-like) + **error diffusion** | the update |
| `refineCoeff ≈ 0.3` | **under-relaxation** factor (anti-SOR) | stability |
| alternating `toggle` sweep | serpentine scan | de-bias |
| `stride` coarse→fine | poor-man's **multigrid** | accelerate low-frequency convergence |
| `ComputeRMSE` | residual norm | convergence gauge |

The one genuine gap vs. textbook solvers: no automatic convergence test and no true multigrid V-cycle (just stride coarsening). Both are easy modern upgrades if the port revisits Engine B.

---

## 6. Related work

FAD sits at the intersection of two literatures; grounding both is the point of this section. **Verify these citations before reusing them** — they are from memory.

### 6.1 Distortion-oriented / focus+context displays (the visualization lineage)

- **Furnas 1986**, *Generalized Fisheye Views* (CHI) — the conceptual origin: a degree-of-interest function trades off intrinsic importance against distance from focus. Magnification as *selective emphasis*.
- **Spence & Apperley 1982**, the Bifocal Display — an early focus+context layout.
- **Mackinlay, Robertson & Card 1991**, the Perspective Wall (CHI) — 3D perspective as a smooth detail-in-context fold.
- **Sarkar & Brown 1992/1994**, *Graphical Fisheye Views* (CHI / CACM) — geometric (Cartesian & polar) fisheye transforms on graphs; the closest direct ancestor of FAD's *closed-form* warps.
- **Leung & Apperley 1994**, *A Review and Taxonomy of Distortion-Oriented Presentation Techniques* (ACM ToCHI) — the survey that formalizes magnification functions vs. transfer functions; the frame FAD's "RISC" composition answers.
- **Lamping, Rao & Pirolli 1995**, the Hyperbolic Browser (CHI) — focus+context via hyperbolic geometry.
- **Rao & Card 1994**, the Table Lens (CHI) — focus+context for tabular data.
- **Carpendale, Cowperthwaite & Fracchia 1995–1997**, the **Elastic Presentation Framework** / 3D Pliable Surfaces — the other major *unifying* theory of this era: all these distortions as viewing operations on a deformed surface. The natural theory to read alongside Keahey's.
- **Keahey & Robertson 1996** (InfoVis) and **Keahey 1998**, *The Generalized Detail-In-Context Problem* (InfoVis) — FAD itself. Its distinguishing move is §6.2: rather than a fixed lens, **specify a magnification field and solve for the transformation**.

### 6.2 Numerical lineage (what the data-driven solver actually is)

- **Floyd & Steinberg 1976** — error diffusion / dithering. The sweep-and-redistribute paradigm FAD borrows for geometry (§5.1).
- **Jacobi / Gauss–Seidel / SOR; multigrid (Brandt 1977)** — iterative relaxation for elliptic PDEs (§5.2). FAD's `EnergyDiffuse` is a damped, nonlinear member of this family.
- **Mesh equidistribution & moving-mesh methods** (Huang & Russell, *Adaptive Moving Mesh Methods*) — solve a Poisson/Monge–Ampère equation so a mesh's cell density matches a monitor function. This is *the same problem* as magnification-field warping, in the meshing community's vocabulary.
- **Optimal transport** (Monge–Kantorovich; Monge–Ampère `det(D²ψ) = ρ`) — the rigorous form of prescribed-Jacobian mapping; modern OT mesh generation (Budd & Williams; Weller et al.) and capacity-constrained stippling (Balzer et al. 2009) are FAD's contemporary cousins.

**The one-line placement:** FAD is *graphical-fisheye / EPF-style focus+context* on the display side, implemented for its data-driven mode as a *damped nonlinear Gauss–Seidel relaxation of a prescribed-Jacobian (Poisson) problem* — i.e. the visualization face of equidistribution / optimal-transport meshing.

---

## 7. Prior porting attempts (chronology)

- **`dev/fad`** — the canonical SGI/Irix-era source; ports noted in `LINUX_PORT_NOTES` (SGI image lib, X screen-grab, `gettimeofday`).
- **SDK 0.6 → 1.1.5** (1997–2005) — Windows distributions, `interact`/`basic`/`image` demos, precompiled libs.
- **`FAD-CG-MagGUI`** (~2008) — a Windows **FLTK + OpenGL + NVIDIA Cg** GUI (`GlobalMag`) doing **GPU** magnification via framebuffer objects and Cg shaders, with trackball + zoom-overview. The reference for *"what a GPU implementation looks like"* → WebGL2/WebGPU analog.
- **`macos-port/xcode/FAD_2_src`** (2011) — **the headless port (§4).** Graphics removed, `FAD*`-renamed, closed-form pipeline + substrate intact, `mag`/`WarpStep` deferred. **Start the TS port by transliterating this**, not the SGI tree.

---

## 8. The port — strategy and first beachhead

**Difficulty: lower than 30-year-old code suggests.** Engine A is self-contained analytic functions; the substrate is ~1k lines of plain vector/grid math; and the 2011 port already did the graphics-decoupling for Engine A.

**Language:** TypeScript + WebGL2, to share tooling with [[Entoptica]] (Svelte 5 + Vite + WebGL2) and make the eventual fold-in trivial. Keep the solver (Engine B) behind an interface so it can drop to WASM/Rust or a compute shader later *iff* JS proves too slow at large mesh sizes — not before.

**Sequencing:**
1. **Closed-form interactive fisheye (done — §8.1).** Engine A → live lens on a grid/image.
2. **Diffusion field-solver (prototyped — §8.3).** Engine B → "paint a magnification field, watch the mesh solve." The more novel, less-seen-on-the-web piece; also where the §5 upgrades (convergence test, multigrid) would land.
3. **3D** (`trans3D` / `mag3D`) later.

### 8.1 First beachhead — `demos/fisheye-2d/` ✅

A standalone Vite + TS + WebGL2 demo of Engine A: a radial fisheye focus over a round domain, applied to a deformable lattice rendered as curved **grid lines** and/or a **textured mesh** (the classic FAD model — warp a `PointGrid`, texture-map it). Focus follows the cursor; `β` (magnification), `γ` (domain radius), and filter-weight are live; three kernels (`fisheye`/`hyper`/`logistic`, boundary-normalized for a seamless rim). Textures: the original **DC Metro** demo image (converted from `dcMetro.ppm`), a procedural test pattern, or drag-and-drop.

- **`src/warp.ts`** is the **portable math core** — `kernels`, `warpPoint`, the lattice — with no WebGL/DOM dependency. *This is the file that graduates into Entoptica.* The renderer recomputes the warped lattice on the CPU each change (≈26k vertices, trivial), keeping `warp.ts` the single source of truth; moving it into a GLSL vertex shader is a later drop-in.
- Run: `cd demos/fisheye-2d && npm install && npm run dev`.

### 8.2 Why this belongs in Entoptica

[[FIELD-THEORY]]: *"stateful points that begin on a regular grid and are free to leave it; legibility is a dial (a spring back home)."* FAD's diffusion warping is **literally** that — a regular grid of points displaced off-grid by a field, with `refineCoeff` playing the damping/spring role and the filter-weight playing the legibility dial. Nonlinear magnification is a **principled, invertible build-op** for the field-theory vocabulary: magnification-as-transform, focus as control, a target field as the source. The `theme`/`texture` apps (semantic, color-mapped magnification) are the dataviz bridge already flagged in [[project-field-theory-dataviz]].

### 8.3 Second prototype — `demos/diffusion-mag/` ✅

Engine B as its own prototype (separate control space — you *paint a field*, you don't aim a lens). `src/diffuse.ts` is a faithful, dependency-free port of `MeshDiff::EnergyDiffuse` + `MoveNeighbour`: measure each node's area-ratio error, damp/clamp, diffuse it to the four neighbours (push-apart / pull-together), serpentine sweep, pinned boundary. A WebGL2 renderer runs the CPU solver live and shows the deforming textured mesh + grid lines + a heat overlay of the field, with brush controls (magnify/minify/erase), solver controls (`refineCoeff`, **step cap** `clampEps`, iters/frame, weighting, pin), and an RMS-error readout. Pinned port **1996** (the InfoVis-paper year). Verified: small/modest fields converge to target almost exactly; large high-mag regions plateau by area conservation (correct physics).

**On speed:** the dominant convergence lever is the **step cap** (`clampEps`, the per-sweep displacement clamp) — `0.1 → 0.25` ≈ 2.5× fewer sweeps; defaults are now `refineCoeff 0.5`, `clampEps 0.25` (~2.5–4× faster than the original `0.3/0.1` with `sqrt`-not-`hypot` and 35 sweeps/frame). **Multi-scale (`stride`) was implemented and benchmarked but does *not* help** — this damped/clamped nonlinear relaxation is *rate-limited*, not propagation-limited, so coarse passes add overhead without payoff (kept on `diffuseStep` for completeness; demo runs stride 1). A useful, slightly counter-intuitive result vs. the classic multigrid intuition.

### 8.4 Third prototype — `demos/data-flow/` ✅ (data-driven)

The **data-driven** field source (FAD `apps/2d` + `MeshMag::ComputeDensityToZ / ComputeProximityToZ`): the magnification field is *derived from data*, not painted. Planes fly over a zone; their **density** (a kernel-density splat — FAD's per-2×2-cell bump was far too weak, so each plane splats a soft blob of `radius` and contributions sum) or **proximity** (`z = 1 + tanh(haloR − dist)·volume`) becomes the field; the **same** `diffuse.ts` solver relaxes the mesh to follow the traffic; the planes are bilinear-mapped onto the warp (`warpLookup`) so clusters visibly spread. Reuses Engine B verbatim — **only the field source changes** (`field.ts`). Pinned port **1997**.

### 8.5 Fourth prototype — `demos/face-magnify/` ✅ (image features → field)

The payoff of the whole port: **image features drive the magnification.** A face is detected with **face-api.js** (TinyFaceDetector + 68-landmark net, on **TensorFlow.js's WebGL backend** — *not* WASM; vendored locally, mirroring Entoptica's `src/core/field/faceApi.ts`), the landmarks become the data points for the *same* `field.ts` density/proximity computation, and the *same* `diffuse.ts` solver magnifies where they cluster — so the **eyes, nose and mouth bulge** while the periphery compresses (a detail-in-context caricature, fitting for port **1998** = Keahey's *Generalized Detail-In-Context Problem*). The only change from the planes demo is the field **source**: `detectFaces(image)` → points → `computeDensityField`. The 68 landmarks are **typed by feature** (face-api's `getMouth/getLeftEye/getNose/getJawOutline/…`), so a checkbox set chooses which groups (eyes / brows / nose / mouth / jaw) become the data points — magnify just the eyes, just the mouth, etc. Drop any front-facing face onto the canvas. This is the general template — swap the detector for edges / saliency / corners and nothing downstream moves. (Detection is browser-only; the field maths now *is* headless-verifiable — see below.)

**Correction (feature shapes, `src/featureField.ts`).** The first build fed the landmark *points* to `field.ts` and produced blobby, hard-to-aim faces. Measured with `demos/face-magnify/test/diagnose.ts` against a committed fixture of real landmarks, the point-cloud field was **infeasible**: `pinBoundary` conserves total area, so mean `z` must be 1, and it asked for **1.132** (density: 1.958). The relaxation therefore never converges — RMS error bottoms out at 0.118 around 600 sweeps and then **drifts back up to 0.150**. Three more faults compounded it: the nearest-point weight in `computeProximityField` (`field.ts:106`) makes `z` **jump 0.625 across Voronoi seams** between differently-weighted features; `1 + tanh(maxDist − minDist)·v` is near-linear over these radii, so each landmark contributes a **cone** (gradient discontinuity at the apex, kink at the rim) rather than a smooth bump; and `tanh(0.22) ≈ 0.2165` meant the "strength 6" slider silently delivered **2.28×**, so the control was uninterpretable.

The replacement keeps Engine B and `field.ts` untouched and changes only *where the field comes from* — the seam doing its job. Per feature, the kernel is the **distance to the landmark outline** (a closed ring for eyes/mouth, an open curve for brows/nose/jaw; 0 inside a ring) through a **C¹ smoothstep** skirt, so an eye magnifies as an eye; features combine by a **soft union** `(Σ eᵖ)^(1/p)` (p = 1 additive → large p separate) instead of a nearest-point pick, which is what removes the creases; the field takes an optional binomial blur; and it is **projected onto mean(z) = 1** by subtracting a constant — the least-squares projection onto the feasible set, which preserves every difference between features and pays for the magnified ones by compressing the periphery. That projection is the literal statement of **detail-in-context**, and it is what makes the solve converge: **mean z 1.000, max Δz 0.122, RMS 0.0037 and monotone**, with 0 folded cells in every case. (The first write-up called that "40× smaller than the field it replaced", but the old run asked for 2.28× against the new 1.55×. Driven with **identical targets**, proximity settles at 0.0160 and shapes + balance at 0.0037 — about 4× — and at a strong setting 0.1052 *drifting* vs 0.0228. `diagnose.ts` now runs that fair comparison.) Sliders are now target magnifications (`1.55×`) in 0.01 steps, and they no longer reset the mesh, so the relaxation walks continuously to each new target. **Calibration** makes those units true: the blur and the balance shave 10–15% off each feature's excess (1.55× delivered 1.47×), so `computeFeatureField` re-aims each feature's gain in a few fixed-point passes until the finished field sits at its target on the feature's core (a ring's interior, a curve's top band) — delivered 1.551× for 1.55× and 2.488× for 2.50×, at a small residual cost (0.0037 → 0.0046) because it asks for what used to be shaved off.

Two notes for the fold-in: `balanceField` and `smoothField` are generic (any Engine-B field wants them, `data-flow` included) and are the parts of `featureField.ts` that should graduate into `field.ts` at Entoptica time; and `test/landmarks.sample.json` — one real detection dumped from the browser — is what makes the whole field pipeline testable with `node --experimental-strip-types`.

---

## 9. Open questions / next decisions

1. **Engine B port priority** — port the diffusion solver next, or push Engine A further first (multi-focus mux modes, image-as-fragment-shader inverse map for full-res magnification)?
2. **Solver upgrades** — when Engine B is ported, adopt a real convergence test + multigrid V-cycle (§5.4 gap), or keep the original's fixed-budget behaviour for fidelity?
3. **Thesis cross-check** — worth pulling the formal derivation from the thesis into §5.3 (replace the linearized sketch with your actual Jacobian/area-element argument)?
4. **Fold-in shape** — standalone demos indefinitely, or define the `warp.ts` → Entoptica `src/core` build-op interface now so each demo lands as an op?

## 10. Publishing & IP — what the browser exposes

Anything that executes client-side is downloaded to the viewer's machine, so a determined reader can always recover the algorithm. You can raise the *cost* of copying; you cannot prevent it. The only hard boundary is keeping the secret on a server and shipping only its output. Layers, weakest → strongest: **minify/bundle** (what `vite build` does — mangles names, strips whitespace; a speed bump); **source maps** (reconstruct the original commented source — never ship them); **WASM** (raises the floor, still decompilable); and the load-bearing fact for Entoptica — **GLSL shaders are always plaintext**: the browser hands shader source to the GPU as a string, recoverable live via `gl.getShaderSource()` no matter how the JS is obfuscated. So any IP encoded in shader math is exposed by definition when it runs in the browser. The only real protection is **server-side rendering** (run the kernel on a server, stream pixels/data) — at the cost of latency, GPU spend, and interactivity.

For NLM specifically the secrecy question is partly moot: the closed-form fisheye kernels and the area-diffusion relaxation are in the published papers (TR455, InfoVis '96/'98), so the value is in the implementation/integration/brand, not the equations. (Magnivista was the author's company; its licensing is no longer enforced.)

**Done here (tier 1, cheap):** every demo's `vite.config.ts` sets `build.sourcemap: false` + `esbuild.legalComments: 'none'` — no maps reach a published build, and all comments (incl. the FAD `file:line` provenance cites, which stay useful *in source*) are stripped from the bundle. Verified: a `fisheye-2d` build ships no `.map`/`sourceMappingURL` and no FAD/`MeshDiff`/`EnergyDiffuse` strings; only GLSL string-literal comments remain, since shaders are plaintext. The same hardening is in the Entoptica app's config.

**Strategy note — Canvas2D as the public tier.** Because the shader-plaintext hole only exists for the WebGL path, a **Canvas2D build is the better public-facing version**: all the math moves into JS, so the whole thing minifies (and could WASM-compile) *uniformly* — there's no unavoidable plaintext escape hatch. Natural tiering: a Canvas2D "lite" version (public, embeddable, end-to-end obfuscatable, slower) over the full WebGL/GLSL engine (kept private, or server-side, or license-gated). Entoptica already has a Canvas2D prototype lineage to build that on.
