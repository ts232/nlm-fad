// warp.ts — closed-form nonlinear magnification (2D), a TypeScript port of the
// FAD toolkit's `trans2D` core (Keahey, "Nonlinear Magnification", 1995–1997).
//
// This is deliberately small and dependency-free: it is the seed of the ported
// library. The structure mirrors FAD's vocabulary —
//
//     point --> [round domain] --> [radial kernel] --> [filter blend] --> point
//
// Everything here is a pure analytic function of a single point, so it is
// embarrassingly parallel (the same math can live in a GLSL vertex shader).
// See NLM-ORIENTATION.md for the math and provenance.

export interface Vec2 {
  x: number;
  y: number;
}

// ---------------------------------------------------------------------------
// Radial kernels — FAD `Warp.h` primitives (file:line in the orientation doc).
// Each maps a normalized radius x>=0 to a warped radius, parameterized by
// `beta` (magnification strength). All are applied to r in the unit domain.
// ---------------------------------------------------------------------------
export const kernels = {
  // hyperWarp(x, b) = tanh(x*b)            — the FAD default ("continuous radial")
  hyper: (x: number, beta: number) => Math.tanh(x * beta),
  // fisheye(x, b)  = (1+b)x / (b*x + 1)    — Sarkar & Brown's rational fisheye (CACM 1994), as used in FAD; equals 1 at x=1
  fisheye: (x: number, beta: number) => ((1 + beta) * x) / (beta * x + 1),
  // logisticHyper(x, b) = 2/(1+e^-2bx) - 1 — sigmoid variant, steeper centre
  logistic: (x: number, beta: number) => 2 / (1 + Math.exp(-2 * beta * x)) - 1,
} as const;

export type KernelName = keyof typeof kernels;

export interface FisheyeParams {
  /** Magnification centre, in normalized [-1,1] space. */
  focus: Vec2;
  /** Magnification strength. 0 = identity; larger = stronger. */
  beta: number;
  /** Round-domain radius: the warp fades to identity at this distance from focus. */
  gamma: number;
  /** Global blend 0 = identity .. 1 = full transform (FAD's smoothing filter weight). */
  filterWeight: number;
  /** Which radial kernel to use. */
  kernel: KernelName;
  /** canvasWidth/canvasHeight — keeps the lens circular on a non-square canvas (1 = square). */
  aspect: number;
  /** Flat (constant-magnification) centre, as a fraction of the feasible max; 0 = pure curve. */
  linearCenter: number;
}

export const DEFAULT_PARAMS: FisheyeParams = {
  focus: { x: 0, y: 0 },
  beta: 3,
  gamma: 0.6,
  filterWeight: 1,
  kernel: "fisheye",
  aspect: 1,
  linearCenter: 0,
};

// ---------------------------------------------------------------------------
// radialTransfer — the boundary-normalized radial profile r'(rc) for rc in
// [0,1] (r'(0)=0, r'(1)=1), with an optional flat (constant-magnification)
// centre.
//
// This is the piecewise spirit of FAD's WarpStep family: magnification is a
// *function* you can shape, and the warp is its integral. Here the centre is a
// linear segment of constant magnification m0 — a uniform "magnifying glass"
// zone that stays undistorted/legible — spliced continuously onto the chosen
// kernel's fisheye falloff. `linearCenter` in [0,1] sets the plateau width as a
// fraction of the feasible maximum; 0 reproduces the pure kernel exactly.
// ---------------------------------------------------------------------------
export function radialTransfer(
  rc: number,
  kernel: KernelName,
  beta: number,
  linearCenter: number,
): number {
  const K = kernels[kernel];
  const norm = K(1, beta) || 1;
  const t = (x: number) => K(x, beta) / norm; // normalized transfer, t(1)=1
  const eps = 1e-4;
  const m0 = t(eps) / eps; // centre magnification = slope of t at 0
  const linMax = 0.9 / Math.max(m0, 1e-6); // a plateau can't out-fill the domain
  const lin = Math.max(0, Math.min(1, linearCenter)) * linMax;
  if (rc <= lin) return m0 * rc; // flat centre: constant magnification m0
  const tlin = t(lin);
  return m0 * lin + (1 - m0 * lin) * ((t(rc) - tlin) / (1 - tlin));
}

// ---------------------------------------------------------------------------
// warpPoint — transform one point through a single radial focus.
//
// FAD pipeline, specialized to one Transform = (radial Warp + round Domain):
//   1. translate so focus is the origin; measure world radius
//   2. if outside the round domain (r >= gamma), leave the point untouched
//   3. map radius into the unit domain (rc = r/gamma)
//   4. apply the radial kernel, *boundary-normalized* so the profile returns 1
//      at the domain edge (rc=1) — this keeps the lens seamless at its rim.
//      For the `fisheye` kernel the normalizer is 1, so this is exactly FAD.
//   5. scale the offset by rPrime/rc and translate back
//   6. blend toward the result by filterWeight
// ---------------------------------------------------------------------------
export function warpPoint(p: Vec2, params: FisheyeParams, out: Vec2 = { x: 0, y: 0 }): Vec2 {
  const { focus, beta, gamma, filterWeight, kernel, aspect, linearCenter } = params;

  const dx = p.x - focus.x;
  const dy = p.y - focus.y;
  // Aspect-corrected radius: with a non-square canvas a clip-space circle would
  // appear as an ellipse, so we measure radius in on-screen-isotropic space
  // (aspect = canvasW/canvasH). The point is still displaced along its raw
  // (dx,dy); only the magnification *amount* is governed by the screen radius.
  const r = Math.hypot(dx, dy / aspect);

  // Outside the domain, at the exact centre, or with no magnification: identity.
  if (r === 0 || r >= gamma || beta === 0 || filterWeight === 0) {
    out.x = p.x;
    out.y = p.y;
    return out;
  }

  const rc = r / gamma; // radius in the unit (round) domain, (0, 1)
  // Boundary-normalized radial profile, with an optional flat (linear) centre.
  const rPrime = radialTransfer(rc, kernel, beta, linearCenter);
  const scale = rPrime / rc; // >1 near the focus => local magnification

  // Warped world position (gamma cancels: (d/gamma * scale) * gamma = d * scale).
  const wx = focus.x + dx * scale;
  const wy = focus.y + dy * scale;

  out.x = p.x + (wx - p.x) * filterWeight;
  out.y = p.y + (wy - p.y) * filterWeight;
  return out;
}

// ---------------------------------------------------------------------------
// A regular point lattice over [-1,1]^2 — FAD's PointGrid. `uv` holds each
// vertex's home position (0..1), used as the texture coordinate; `pos` holds
// the live warped position (clip space), recomputed by `warpLattice`.
// ---------------------------------------------------------------------------
export interface Lattice {
  /** vertices per side (n+1 along each axis for n cells). */
  n: number;
  /** home/texture coords, length (n+1)^2 * 2, in [0,1]. */
  uv: Float32Array;
  /** warped positions, length (n+1)^2 * 2, in [-1,1] clip space. */
  pos: Float32Array;
}

export function makeLattice(n: number): Lattice {
  const side = n + 1;
  const count = side * side;
  const uv = new Float32Array(count * 2);
  const pos = new Float32Array(count * 2);
  let i = 0;
  for (let row = 0; row < side; row++) {
    for (let col = 0; col < side; col++) {
      const u = col / n; // 0..1
      const v = row / n; // 0..1
      uv[i] = u;
      uv[i + 1] = v;
      pos[i] = u * 2 - 1; // -1..1
      pos[i + 1] = v * 2 - 1;
      i += 2;
    }
  }
  return { n, uv, pos };
}

const _scratch: Vec2 = { x: 0, y: 0 };
const _p: Vec2 = { x: 0, y: 0 };

/** Recompute every warped lattice position in-place from its home coords. */
export function warpLattice(lat: Lattice, params: FisheyeParams): void {
  const { uv, pos } = lat;
  for (let i = 0; i < uv.length; i += 2) {
    _p.x = uv[i] * 2 - 1;
    _p.y = uv[i + 1] * 2 - 1;
    warpPoint(_p, params, _scratch);
    pos[i] = _scratch.x;
    pos[i + 1] = _scratch.y;
  }
}
