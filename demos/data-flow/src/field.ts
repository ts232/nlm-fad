// field.ts — data objects ("aircraft") + data-driven magnification fields.
// Ports FAD's apps/2d data model and MeshMag::ComputeDensityToZ /
// ComputeProximityToZ: the magnification field is DERIVED FROM DATA (where the
// traffic is), not painted. That same field then drives the diffusion solver.

import type { DiffMesh } from "./diffuse";

// ---------------------------------------------------------------------------
// Moving data objects (planes) — position + velocity in the [-1,1] zone.
// ---------------------------------------------------------------------------
export interface Planes {
  n: number;
  x: Float32Array; // position in [-1,1]
  y: Float32Array;
  vx: Float32Array; // velocity (per second)
  vy: Float32Array;
}

// Deterministic pseudo-random (no Math.random — keeps things reproducible).
function mulberry(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makePlanes(n: number, seed = 1): Planes {
  const rnd = mulberry(seed);
  const x = new Float32Array(n), y = new Float32Array(n);
  const vx = new Float32Array(n), vy = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = (rnd() * 2 - 1) * 0.9;
    y[i] = (rnd() * 2 - 1) * 0.9;
    const ang = rnd() * Math.PI * 2;
    const spd = 0.05 + rnd() * 0.12;
    vx[i] = Math.cos(ang) * spd;
    vy[i] = Math.sin(ang) * spd;
  }
  return { n, x, y, vx, vy };
}

// Advance planes; bounce off the zone edges.
export function stepPlanes(p: Planes, dt: number, speed: number): void {
  const lim = 0.96;
  for (let i = 0; i < p.n; i++) {
    p.x[i] += p.vx[i] * speed * dt;
    p.y[i] += p.vy[i] * speed * dt;
    if (p.x[i] > lim) { p.x[i] = lim; p.vx[i] = -Math.abs(p.vx[i]); }
    else if (p.x[i] < -lim) { p.x[i] = -lim; p.vx[i] = Math.abs(p.vx[i]); }
    if (p.y[i] > lim) { p.y[i] = lim; p.vy[i] = -Math.abs(p.vy[i]); }
    else if (p.y[i] < -lim) { p.y[i] = -lim; p.vy[i] = Math.abs(p.vy[i]); }
  }
}

// ---------------------------------------------------------------------------
// Data → magnification field. Both write the target field into mesh.z, defined
// on the REST grid (material space), so it's a stable function of the data.
// ---------------------------------------------------------------------------

// Kernel-density magnification. FAD's ComputeDensityToZ bumped only the 2×2 cell
// under each plane by volume/N — far too weak/tiny to deform a 64-grid (an
// isolated plane gave z≈1.1). Instead we splat each plane as a soft blob of the
// given `radius` (a kernel-density estimate) and *sum* contributions, so an
// isolated plane already reaches z = 1 + volume and clusters pile up higher.
// Capped so dense clusters don't chase impossible targets.
export function computeDensityField(m: DiffMesh, p: Planes, volume: number, radius: number): void {
  const N = m.N;
  const h = 2 / (N - 1);
  const r2 = radius * radius;
  const Z_CAP = 8;
  for (let r = 0; r < N; r++) {
    const ny = r * h - 1;
    for (let c = 0; c < N; c++) {
      const nx = c * h - 1;
      let sum = 0;
      for (let k = 0; k < p.n; k++) {
        const dx = nx - p.x[k], dy = ny - p.y[k];
        const d2 = dx * dx + dy * dy;
        if (d2 < r2) { const w = 1 - Math.sqrt(d2) / radius; sum += w * w; } // smooth (quadratic) splat
      }
      const z = 1 + sum * volume;
      m.z[r * N + c] = z > Z_CAP ? Z_CAP : z;
    }
  }
}

// FAD MeshMag::ComputeProximityToZ — each node's z from its distance to the
// nearest plane: z = 1 + tanh(maxDist − minDist)·volume within maxDist, else 1.
export function computeProximityField(m: DiffMesh, p: Planes, volume: number, maxDist: number): void {
  const N = m.N;
  const h = 2 / (N - 1);
  for (let r = 0; r < N; r++) {
    const ny = r * h - 1;
    for (let c = 0; c < N; c++) {
      const nx = c * h - 1;
      let minDist = Infinity;
      for (let k = 0; k < p.n; k++) {
        const dx = nx - p.x[k], dy = ny - p.y[k];
        const d = dx * dx + dy * dy; // compare squared; sqrt once below
        if (d < minDist) minDist = d;
      }
      minDist = Math.sqrt(minDist);
      m.z[r * N + c] = minDist > maxDist ? 1 : 1 + Math.tanh(maxDist - minDist) * volume;
    }
  }
}

// ---------------------------------------------------------------------------
// Map a material [-1,1] position onto the deformed mesh (bilinear) — used to
// draw each plane riding the warp, so dense clusters visibly spread apart.
// ---------------------------------------------------------------------------
export function warpLookup(m: DiffMesh, px: number, py: number, out: [number, number]): void {
  const N = m.N;
  const gx = (px + 1) * 0.5 * (N - 1);
  const gy = (py + 1) * 0.5 * (N - 1);
  let c = Math.floor(gx), r = Math.floor(gy);
  c = c < 0 ? 0 : c > N - 2 ? N - 2 : c;
  r = r < 0 ? 0 : r > N - 2 ? N - 2 : r;
  const fx = gx - c, fy = gy - r;
  const i00 = r * N + c, i10 = i00 + 1, i01 = i00 + N, i11 = i01 + 1;
  const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
  out[0] = m.x[i00] * w00 + m.x[i10] * w10 + m.x[i01] * w01 + m.x[i11] * w11;
  out[1] = m.y[i00] * w00 + m.y[i10] * w10 + m.y[i01] * w01 + m.y[i11] * w11;
}
