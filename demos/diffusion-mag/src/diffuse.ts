// diffuse.ts — area-based diffusion solver, a TypeScript port of FAD's
// MeshDiff / MeshMag (Keahey, IEEE InfoVis 1996).
//
// The idea (the data-driven inverse of the closed-form fisheye): you specify a
// desired *magnification field* z(node) — how much each region should be
// enlarged — and the solver iteratively displaces a mesh of points so that each
// node's local AREA RATIO (current area / rest area) approaches its target z.
//
// Each sweep is one relaxation step: measure the magnification error at a node,
// damp + clamp it, and "diffuse" it to the four neighbours by physically
// pushing them apart (under-magnified) or pulling them together (over-
// magnified). This is error diffusion (Floyd–Steinberg in spirit) applied to
// mesh geometry, and equivalently a damped nonlinear Gauss–Seidel relaxation of
// a prescribed-Jacobian (Poisson) problem. See ../../NLM-ORIENTATION.md §5.

export interface DiffMesh {
  N: number; // nodes per side
  x: Float32Array; // current positions, length N*N, in [-1,1]
  y: Float32Array;
  z: Float32Array; // desired magnification per node (1 = none)
  x0: Float32Array; // rest (home) positions
  y0: Float32Array;
}

export interface DiffParams {
  refineCoeff: number; // step damping (FAD default ~0.3) — trades speed for stability
  clampEps: number; // per-step displacement clamp (~0.1)
  weightByMag: number; // 0 none · 1 weight by z · -1 weight by 1/z
  pinBoundary: boolean; // hold the outer ring fixed (keeps the frame)
}

export const DEFAULT_DIFF: DiffParams = {
  refineCoeff: 0.3,
  clampEps: 0.1,
  weightByMag: 1,
  pinBoundary: true,
};

export function makeDiffMesh(N: number): DiffMesh {
  const x = new Float32Array(N * N);
  const y = new Float32Array(N * N);
  const x0 = new Float32Array(N * N);
  const y0 = new Float32Array(N * N);
  const z = new Float32Array(N * N);
  z.fill(1);
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const i = r * N + c;
      const px = (c / (N - 1)) * 2 - 1;
      const py = (r / (N - 1)) * 2 - 1;
      x[i] = x0[i] = px;
      y[i] = y0[i] = py;
    }
  }
  return { N, x, y, z, x0, y0 };
}

export function resetPositions(m: DiffMesh): void {
  m.x.set(m.x0);
  m.y.set(m.y0);
}
export function clearField(m: DiffMesh): void {
  m.z.fill(1);
}

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
const isBoundary = (r: number, c: number, N: number) =>
  r === 0 || c === 0 || r === N - 1 || c === N - 1;

// Reflected node access so boundary quads don't collapse when not pinned.
function rx(m: DiffMesh, r: number, c: number): number {
  const N = m.N;
  if (r < 0) r = -r; else if (r >= N) r = 2 * N - 2 - r;
  if (c < 0) c = -c; else if (c >= N) c = 2 * N - 2 - c;
  return m.x[r * N + c];
}
function ry(m: DiffMesh, r: number, c: number): number {
  const N = m.N;
  if (r < 0) r = -r; else if (r >= N) r = 2 * N - 2 - r;
  if (c < 0) c = -c; else if (c >= N) c = 2 * N - 2 - c;
  return m.y[r * N + c];
}

const dist = (ax: number, ay: number, bx: number, by: number) => Math.hypot(ax - bx, ay - by);

// FAD QuadMetricArea: (|W|+|E|) * (|N|+|S|) of neighbour distances.
function quadArea(m: DiffMesh, r: number, c: number): number {
  const i = r * m.N + c;
  const px = m.x[i], py = m.y[i];
  const dW = dist(px, py, rx(m, r, c - 1), ry(m, r, c - 1));
  const dE = dist(px, py, rx(m, r, c + 1), ry(m, r, c + 1));
  const dN = dist(px, py, rx(m, r + 1, c), ry(m, r + 1, c));
  const dS = dist(px, py, rx(m, r - 1, c), ry(m, r - 1, c));
  return (dW + dE) * (dN + dS);
}

// FAD MoveNeighbour: push the neighbour outward (err>0) or pull it inward
// (err<0). "Outward" is defined by the next node further out along the spoke.
function moveNeighbour(
  m: DiffMesh, r: number, c: number, dr: number, dc: number, err: number, pin: boolean,
): void {
  const N = m.N;
  const r2 = r + dr, c2 = c + dc;
  if (r2 < 0 || r2 >= N || c2 < 0 || c2 >= N) return;
  if (pin && isBoundary(r2, c2, N)) return; // pinned neighbour doesn't move
  const i = r * N + c, j = r2 * N + c2;
  let dx: number, dy: number;
  if (err > 0) {
    // push away: along (next-out − neighbour)
    const r3 = r + 2 * dr, c3 = c + 2 * dc;
    if (r3 < 0 || r3 >= N || c3 < 0 || c3 >= N) return;
    const k = r3 * N + c3;
    dx = (m.x[k] - m.x[j]) * err;
    dy = (m.y[k] - m.y[j]) * err;
  } else {
    // pull closer: toward the centre node (err<0 ⇒ moves toward i)
    dx = (m.x[j] - m.x[i]) * err;
    dy = (m.y[j] - m.y[i]) * err;
  }
  m.x[j] += dx;
  m.y[j] += dy;
}

// One relaxation sweep. `iter` alternates the sweep direction (serpentine) to
// cancel raster bias, exactly as FAD's `toggle`.
export function diffuseStep(m: DiffMesh, p: DiffParams, iter: number): void {
  const N = m.N;
  const h = 2 / (N - 1);
  const refArea = 4 * h * h; // (h+h)*(h+h) of the rest grid
  const eps = p.clampEps;
  const pin = p.pinBoundary;

  const fwd = (iter & 1) === 0;
  const rStart = fwd ? 0 : N - 1, rEnd = fwd ? N : -1, rStep = fwd ? 1 : -1;
  const cStart = fwd ? 0 : N - 1, cEnd = fwd ? N : -1, cStep = fwd ? 1 : -1;

  for (let r = rStart; r !== rEnd; r += rStep) {
    for (let c = cStart; c !== cEnd; c += cStep) {
      if (pin && isBoundary(r, c, N)) continue;
      const i = r * N + c;
      const area = quadArea(m, r, c);
      if (area <= 1e-12) continue;
      const achieved = area / refArea;
      let error = m.z[i] / achieved - 1; // >0 under-magnified, <0 over

      if (p.weightByMag === 1) error *= m.z[i];
      else if (p.weightByMag === -1) error /= m.z[i];

      error *= p.refineCoeff;
      error = clamp(error, -eps, eps);
      if (error === 0) continue;

      const px = m.x[i], py = m.y[i];
      const dW = dist(px, py, rx(m, r, c - 1), ry(m, r, c - 1));
      const dE = dist(px, py, rx(m, r, c + 1), ry(m, r, c + 1));
      const dN = dist(px, py, rx(m, r + 1, c), ry(m, r + 1, c));
      const dS = dist(px, py, rx(m, r - 1, c), ry(m, r - 1, c));
      const sum = dW + dE + dN + dS;
      if (sum <= 1e-12) continue;
      const e = error / sum;

      let wW: number, wE: number, wN: number, wS: number;
      if (e > 0) {
        // push: nearer neighbours get the larger share
        wW = e * (sum - dW); wE = e * (sum - dE);
        wN = e * (sum - dN); wS = e * (sum - dS);
      } else {
        // pull: farther neighbours get the larger share
        wW = e * dW; wE = e * dE; wN = e * dN; wS = e * dS;
      }
      moveNeighbour(m, r, c, 0, -1, clamp(wW, -eps, eps), pin);
      moveNeighbour(m, r, c, 0, 1, clamp(wE, -eps, eps), pin);
      moveNeighbour(m, r, c, 1, 0, clamp(wN, -eps, eps), pin);
      moveNeighbour(m, r, c, -1, 0, clamp(wS, -eps, eps), pin);
    }
  }
}

// RMS of the magnification error over the interior — a convergence gauge.
export function rmsError(m: DiffMesh): number {
  const N = m.N;
  const h = 2 / (N - 1);
  const refArea = 4 * h * h;
  let acc = 0, n = 0;
  for (let r = 1; r < N - 1; r++) {
    for (let c = 1; c < N - 1; c++) {
      const i = r * N + c;
      const achieved = quadArea(m, r, c) / refArea;
      if (achieved <= 1e-9) continue;
      const e = m.z[i] / achieved - 1;
      acc += e * e;
      n++;
    }
  }
  return n ? Math.sqrt(acc / n) : 0;
}

// Soft circular brush onto the magnification field, in the mesh's CURRENT
// (deformed) coordinate space. `delta` adds magnification; `target` (e.g. 1 for
// erase) pulls toward a value. Returns true if anything changed.
export function paintField(
  m: DiffMesh, cx: number, cy: number, radius: number,
  delta: number, minZ: number, maxZ: number, mode: "magnify" | "minify" | "erase",
): boolean {
  const N = m.N;
  const r2 = radius * radius;
  let changed = false;
  for (let i = 0; i < N * N; i++) {
    const dx = m.x[i] - cx, dy = m.y[i] - cy;
    const d2 = dx * dx + dy * dy;
    if (d2 > r2) continue;
    const falloff = 1 - Math.sqrt(d2) / radius; // soft edge
    if (mode === "erase") {
      m.z[i] += (1 - m.z[i]) * Math.min(1, delta * falloff * 4);
    } else {
      const s = mode === "minify" ? -1 : 1;
      m.z[i] = clamp(m.z[i] + s * delta * falloff, minZ, maxZ);
    }
    changed = true;
  }
  return changed;
}
