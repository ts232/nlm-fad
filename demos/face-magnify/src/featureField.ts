// featureField.ts — build a magnification field from face FEATURES rather than
// from a cloud of landmark points.
//
// Why this exists (measured, see test/diagnose.ts against the shipped defaults):
//
//   1. INFEASIBLE TARGET. `pinBoundary` holds the frame, so total area is
//      conserved and the area-weighted mean of z MUST be 1. The point-cloud
//      field asks for mean z = 1.132 (density: 1.958) — 13% more area than the
//      frame has. The solver cannot satisfy it, so it does not converge: RMS
//      error falls to 0.118 by ~600 sweeps and then climbs back to 0.150. That
//      slow drift, not folding, is what reads as "blobby and uncontrollable".
//   2. DISCONTINUOUS FIELD. `computeProximityField` takes the weight of the
//      NEAREST point (field.ts:106), so z jumps wherever the nearest-neighbour
//      identity changes between features of different strength — measured max
//      jump 0.625 between adjacent nodes, across a zero-width Voronoi seam.
//   3. CONE PROFILE. `1 + tanh(maxDist − minDist)·v` is near-linear in distance
//      for the radii used here, so each landmark contributes a cone: gradient
//      discontinuity at the apex and a kink at the rim. 68 cones unioned by
//      nearest-point = a lumpy surface, not a feature.
//   4. OPAQUE UNITS. tanh(0.22) ≈ 0.2165, so a "strength 6" slider actually
//      delivers z ≈ 2.28. The number on the control means nothing.
//
// The fix, in the same Engine-B seam (this file only changes WHERE THE FIELD
// COMES FROM; diffuse.ts is untouched):
//
//   * kernels shaped like the FEATURE, not like a point — distance to the
//     landmark polygon/polyline (0 inside a closed ring), so an eye magnifies
//     as an eye;
//   * a C¹ smoothstep skirt instead of a cone;
//   * a soft union (p-norm) across features instead of nearest-point, so
//     overlapping regions blend continuously — no seams, and one knob that
//     moves between "additive" and "separate";
//   * an optional binomial blur of the field;
//   * projection onto the feasible set (mean z = 1), which is the honest
//     detail-in-context statement: magnifying the features is PAID FOR by
//     compressing the periphery. This is what makes the solver converge.
//
// Units are magnification factors throughout: `mag: 1.5` means "this feature
// should end up 1.5× its area", which is what the UI now shows — and, with
// `calibrate`, what the finished field actually asks for after blur + balance.

export type FeatureKey = "eyes" | "brows" | "nose" | "mouth" | "jaw";

export interface Pt {
  x: number;
  y: number;
}

export interface FeatureShape {
  key: FeatureKey;
  /** One or more landmark polylines (eyes and brows contribute two). */
  polys: Pt[][];
  /** Closed ring (eye, mouth) — the interior is fully inside the kernel. */
  closed: boolean;
  /** Peak magnification for this feature; 1 = leave it alone. */
  mag: number;
}

export interface FeatureFieldParams {
  /** Width of the smooth skirt outside each feature, in material units. */
  falloff: number;
  /** Scales every feature's excess at once (0 = flat field, 1 = as set). */
  master: number;
  /** Soft-union exponent: 1 = additive (overlaps pile up), large = separate. */
  blend: number;
  /** Binomial blur passes over the finished field. */
  smoothPasses: number;
  /** Project onto mean(z) = 1 so the target is actually achievable. */
  balanceArea: boolean;
  /** Lower bound for z when balancing, so no cell is asked to vanish. */
  floor: number;
  /**
   * Re-aim each feature so the FINISHED field (after blur and balance) sits at
   * its target on the feature. Without it the blur and the area budget shave
   * 10–15% off every feature's excess — a 1.55× slider delivered 1.47×.
   */
  calibrate: boolean;
}

export const DEFAULT_FEATURE_FIELD: FeatureFieldParams = {
  falloff: 0.09,
  master: 1,
  blend: 3,
  smoothPasses: 4,
  balanceArea: true,
  floor: 0.35,
  calibrate: true,
};

/**
 * The legacy point-cloud fields' per-point weight for a target magnification.
 * Proximity delivers z = 1 + tanh(radius)·weight at a landmark, so this makes a
 * `mag` slider mean the same thing under all three methods. Density *sums*
 * overlapping splats, so it only approximates the target.
 */
export function legacyWeight(mag: number, radius: number): number {
  return mag <= 1 ? 0 : (mag - 1) / Math.tanh(radius);
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

function distToSegment2(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const vx = bx - ax, vy = by - ay;
  const wx = px - ax, wy = py - ay;
  const vv = vx * vx + vy * vy;
  let t = vv > 1e-20 ? (wx * vx + wy * vy) / vv : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const dx = wx - t * vx, dy = wy - t * vy;
  return dx * dx + dy * dy;
}

// Even-odd crossing test. Landmark rings are simple polygons, so this is enough.
function pointInPoly(px: number, py: number, poly: Pt[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const yi = poly[i].y, yj = poly[j].y;
    if (yi > py !== yj > py) {
      const t = (py - yi) / (yj - yi);
      if (px < poly[i].x + t * (poly[j].x - poly[i].x)) inside = !inside;
    }
  }
  return inside;
}

/** Distance from (px,py) to a polyline; 0 inside a closed one. */
function distToPoly(px: number, py: number, poly: Pt[], closed: boolean): number {
  const n = poly.length;
  if (n === 0) return Infinity;
  if (n === 1) return Math.hypot(px - poly[0].x, py - poly[0].y);
  if (closed && pointInPoly(px, py, poly)) return 0;
  let best = Infinity;
  const last = closed ? n : n - 1;
  for (let i = 0; i < last; i++) {
    const a = poly[i], b = poly[(i + 1) % n];
    const d2 = distToSegment2(px, py, a.x, a.y, b.x, b.y);
    if (d2 < best) best = d2;
  }
  return Math.sqrt(best);
}

/** 1 at the feature, 0 beyond `falloff`, C¹ at both ends (1 − smoothstep). */
function skirt(d: number, falloff: number): number {
  if (d <= 0) return 1;
  if (d >= falloff) return 0;
  const t = d / falloff;
  return 1 - t * t * (3 - 2 * t);
}

interface Box { x0: number; y0: number; x1: number; y1: number }

function bboxOf(polys: Pt[][], pad: number): Box {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const poly of polys) {
    for (const p of poly) {
      if (p.x < x0) x0 = p.x;
      if (p.y < y0) y0 = p.y;
      if (p.x > x1) x1 = p.x;
      if (p.y > y1) y1 = p.y;
    }
  }
  return { x0: x0 - pad, y0: y0 - pad, x1: x1 + pad, y1: y1 + pad };
}

// ---------------------------------------------------------------------------
// Field assembly
// ---------------------------------------------------------------------------

/**
 * Soft union of non-negative excesses: (Σ eᵖ)^(1/p).
 * p = 1 is a plain sum (features add where they overlap); large p approaches
 * max (features stay separate). C¹ everywhere for p > 1, which is the point —
 * a hard max, or a nearest-point pick, creases the field along its seams.
 */
function softUnion(vals: number[], p: number): number {
  if (vals.length === 0) return 0;
  if (p <= 1.001) {
    let s = 0;
    for (const v of vals) s += v;
    return s;
  }
  let s = 0;
  for (const v of vals) if (v > 0) s += Math.pow(v, p);
  return s > 0 ? Math.pow(s, 1 / p) : 0;
}

/** Separable [1,2,1]/4 binomial blur over the N×N field, edges clamped. */
export function smoothField(z: Float32Array, N: number, passes: number): void {
  if (passes <= 0) return;
  const tmp = new Float32Array(N * N);
  const at = (i: number) => (i < 0 ? 0 : i > N - 1 ? N - 1 : i);
  for (let k = 0; k < passes; k++) {
    for (let r = 0; r < N; r++)
      for (let c = 0; c < N; c++)
        tmp[r * N + c] = 0.25 * z[r * N + at(c - 1)] + 0.5 * z[r * N + c] + 0.25 * z[r * N + at(c + 1)];
    for (let r = 0; r < N; r++)
      for (let c = 0; c < N; c++)
        z[r * N + c] = 0.25 * tmp[at(r - 1) * N + c] + 0.5 * tmp[r * N + c] + 0.25 * tmp[at(r + 1) * N + c];
  }
}

/**
 * Project the field onto the feasible set mean(z) = 1.
 *
 * With a pinned boundary the mesh's total area is fixed, so Σ(z − 1) must be 0;
 * any other target is unachievable and the relaxation drifts instead of
 * converging. Subtracting a constant is the least-squares projection onto that
 * constraint — it preserves every *difference* between features and pays for
 * the magnified regions by compressing the periphery, which is exactly the
 * detail-in-context trade. Nodes clamped at `floor` stop absorbing, so the
 * shift is re-spread over the ones still free (a few passes converge).
 */
export function balanceField(z: Float32Array, floor: number, maxIter = 24): number {
  const n = z.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += z[i];
  mean /= n;
  for (let k = 0; k < maxIter; k++) {
    const err = mean - 1;
    if (Math.abs(err) < 1e-4) break;
    let free = 0;
    for (let i = 0; i < n; i++) if (z[i] - err > floor) free++;
    if (free === 0) break;
    const shift = (err * n) / free;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const v = z[i] - shift;
      z[i] = v < floor ? floor : v;
      sum += z[i];
    }
    mean = sum / n;
  }
  return mean;
}

/**
 * Write the feature magnification field into `z` (an N×N grid over the rest
 * domain [-1,1]²). Shapes with mag <= 1 contribute nothing.
 */
export function computeFeatureField(
  z: Float32Array, N: number, shapes: FeatureShape[], p: FeatureFieldParams,
): void {
  const h = 2 / (N - 1);
  const falloff = Math.max(1e-4, p.falloff);

  const active = shapes
    .filter((s) => s.mag > 1 && s.polys.some((poly) => poly.length > 0))
    .map((s) => ({ ...s, box: bboxOf(s.polys, falloff) }));

  if (active.length === 0) {
    z.fill(1);
    return;
  }

  // Kernel weight of each feature at each node — the geometry, computed once, so
  // re-assembling the field under new gains (calibration) costs no distances.
  const F = active.length, NN = N * N;
  const W = new Float32Array(F * NN);
  for (let r = 0; r < N; r++) {
    const ny = r * h - 1;
    for (let c = 0; c < N; c++) {
      const nx = c * h - 1;
      for (let f = 0; f < F; f++) {
        const s = active[f], b = s.box;
        if (nx < b.x0 || nx > b.x1 || ny < b.y0 || ny > b.y1) continue; // cheap reject
        let w = 0;
        for (const poly of s.polys) {
          const t = skirt(distToPoly(nx, ny, poly, s.closed), falloff);
          if (t > w) w = t; // the two eyes are one feature, not two stacked ones
        }
        W[f * NN + r * N + c] = w;
      }
    }
  }

  const gain = new Float64Array(F).fill(1);
  const excess: number[] = [];
  const assemble = () => {
    for (let i = 0; i < NN; i++) {
      excess.length = 0;
      for (let f = 0; f < F; f++) {
        const w = W[f * NN + i];
        if (w > 0) excess.push((active[f].mag - 1) * gain[f] * w);
      }
      z[i] = 1 + softUnion(excess, p.blend) * p.master;
    }
    smoothField(z, N, p.smoothPasses);
    if (p.balanceArea) balanceField(z, p.floor);
  };
  assemble();
  if (!p.calibrate || p.master <= 0) return;

  // CALIBRATION. Measure each feature's excess on its CORE — the nodes inside a
  // ring, or within the kernel's top band along a curve (which has no inside) —
  // and scale its gain toward the target. The blur and the balance are
  // near-linear in the excess, so a few fixed-point passes land it; overlapping
  // features converge jointly. Each pass is one assemble (no distances).
  // A ring's core must be its interior only: counting the rim band (where the
  // blur has already pulled z down) under-reads it and the interior overshoots.
  const PASSES = 4;
  const core: Int32Array[] = [];
  for (let f = 0; f < F; f++) {
    const idx: number[] = [];
    let wMax = 0;
    for (let i = 0; i < NN; i++) wMax = Math.max(wMax, W[f * NN + i]);
    const band = active[f].closed ? 0.9999 : 0.9;
    const cut = Math.min(band, wMax * 0.999); // a feature smaller than a cell: its best-covered nodes
    for (let i = 0; i < NN; i++) if (wMax > 0 && W[f * NN + i] >= cut) idx.push(i);
    core.push(Int32Array.from(idx));
  }
  for (let k = 0; k < PASSES; k++) {
    let worst = 0;
    for (let f = 0; f < F; f++) {
      const idx = core[f];
      if (idx.length === 0) continue;
      let sum = 0;
      for (const i of idx) sum += z[i] - 1;
      const got = sum / idx.length, want = (active[f].mag - 1) * p.master;
      if (got <= 1e-6) continue;
      const ratio = want / got;
      worst = Math.max(worst, Math.abs(ratio - 1));
      gain[f] = Math.min(4, Math.max(0.25, gain[f] * ratio));
    }
    if (worst < 2e-3) break;
    assemble();
  }
}

/** The landmark groups face-api returns, as the five controllable features. */
export function shapesFromFeatures(
  f: {
    jaw: Pt[]; leftBrow: Pt[]; rightBrow: Pt[]; nose: Pt[];
    leftEye: Pt[]; rightEye: Pt[]; mouth: Pt[];
  },
  mag: Record<FeatureKey, number>,
): FeatureShape[] {
  // The 68-point mouth group is an outer lip ring (48–59) followed by an inner
  // one (60–67); the outer ring alone is the shape we want to fill.
  const mouthOuter = f.mouth.length >= 12 ? f.mouth.slice(0, 12) : f.mouth;
  return [
    { key: "eyes", polys: [f.leftEye, f.rightEye], closed: true, mag: mag.eyes },
    { key: "brows", polys: [f.leftBrow, f.rightBrow], closed: false, mag: mag.brows },
    { key: "nose", polys: [f.nose], closed: false, mag: mag.nose },
    { key: "mouth", polys: [mouthOuter], closed: true, mag: mag.mouth },
    { key: "jaw", polys: [f.jaw], closed: false, mag: mag.jaw },
  ];
}
