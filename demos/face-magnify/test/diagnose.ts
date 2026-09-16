// diagnose.ts — headless measurement of the landmark magnification field.
// Run: node --experimental-strip-types test/diagnose.ts
//
// Answers three questions about a candidate field, without a browser:
//   1. is the target FEASIBLE?  a pinned boundary conserves total area, so the
//      area-weighted mean of z must be 1. mean(z) >> 1 = an impossible ask.
//   2. is it SMOOTH?  max |Δz| between 4-neighbours; a nearest-point field jumps
//      at Voronoi seams between differently-weighted points.
//   3. does the mesh FOLD?  count quads whose signed area is <= 0 after relaxing.
//   4. does it DELIVER?  the achieved area ratio inside the eye and mouth rings,
//      against the magnification the slider claims.
//
// Every method in the comparison is driven by the SAME per-feature targets
// (the legacy fields via legacyWeight), at the demo's defaults and at a strong
// setting — the old "as shipped" run asked for 2.28× and is kept only as history.

import { makeDiffMesh, diffuseStep, rmsError, DEFAULT_DIFF, type DiffMesh } from "../src/diffuse.ts";
import { computeProximityField, computeDensityField, type Planes } from "../src/field.ts";
import { computeFeatureField, shapesFromFeatures, legacyWeight, DEFAULT_FEATURE_FIELD, type FeatureFieldParams } from "../src/featureField.ts";
import { readFileSync } from "node:fs";

const LM = JSON.parse(readFileSync(new URL("./landmarks.sample.json", import.meta.url), "utf8"));

export function pointsFromStrengths(str: Record<string, number>): Planes {
  const pts: { x: number; y: number }[] = [];
  const ws: number[] = [];
  const add = (arr: { x: number; y: number }[], s: number) => {
    if (s <= 0) return;
    for (const p of arr) { pts.push(p); ws.push(s); }
  };
  add([...LM.leftEye, ...LM.rightEye], str.eyes);
  add([...LM.leftBrow, ...LM.rightBrow], str.brows);
  add(LM.nose, str.nose);
  add(LM.mouth, str.mouth);
  add(LM.jaw, str.jaw);
  const n = pts.length;
  const x = new Float32Array(n), y = new Float32Array(n), w = new Float32Array(n);
  for (let i = 0; i < n; i++) { x[i] = pts[i].x; y[i] = pts[i].y; w[i] = ws[i]; }
  return { n, x, y, w, vx: new Float32Array(n), vy: new Float32Array(n) };
}

export function fieldStats(m: DiffMesh) {
  const N = m.N;
  let sum = 0, min = Infinity, max = -Infinity, maxJump = 0;
  for (let i = 0; i < N * N; i++) {
    const z = m.z[i];
    sum += z; if (z < min) min = z; if (z > max) max = z;
  }
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const z = m.z[r * N + c];
      if (c + 1 < N) maxJump = Math.max(maxJump, Math.abs(m.z[r * N + c + 1] - z));
      if (r + 1 < N) maxJump = Math.max(maxJump, Math.abs(m.z[(r + 1) * N + c] - z));
    }
  }
  return { mean: sum / (N * N), min, max, maxJump };
}

// Signed area of each mesh quad; <= 0 means the cell has folded over itself.
export function foldCount(m: DiffMesh): number {
  const N = m.N;
  let folded = 0;
  for (let r = 0; r < N - 1; r++) {
    for (let c = 0; c < N - 1; c++) {
      const a = r * N + c, b = a + 1, d = a + N, e = d + 1;
      const q = [a, b, e, d];
      let s = 0;
      for (let k = 0; k < 4; k++) {
        const i = q[k], j = q[(k + 1) & 3];
        s += m.x[i] * m.y[j] - m.x[j] * m.y[i];
      }
      if (s * 0.5 <= 0) folded++;
    }
  }
  return folded;
}

export function relax(m: DiffMesh, sweeps: number): number[] {
  const trace: number[] = [];
  for (let k = 0; k < sweeps; k++) {
    diffuseStep(m, DEFAULT_DIFF, k);
    if (k % 200 === 199 || k === 0) trace.push(+rmsError(m).toFixed(4));
  }
  return trace;
}

// Achieved area ratio (current / rest) over the quads whose rest centre lies inside the rings.
function inPoly(px: number, py: number, poly: { x: number; y: number }[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const yi = poly[i].y, yj = poly[j].y;
    if (yi > py !== yj > py && px < poly[i].x + ((py - yi) / (yj - yi)) * (poly[j].x - poly[i].x)) inside = !inside;
  }
  return inside;
}
export function delivered(m: DiffMesh, rings: { x: number; y: number }[][]): number {
  const N = m.N, h = 2 / (N - 1);
  let area = 0, rest = 0;
  for (let r = 0; r < N - 1; r++) {
    for (let c = 0; c < N - 1; c++) {
      if (!rings.some((g) => inPoly((c + 0.5) * h - 1, (r + 0.5) * h - 1, g))) continue;
      const q = [r * N + c, r * N + c + 1, (r + 1) * N + c + 1, (r + 1) * N + c];
      let s = 0;
      for (let k = 0; k < 4; k++) { const i = q[k], j = q[(k + 1) & 3]; s += m.x[i] * m.y[j] - m.x[j] * m.y[i]; }
      area += Math.abs(s) / 2; rest += h * h;
    }
  }
  return area / rest;
}

type Mag = { eyes: number; brows: number; nose: number; mouth: number; jaw: number };

function report(label: string, m: DiffMesh, sweeps: number, mag?: Mag): void {
  const st = fieldStats(m);
  const trace = relax(m, sweeps);
  const settled = trace[trace.length - 1];
  const best = Math.min(...trace);
  console.log(`\n== ${label}`);
  console.log(`   mean z ....... ${st.mean.toFixed(3)}   (feasible target is 1.000)`);
  console.log(`   z range ...... ${st.min.toFixed(3)} .. ${st.max.toFixed(3)}`);
  console.log(`   max neighbour jump ... ${st.maxJump.toFixed(4)}`);
  console.log(`   RMS error .... ${trace.join(" → ")}`);
  console.log(`   settled at ${settled.toFixed(4)}; best was ${best.toFixed(4)}` +
    (settled > best * 1.05 ? `  ** DRIFTS AWAY from its best by ${((settled / best - 1) * 100).toFixed(0)}% **` : "  (monotone)"));
  console.log(`   folded cells . ${foldCount(m)} / ${(m.N - 1) * (m.N - 1)}`);
  if (mag) {
    const eyes = delivered(m, [LM.leftEye, LM.rightEye]), mouth = delivered(m, [LM.mouth.slice(0, 12)]);
    console.log(`   delivered .... eyes ${eyes.toFixed(3)}× (asked ${mag.eyes.toFixed(2)}×) · mouth ${mouth.toFixed(3)}× (asked ${mag.mouth.toFixed(2)}×)`);
  }
}

if (import.meta.filename === process.argv[1]) {
  const N = 96;
  const SWEEPS = 1200;
  const R = 0.22;

  // History: the first build's defaults. "Strength 6" through tanh(0.22) asked for 2.28×.
  const shipped = { eyes: 6, brows: 3, nose: 4, mouth: 6, jaw: 3 };
  for (const method of ["proximity", "density"] as const) {
    const m = makeDiffMesh(N);
    const p = pointsFromStrengths(shipped);
    if (method === "proximity") computeProximityField(m, p, 1, R);
    else computeDensityField(m, p, 1, R);
    report(`[history] ${method} — first build's defaults (strength 6 ≈ 2.28×, ${p.n} points)`, m, SWEEPS);
  }

  // The comparison that isolates each change: identical targets for every method.
  const settings: [string, Mag][] = [
    ["demo defaults", { eyes: 1.55, brows: 1.1, nose: 1.2, mouth: 1.4, jaw: 1 }],
    ["strong", { eyes: 2.5, brows: 1.6, nose: 1.8, mouth: 2.3, jaw: 1.4 }],
  ];
  const F = DEFAULT_FEATURE_FIELD;
  const shapeRuns: [string, FeatureFieldParams][] = [
    ["shapes, no balance, no calibration", { ...F, balanceArea: false, calibrate: false }],
    ["shapes + balance, no calibration", { ...F, calibrate: false }],
    ["shapes + balance + calibration (default)", F],
  ];
  for (const [name, mag] of settings) {
    const weights = Object.fromEntries(Object.entries(mag).map(([k, v]) => [k, legacyWeight(v, R)]));
    for (const method of ["proximity", "density"] as const) {
      const m = makeDiffMesh(N);
      const p = pointsFromStrengths(weights);
      if (method === "proximity") computeProximityField(m, p, 1, R);
      else computeDensityField(m, p, 1, R);
      report(`[${name}] ${method} — point cloud, same targets`, m, SWEEPS, mag);
    }
    for (const [label, params] of shapeRuns) {
      const m = makeDiffMesh(N);
      const t0 = performance.now();
      computeFeatureField(m.z, N, shapesFromFeatures(LM, mag), params);
      const ms = performance.now() - t0;
      report(`[${name}] ${label} (field built in ${ms.toFixed(1)} ms)`, m, SWEEPS, mag);
    }
  }
}
