// Headless check: how much magnification each aircraft actually gets, old set-up vs new.
//   node --experimental-strip-types demos/data-flow/test/lens.ts
import { makeDiffMesh, diffuseStep, rmsError, DEFAULT_DIFF } from "../src/diffuse.ts";
import { makePlanes, stepPlanes, computeDensityField, computeLensField, deliveredAt } from "../src/field.ts";
import { Traffic } from "../src/traffic.ts";

const stats = (v: number[]) => { const s = v.filter(Number.isFinite).sort((a, b) => a - b); const q = (f: number) => s[Math.floor(f * (s.length - 1))]; return `median ${q(0.5).toFixed(2)}× · p10 ${q(0.1).toFixed(2)}× · p90 ${q(0.9).toFixed(2)}× · n ${s.length}`; };
const inside = (x: number, y: number) => Math.abs(x) < 0.9 && Math.abs(y) < 0.9;

function run(label: string, setup: () => { step: (dt: number) => void; field: (m: any) => void; pts: () => { x: number; y: number; w: number }[] }, iters: number) {
  const m = makeDiffMesh(64); const s = setup(); const got: number[] = []; let counts: number[] = [];
  const t0 = performance.now();
  for (let f = 0; f < 60 * 20; f++) {
    s.step(1 / 60); s.field(m);
    for (let k = 0; k < iters; k++) diffuseStep(m, DEFAULT_DIFF, k);
    if (f > 60 * 5 && f % 6 === 0) { const P = s.pts().filter((p) => inside(p.x, p.y)); counts.push(P.length); for (const p of P) if (p.w > 0.99) got.push(deliveredAt(m, p.x, p.y)); }
  }
  const ms = (performance.now() - t0) / (60 * 20);
  console.log(`${label}\n  at aircraft: ${stats(got)}\n  on screen: ${Math.min(...counts)}–${Math.max(...counts)} · rms ${rmsError(m).toFixed(3)} · ${ms.toFixed(2)} ms/frame`);
}

run("OLD: 160 random planes, density 4×, r 0.20", () => {
  const p = makePlanes(160);
  return { step: (dt) => stepPlanes(p, dt, 1), field: (m) => computeDensityField(m, p, 4, 0.2), pts: () => Array.from({ length: p.n }, (_, i) => ({ x: p.x[i], y: p.y[i], w: 1 })) };
}, 18);

for (const flow of ["south", "north"] as const) for (const [peak, rad] of [[4, 0.14], [5, 0.16]]) {
  run(`NEW ${flow} flow: lens ${peak}×, r ${rad}`, () => {
    const t = new Traffic(flow);
    return { step: (dt) => t.step(dt, 1), field: (m) => computeLensField(m, t.planes, peak, rad), pts: () => Array.from({ length: t.planes.n }, (_, i) => ({ x: t.planes.x[i], y: t.planes.y[i], w: t.planes.w![i] })) };
  }, 24);
}
