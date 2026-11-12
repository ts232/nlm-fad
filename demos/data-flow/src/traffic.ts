// traffic.ts — scheduled traffic into Reagan National (DCA), schematic.
//
// The first version of this demo scattered 160 planes in random directions and bounced them off
// the frame. That is the classic FAD apps/2d set-up, but it reads as noise: the whole map was
// "dense", so the field saturated everywhere and no one plane got much room (Alan, 2026-11-11).
// Real traffic runs in streams. Here a couple of dozen aircraft follow routes traced from the
// map itself: arrivals follow the Potomac down to the runway, departures leave the other way,
// and helicopters run low along the rivers. The field each one makes is in field.ts
// (computeLensField); this file only says where they are.
//
// Positions are in the zone's [-1,1]² (y up). Routes are written in the map image's pixels
// (dcMetro.png, 512², y down) so they can be checked against the picture, and converted once.

import type { Planes } from "./field";

export type Kind = "plane" | "heli";
export type Flow = "south" | "north";

interface Route {
  kind: Kind;
  pts: [number, number][]; // map pixels, y down
  every: number;           // seconds between spawns at traffic level 1
  speed: number;           // cruise speed, zone units per second
  // speed along the route as a fraction of cruise, given distance travelled and total length
  profile?: (s: number, len: number, r: Built) => number;
  touchdown?: number;      // index of the waypoint where an arrival lands (for the profile)
  liftoff?: number;        // index of the waypoint where a departure leaves the ground
}

interface Built {
  def: Route;
  x: Float32Array; y: Float32Array; s: Float32Array; // dense samples + arc length
  len: number;
  sTouch: number; sLift: number;
  clock: number;           // seconds until the next spawn
}

export interface Aircraft { route: number; s: number; age: number; }

// ---------------------------------------------------------------------------
// Routes (map pixels). The river centreline was read off the map's water colour.
// Runway 1/19 sits on the river's west bank at x ≈ 262, from y ≈ 418 (north) to 490 (south).
// ---------------------------------------------------------------------------
const RIVER_IN: [number, number][] = [
  [-40, 108], [0, 132], [66, 176], [112, 216], [156, 264], [192, 304], [232, 344], [255, 372], [262, 398],
];
const POTOMAC_HELI: [number, number][] = [
  [-30, 126], [16, 140], [76, 172], [140, 230], [176, 268], [200, 300], [240, 336], [288, 384], [308, 420], [310, 540],
];
const ANACOSTIA_HELI: [number, number][] = [
  [444, -30], [444, 112], [444, 300], [430, 330], [410, 350], [385, 375], [355, 395], [318, 412], [310, 440], [310, 540],
];
const rev = (p: [number, number][]) => p.slice().reverse();

// Speed profiles, as a fraction of the route's cruise speed.
const arrival = (s: number, _len: number, r: Built) => {
  const toGo = r.sTouch - s;
  if (toGo > 0.35) return 1;                              // on the approach
  if (toGo > 0) return 0.75 + 0.25 * (toGo / 0.35);       // slowing on final
  return Math.max(0.12, 0.75 + toGo * 3.2);               // rolling out
};
const departure = (s: number, _len: number, r: Built) => {
  if (s >= r.sLift) return 1;
  return 0.12 + 0.88 * Math.sqrt(s / Math.max(r.sLift, 1e-3)); // the take-off roll
};

const ROUTES: Record<Flow, Route[]> = {
  // Landing south on runway 19: arrivals come down the river from the north-west.
  south: [
    { kind: "plane", pts: [...RIVER_IN, [262, 425], [262, 455], [255, 463], [240, 466]], every: 1.4, speed: 0.12, profile: arrival, touchdown: RIVER_IN.length },
    { kind: "plane", pts: [[262, 418], [262, 470], [262, 500], [248, 530], [205, 570]], every: 7, speed: 0.15, profile: departure, liftoff: 1 },
    { kind: "plane", pts: [[262, 418], [262, 470], [264, 500], [282, 530], [330, 570]], every: 9, speed: 0.15, profile: departure, liftoff: 1 },
    { kind: "heli", pts: POTOMAC_HELI, every: 24, speed: 0.05 },
    { kind: "heli", pts: rev(POTOMAC_HELI), every: 24, speed: 0.05 },
    { kind: "heli", pts: ANACOSTIA_HELI, every: 40, speed: 0.05 },
    { kind: "heli", pts: rev(ANACOSTIA_HELI), every: 40, speed: 0.05 },
  ],
  // Taking off north on runway 1: departures climb out up the river; arrivals come in from the south.
  north: [
    { kind: "plane", pts: [[262, 490], [262, 440], ...rev(RIVER_IN)], every: 1.6, speed: 0.13, profile: departure, liftoff: 1 },
    { kind: "plane", pts: [[262, 580], [262, 520], [262, 488], [262, 455], [255, 446], [240, 442]], every: 6, speed: 0.12, profile: arrival, touchdown: 2 },
    { kind: "plane", pts: [[400, 600], [310, 545], [264, 510], [262, 488], [262, 455], [255, 446], [240, 442]], every: 8, speed: 0.12, profile: arrival, touchdown: 3 },
    { kind: "heli", pts: POTOMAC_HELI, every: 24, speed: 0.05 },
    { kind: "heli", pts: rev(POTOMAC_HELI), every: 24, speed: 0.05 },
    { kind: "heli", pts: ANACOSTIA_HELI, every: 40, speed: 0.05 },
    { kind: "heli", pts: rev(ANACOSTIA_HELI), every: 40, speed: 0.05 },
  ],
};

const toZone = (p: [number, number]): [number, number] => [p[0] / 256 - 1, 1 - p[1] / 256];

// Catmull-Rom through the waypoints, sampled densely, with arc length.
function build(def: Route): Built {
  const P = def.pts.map(toZone);
  const xs: number[] = [], ys: number[] = [], wpS: number[] = [0];
  const SUB = 16;
  for (let i = 0; i < P.length - 1; i++) {
    const p0 = P[Math.max(0, i - 1)], p1 = P[i], p2 = P[i + 1], p3 = P[Math.min(P.length - 1, i + 2)];
    for (let k = 0; k < SUB; k++) {
      const t = k / SUB, t2 = t * t, t3 = t2 * t;
      const f = (a: number, b: number, c: number, d: number) =>
        0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
      xs.push(f(p0[0], p1[0], p2[0], p3[0])); ys.push(f(p0[1], p1[1], p2[1], p3[1]));
    }
  }
  xs.push(P[P.length - 1][0]); ys.push(P[P.length - 1][1]);
  const n = xs.length, s = new Float32Array(n);
  for (let i = 1; i < n; i++) s[i] = s[i - 1] + Math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]);
  for (let i = 1; i < P.length; i++) wpS.push(s[i * SUB]);
  return {
    def, x: Float32Array.from(xs), y: Float32Array.from(ys), s, len: s[n - 1],
    sTouch: def.touchdown != null ? wpS[def.touchdown] : Infinity,
    sLift: def.liftoff != null ? wpS[def.liftoff] : 0,
    clock: 0,
  };
}

// Position + unit heading at arc length s.
function at(r: Built, s: number, out: { x: number; y: number; hx: number; hy: number }): void {
  const S = r.s; let lo = 0, hi = S.length - 1;
  if (s <= 0) s = 0; else if (s >= r.len) s = r.len;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (S[mid] <= s) lo = mid; else hi = mid; }
  const seg = S[hi] - S[lo] || 1, f = (s - S[lo]) / seg;
  out.x = r.x[lo] + (r.x[hi] - r.x[lo]) * f;
  out.y = r.y[lo] + (r.y[hi] - r.y[lo]) * f;
  const dx = r.x[hi] - r.x[lo], dy = r.y[hi] - r.y[lo], L = Math.hypot(dx, dy) || 1;
  out.hx = dx / L; out.hy = dy / L;
}

function mulberry(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const MAX = 120;
const FADE = 1.2; // seconds to fade a field in at spawn and out at the end of the route

export class Traffic {
  routes: Built[] = [];
  live: Aircraft[] = [];
  /** The data the field and the renderer read; `w` fades each aircraft's field in and out. */
  planes: Planes;
  kind: Kind[] = [];
  private rnd: () => number;
  private tmp = { x: 0, y: 0, hx: 0, hy: 0 };

  flow: Flow;
  level: number;

  constructor(flow: Flow = "south", level = 1, seed = 7) {
    this.flow = flow;
    this.level = level;
    this.rnd = mulberry(seed);
    this.planes = { n: 0, x: new Float32Array(MAX), y: new Float32Array(MAX), vx: new Float32Array(MAX), vy: new Float32Array(MAX), w: new Float32Array(MAX) };
    this.setFlow(flow);
  }

  /** Switch runway direction and start from a full sky (the routes are pre-run, not empty). */
  setFlow(flow: Flow): void {
    this.flow = flow;
    this.routes = ROUTES[flow].map(build);
    this.live = [];
    this.routes.forEach((r) => (r.clock = this.rnd() * r.def.every));
    for (let t = 0; t < 60; t += 0.1) this.step(0.1, 1);
    this.sync();
  }

  /** Route polylines in zone coordinates, for drawing. */
  paths(): { kind: Kind; x: Float32Array; y: Float32Array }[] {
    return this.routes.map((r) => ({ kind: r.def.kind, x: r.x, y: r.y }));
  }

  step(dt: number, speed: number): void {
    const sdt = dt * speed;
    for (let i = 0; i < this.routes.length; i++) {
      const r = this.routes[i];
      r.clock -= sdt * this.level;
      if (r.clock <= 0 && this.live.length < MAX) {
        this.live.push({ route: i, s: 0, age: 0 });
        r.clock += r.def.every * (0.75 + 0.5 * this.rnd()); // ±25% so the stream isn't a metronome
      }
    }
    for (const a of this.live) {
      const r = this.routes[a.route];
      const f = r.def.profile ? r.def.profile(a.s, r.len, r) : 1;
      a.s += r.def.speed * f * sdt;
      a.age += sdt;
    }
    this.live = this.live.filter((a) => a.s < this.routes[a.route].len);
    this.sync();
  }

  private sync(): void {
    const p = this.planes, o = this.tmp;
    let n = 0;
    for (const a of this.live) {
      const r = this.routes[a.route];
      at(r, a.s, o);
      p.x[n] = o.x; p.y[n] = o.y; p.vx[n] = o.hx; p.vy[n] = o.hy;
      // Fade the field in after spawn and out over the last stretch, so nothing pops.
      const left = (r.len - a.s) / Math.max(r.def.speed * 0.3, 1e-3);
      p.w![n] = Math.min(1, a.age / FADE, left / FADE);
      this.kind[n] = r.def.kind;
      n++;
    }
    p.n = n;
  }
}
