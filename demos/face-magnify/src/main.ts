// main.ts — magnify a face from its landmarks. face-api gives 68 points per face,
// typed by feature; each feature's OUTLINE becomes a smooth kernel in the
// diffusion magnification field (Engine B, reused verbatim — see featureField.ts
// for why the earlier point-cloud field was replaced). The legacy point-cloud
// fields from field.ts stay selectable for comparison.

import {
  makeDiffMesh, diffuseStep, resetPositions, rmsError, DEFAULT_DIFF, type DiffMesh, type DiffParams,
  makeSettle, settleBegin, settleCheck, settleWake,
} from "./diffuse";
import { computeDensityField, computeProximityField, warpLookup, type Planes } from "./field";
import {
  computeFeatureField, shapesFromFeatures, legacyWeight, DEFAULT_FEATURE_FIELD,
  type FeatureKey, type FeatureFieldParams,
} from "./featureField";
import { detectFaces, type FaceFeatures } from "./faceApi";

const canvas = document.getElementById("gl") as HTMLCanvasElement;
const gl = canvas.getContext("webgl2", { antialias: true, premultipliedAlpha: false })!;
if (!gl) throw new Error("WebGL2 not available");

function compile(t: number, src: string): WebGLShader {
  const s = gl.createShader(t)!; gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "compile");
  return s;
}
function program(vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs)); gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
  return p;
}
const VS = `#version 300 es
in vec2 a_pos; in vec2 a_uv; in float a_z; out vec2 v_uv; out float v_z;
void main(){ v_uv=a_uv; v_z=a_z; gl_Position=vec4(a_pos,0.0,1.0); }`;
// The field overlay is two-sided now: balancing the area budget makes z < 1 over
// the periphery (that compression is what pays for the features), and you can't
// steer what you can't see — warm = magnified, cool = compressed.
const FS = `#version 300 es
precision highp float; in vec2 v_uv; in float v_z; out vec4 frag;
uniform sampler2D u_tex; uniform int u_mode; uniform vec4 u_color; uniform float u_maxZ; uniform float u_minZ;
vec4 heat(float z){
  if (z >= 1.0) { float up = clamp((z-1.0)/max(u_maxZ-1.0,1e-3),0.0,1.0); return vec4(0.80,0.30,0.16, up*0.55); }
  float dn = clamp((1.0-z)/max(1.0-u_minZ,1e-3),0.0,1.0); return vec4(0.22,0.46,0.78, dn*0.55);
}
void main(){ if(u_mode==1) frag=texture(u_tex,v_uv); else if(u_mode==2) frag=heat(v_z); else frag=u_color; }`;
const prog = program(VS, FS);
const loc = {
  a_pos: gl.getAttribLocation(prog, "a_pos"), a_uv: gl.getAttribLocation(prog, "a_uv"), a_z: gl.getAttribLocation(prog, "a_z"),
  u_tex: gl.getUniformLocation(prog, "u_tex"), u_mode: gl.getUniformLocation(prog, "u_mode"),
  u_color: gl.getUniformLocation(prog, "u_color"), u_maxZ: gl.getUniformLocation(prog, "u_maxZ"),
  u_minZ: gl.getUniformLocation(prog, "u_minZ"),
};

// ---------------------------------------------------------------------------
// Mesh + buffers
// ---------------------------------------------------------------------------
const N = 96;
const mesh: DiffMesh = makeDiffMesh(N);
const posArr = new Float32Array(N * N * 2);
const uvArr = new Float32Array(N * N * 2);
for (let i = 0; i < N * N; i++) { uvArr[2 * i] = (mesh.x0[i] + 1) * 0.5; uvArr[2 * i + 1] = (mesh.y0[i] + 1) * 0.5; }

const vao = gl.createVertexArray()!;
gl.bindVertexArray(vao);
const posBuf = gl.createBuffer()!;
gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferData(gl.ARRAY_BUFFER, posArr, gl.DYNAMIC_DRAW);
gl.enableVertexAttribArray(loc.a_pos); gl.vertexAttribPointer(loc.a_pos, 2, gl.FLOAT, false, 0, 0);
const uvBuf = gl.createBuffer()!;
gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf); gl.bufferData(gl.ARRAY_BUFFER, uvArr, gl.STATIC_DRAW);
gl.enableVertexAttribArray(loc.a_uv); gl.vertexAttribPointer(loc.a_uv, 2, gl.FLOAT, false, 0, 0);
const zBuf = gl.createBuffer()!;
gl.bindBuffer(gl.ARRAY_BUFFER, zBuf); gl.bufferData(gl.ARRAY_BUFFER, mesh.z, gl.DYNAMIC_DRAW);
gl.enableVertexAttribArray(loc.a_z); gl.vertexAttribPointer(loc.a_z, 1, gl.FLOAT, false, 0, 0);

const triBuf = gl.createBuffer()!; let triCount = 0;
{
  const idx = new Uint32Array((N - 1) * (N - 1) * 6); let k = 0;
  for (let r = 0; r < N - 1; r++) for (let c = 0; c < N - 1; c++) {
    const a = r * N + c, b = a + 1, cc = a + N, d = cc + 1;
    idx[k++] = a; idx[k++] = b; idx[k++] = cc; idx[k++] = b; idx[k++] = d; idx[k++] = cc;
  }
  triCount = idx.length;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, triBuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
}
const lineBuf = gl.createBuffer()!; let lineCount = 0;
{
  const step = Math.max(1, Math.round(N / 28)); const seg: number[] = [];
  for (let r = 0; r < N; r += step) for (let c = 0; c < N - 1; c++) seg.push(r * N + c, r * N + c + 1);
  for (let c = 0; c < N; c += step) for (let r = 0; r < N - 1; r++) seg.push(r * N + c, (r + 1) * N + c);
  const idx = new Uint32Array(seg); lineCount = idx.length;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineBuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
}

// landmark dots — own VAO/buffer
const MAX_PTS = 600;
const ptVerts = new Float32Array(MAX_PTS * 3 * 2);
const ptsVao = gl.createVertexArray()!;
gl.bindVertexArray(ptsVao);
const ptsBuf = gl.createBuffer()!;
gl.bindBuffer(gl.ARRAY_BUFFER, ptsBuf); gl.bufferData(gl.ARRAY_BUFFER, ptVerts, gl.DYNAMIC_DRAW);
gl.enableVertexAttribArray(loc.a_pos); gl.vertexAttribPointer(loc.a_pos, 2, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

// ---------------------------------------------------------------------------
// Texture (the face)
// ---------------------------------------------------------------------------
const texture = gl.createTexture()!;
gl.bindTexture(gl.TEXTURE_2D, texture);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([26, 28, 32, 255]));
function uploadImage(img: HTMLImageElement): void {
  gl.bindTexture(gl.TEXTURE_2D, texture); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  gl.generateMipmap(gl.TEXTURE_2D); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const params: DiffParams = { ...DEFAULT_DIFF };
const ff: FeatureFieldParams = { ...DEFAULT_FEATURE_FIELD };
let points: Planes = { n: 0, x: new Float32Array(0), y: new Float32Array(0), w: new Float32Array(0), vx: new Float32Array(0), vy: new Float32Array(0) };
const FEATURES: FeatureKey[] = ["eyes", "brows", "nose", "mouth", "jaw"];
const ui = {
  running: true, itersPerFrame: 20,
  method: "shapes" as "shapes" | "density" | "proximity",
  radius: 0.22, // legacy point-cloud methods only
  showImage: true, showField: false, showMesh: true, showPoints: true,
  // Per-feature TARGET MAGNIFICATION, in honest units: 1.55 = "this feature
  // should end up 1.55× its area". 1.00 = leave it alone. The old control was a
  // unitless 0–12 "strength" that reached the mesh through tanh(radius) ≈ 0.22,
  // so "6" silently meant 2.28× — which is why it was so hard to aim.
  mag: { eyes: 1.55, brows: 1.1, nose: 1.2, mouth: 1.4, jaw: 1 } as Record<FeatureKey, number>,
};
let fieldDirty = true;
let features: FaceFeatures | null = null; // grouped landmarks from the last detect

// Two point sets, deliberately: `landmarks` is every detected point and is what
// the overlay draws (a feature dialled to 1.00× is still a detected feature),
// while `points` is the weighted subset the LEGACY fields consume — those pick
// the nearest point's weight, so a feature that is off has to be absent rather
// than present with weight 0, or it would punch a hole in its neighbour's halo.
let landmarks: Planes = points;
function rebuildPoints(): void {
  if (!features) return;
  const groups: [FeatureKey, { x: number; y: number }[]][] = [
    ["eyes", [...features.leftEye, ...features.rightEye]],
    ["brows", [...features.leftBrow, ...features.rightBrow]],
    ["nose", features.nose],
    ["mouth", features.mouth],
    ["jaw", features.jaw],
  ];
  const all: { x: number; y: number }[] = [];
  const pts: { x: number; y: number }[] = [];
  const ws: number[] = [];
  for (const [key, arr] of groups) {
    all.push(...arr);
    const w = legacyWeight(ui.mag[key], ui.radius);
    if (w <= 0) continue;
    for (const p of arr) { pts.push(p); ws.push(w); }
  }
  landmarks = makePointSet(all);
  setPoints(pts, ws);
}

function makePointSet(pts: { x: number; y: number }[], weights?: number[]): Planes {
  const n = Math.min(pts.length, MAX_PTS);
  const x = new Float32Array(n), y = new Float32Array(n), w = new Float32Array(n);
  for (let i = 0; i < n; i++) { x[i] = pts[i].x; y[i] = pts[i].y; w[i] = weights ? weights[i] : 1; }
  return { n, x, y, w, vx: new Float32Array(n), vy: new Float32Array(n) };
}

function setPoints(pts: { x: number; y: number }[], weights?: number[]): void {
  points = makePointSet(pts, weights);
  fieldDirty = true;
}
function syncPos(): void {
  for (let i = 0; i < N * N; i++) { posArr[2 * i] = mesh.x[i]; posArr[2 * i + 1] = mesh.y[i]; }
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, posArr);
}
function syncZ(): void { gl.bindBuffer(gl.ARRAY_BUFFER, zBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, mesh.z); }

let zMin = 1, zMax = 2;
function computeField(): void {
  if (ui.method === "shapes") {
    if (!features) mesh.z.fill(1);
    else computeFeatureField(mesh.z, N, shapesFromFeatures(features, ui.mag), ff);
  } else if (points.n === 0) mesh.z.fill(1);
  // strength is carried per-point (per feature) in points.w, so volume = 1 here
  else if (ui.method === "density") computeDensityField(mesh, points, 1, ui.radius);
  else computeProximityField(mesh, points, 1, ui.radius);

  zMin = Infinity; zMax = -Infinity;
  for (let i = 0; i < N * N; i++) { const z = mesh.z[i]; if (z < zMin) zMin = z; if (z > zMax) zMax = z; }
  syncZ(); fieldDirty = false;
}

const tmp: [number, number] = [0, 0];
function buildPoints(): number {
  const s = 0.009; let o = 0;
  for (let i = 0; i < landmarks.n; i++) {
    warpLookup(mesh, landmarks.x[i], landmarks.y[i], tmp);
    const cx = tmp[0], cy = tmp[1];
    ptVerts[o++] = cx; ptVerts[o++] = cy + s * 1.4;
    ptVerts[o++] = cx - s; ptVerts[o++] = cy - s;
    ptVerts[o++] = cx + s; ptVerts[o++] = cy - s;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, ptsBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, ptVerts);
  return landmarks.n * 3;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------
function resize(): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const size = Math.min(canvas.clientWidth, canvas.clientHeight);
  const px = Math.round(size * dpr);
  if (canvas.width !== px || canvas.height !== px) canvas.width = canvas.height = px;
}
function draw(ptCount: number): void {
  resize();
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.07, 0.07, 0.08, 1); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(prog);
  gl.uniform1f(loc.u_maxZ, Math.max(1.05, zMax));
  gl.uniform1f(loc.u_minZ, Math.min(0.95, zMin));
  gl.bindVertexArray(vao);
  if (ui.showImage) {
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(loc.u_tex, 0); gl.uniform1i(loc.u_mode, 1);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, triBuf); gl.drawElements(gl.TRIANGLES, triCount, gl.UNSIGNED_INT, 0);
  }
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  if (ui.showField) { gl.uniform1i(loc.u_mode, 2); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, triBuf); gl.drawElements(gl.TRIANGLES, triCount, gl.UNSIGNED_INT, 0); }
  if (ui.showMesh) {
    gl.uniform1i(loc.u_mode, 0); gl.uniform4f(loc.u_color, 0.6, 0.62, 0.7, ui.showImage ? 0.28 : 0.5);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineBuf); gl.drawElements(gl.LINES, lineCount, gl.UNSIGNED_INT, 0);
  }
  if (ui.showPoints) {
    gl.bindVertexArray(ptsVao); gl.uniform1i(loc.u_mode, 0); gl.uniform4f(loc.u_color, 0.98, 0.85, 0.2, 0.95);
    gl.drawArrays(gl.TRIANGLES, 0, ptCount);
  }
  gl.disable(gl.BLEND);
}

// Auto-idle: stop sweeping once the relaxation has settled or stalled (see
// diffuse.ts settleCheck). Every field change goes through computeField, which
// wakes it; so do the solver params, Reset mesh and Play.
const settle = makeSettle(mesh);
function syncSolverState(): void {
  el("solverState").textContent = !ui.running ? "paused" : !features ? "no face" : settle.state || "running";
}
function wake(): void { settleWake(settle); syncSolverState(); }

let rmsTick = 0;
function frame(): void {
  if (fieldDirty) { computeField(); wake(); }
  if (ui.running && features && !settle.state) {
    settleBegin(settle, mesh);
    for (let k = 0; k < ui.itersPerFrame; k++) diffuseStep(mesh, params, k);
    syncPos();
    // The residual is the honest read on whether the target is achievable at
    // all: a balanced field drives it toward 0, an infeasible one stalls or
    // drifts back up. Computed every frame for the settle check; shown at 4 Hz.
    const rms = rmsError(mesh);
    if (settleCheck(settle, mesh, rms)) { syncSolverState(); el("rms").textContent = rms.toFixed(4); }
    else if (++rmsTick % 15 === 0) el("rms").textContent = rms.toFixed(4);
  }
  draw(buildPoints());
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------
// Face loading + detection
// ---------------------------------------------------------------------------
function setStatus(s: string): void { el("status").textContent = s; }
function loadFace(src: string | File): void {
  setStatus("loading image…");
  const img = new Image();
  img.onload = async () => {
    uploadImage(img);
    setStatus("detecting face…");
    try {
      const { features: f, faces } = await detectFaces(img);
      features = faces === 0 ? null : f;
      resetPositions(mesh); // a new face starts the relaxation fresh
      if (faces === 0) { setPoints([]); setStatus("no face found"); }
      else { rebuildPoints(); setStatus(`${landmarks.n} landmark points · ${faces} face${faces > 1 ? "s" : ""}`); }
      fieldDirty = true;
    } catch (e) { setStatus("face detection failed (see console)"); console.error(e); }
    if (typeof src !== "string") URL.revokeObjectURL(img.src);
  };
  img.onerror = () => setStatus("image failed to load");
  img.src = typeof src === "string" ? src : URL.createObjectURL(src);
}

canvas.addEventListener("dragover", (e) => e.preventDefault());
canvas.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer?.files?.[0]; if (f) loadFace(f); });

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
function el<T extends HTMLElement>(id: string): T { return document.getElementById(id) as T; }
function bindSlider(id: string, set: (v: number) => void, fmtId: string, fmt?: (v: number) => string) {
  const s = el<HTMLInputElement>(id);
  const update = () => { const v = parseFloat(s.value); set(v); el(fmtId).textContent = fmt ? fmt(v) : v.toFixed(2); };
  s.addEventListener("input", update); update();
}
const times = (v: number) => `${v.toFixed(2)}×`;

// Nothing here resets the mesh. Every control retargets the field and lets the
// running relaxation walk to the new answer, which is what makes the sliders
// feel continuous — the old build reset the mesh on every strength change, so
// each nudge restarted the solve from a flat grid.
bindSlider("radius", (v) => { ui.radius = v; rebuildPoints(); }, "radiusVal");
bindSlider("master", (v) => { ff.master = v; fieldDirty = true; }, "masterVal");
bindSlider("falloff", (v) => { ff.falloff = v; fieldDirty = true; }, "falloffVal", (v) => v.toFixed(3));
bindSlider("blend", (v) => { ff.blend = v; fieldDirty = true; }, "blendVal", (v) => v.toFixed(1));
bindSlider("smooth", (v) => { ff.smoothPasses = Math.round(v); fieldDirty = true; }, "smoothVal", (v) => String(Math.round(v)));
for (const key of FEATURES) {
  bindSlider(`str-${key}`, (v) => { ui.mag[key] = v; rebuildPoints(); fieldDirty = true; }, `str-${key}Val`, times);
}
bindSlider("refine", (v) => { params.refineCoeff = v; wake(); }, "refineVal");
bindSlider("clampEps", (v) => { params.clampEps = v; wake(); }, "clampEpsVal");
bindSlider("iters", (v) => (ui.itersPerFrame = Math.round(v)), "itersVal", (v) => String(Math.round(v)));

el<HTMLInputElement>("balance").addEventListener("change", (e) => {
  ff.balanceArea = (e.target as HTMLInputElement).checked; fieldDirty = true;
});

// Only one of the two parameter sets applies at a time; showing both invites
// turning a knob that does nothing.
function syncMethodUI(): void {
  const shapes = ui.method === "shapes";
  for (const id of ["g-falloff", "g-blend", "g-smooth", "g-balance"]) el(id).hidden = !shapes;
  el("g-radius").hidden = shapes;
}
el<HTMLSelectElement>("method").addEventListener("change", (e) => {
  const t = e.target as HTMLSelectElement; ui.method = t.value as typeof ui.method; t.blur();
  syncMethodUI(); fieldDirty = true;
});
el<HTMLInputElement>("showImage").addEventListener("change", (e) => { ui.showImage = (e.target as HTMLInputElement).checked; });
el<HTMLInputElement>("showField").addEventListener("change", (e) => { ui.showField = (e.target as HTMLInputElement).checked; });
el<HTMLInputElement>("showMesh").addEventListener("change", (e) => { ui.showMesh = (e.target as HTMLInputElement).checked; });
el<HTMLInputElement>("showPoints").addEventListener("change", (e) => { ui.showPoints = (e.target as HTMLInputElement).checked; });


const runBtn = el<HTMLButtonElement>("run");
function syncRun() { runBtn.textContent = ui.running ? "Pause" : "Play"; }
runBtn.addEventListener("click", () => { ui.running = !ui.running; syncRun(); wake(); });
el<HTMLButtonElement>("resetMesh").addEventListener("click", () => { resetPositions(mesh); syncPos(); wake(); });
el<HTMLInputElement>("file").addEventListener("change", (e) => { const f = (e.target as HTMLInputElement).files?.[0]; if (f) loadFace(f); });

// ---------------------------------------------------------------------------
// Go
// ---------------------------------------------------------------------------
syncRun(); syncPos(); syncMethodUI();
loadFace(import.meta.env.BASE_URL + "sample-face.png");
requestAnimationFrame(frame);
