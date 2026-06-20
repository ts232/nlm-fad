// main.ts — magnify a face from its landmarks. face-api gives 68 points per face;
// those become the data points for the diffusion magnification field (Engine B,
// reused verbatim). Dense feature regions (eyes, nose, mouth) bulge outward.

import { makeDiffMesh, diffuseStep, resetPositions, DEFAULT_DIFF, type DiffMesh, type DiffParams } from "./diffuse";
import { computeDensityField, computeProximityField, warpLookup, type Planes } from "./field";
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
const FS = `#version 300 es
precision highp float; in vec2 v_uv; in float v_z; out vec4 frag;
uniform sampler2D u_tex; uniform int u_mode; uniform vec4 u_color; uniform float u_maxZ;
vec4 heat(float z){ float up=clamp((z-1.0)/max(u_maxZ-1.0,1e-3),0.0,1.0); return vec4(0.80,0.30,0.16, up*0.5); }
void main(){ if(u_mode==1) frag=texture(u_tex,v_uv); else if(u_mode==2) frag=heat(v_z); else frag=u_color; }`;
const prog = program(VS, FS);
const loc = {
  a_pos: gl.getAttribLocation(prog, "a_pos"), a_uv: gl.getAttribLocation(prog, "a_uv"), a_z: gl.getAttribLocation(prog, "a_z"),
  u_tex: gl.getUniformLocation(prog, "u_tex"), u_mode: gl.getUniformLocation(prog, "u_mode"),
  u_color: gl.getUniformLocation(prog, "u_color"), u_maxZ: gl.getUniformLocation(prog, "u_maxZ"),
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
let points: Planes = { n: 0, x: new Float32Array(0), y: new Float32Array(0), w: new Float32Array(0), vx: new Float32Array(0), vy: new Float32Array(0) };
type FeatureKey = "eyes" | "brows" | "nose" | "mouth" | "jaw";
const ui = {
  running: true, itersPerFrame: 20,
  method: "proximity" as "density" | "proximity",
  radius: 0.22,
  showImage: true, showField: false, showMesh: true, showPoints: true,
  // per-feature magnification strength (0 = ignore that feature)
  str: { eyes: 6, brows: 3, nose: 4, mouth: 6, jaw: 3 } as Record<FeatureKey, number>,
};
let fieldDirty = true;
let features: FaceFeatures | null = null; // grouped landmarks from the last detect

// Compose the data points from the feature groups, each carrying its group's
// strength as a per-point weight, so the field magnifies features differentially.
function rebuildPoints(): void {
  if (!features) return;
  const pts: { x: number; y: number }[] = [];
  const ws: number[] = [];
  const add = (arr: { x: number; y: number }[], strength: number) => {
    if (strength <= 0) return;
    for (const p of arr) { pts.push(p); ws.push(strength); }
  };
  add([...features.leftEye, ...features.rightEye], ui.str.eyes);
  add([...features.leftBrow, ...features.rightBrow], ui.str.brows);
  add(features.nose, ui.str.nose);
  add(features.mouth, ui.str.mouth);
  add(features.jaw, ui.str.jaw);
  setPoints(pts, ws);
  setStatus(`${pts.length} landmark points`);
}

function setPoints(pts: { x: number; y: number }[], weights?: number[]): void {
  const n = Math.min(pts.length, MAX_PTS);
  const x = new Float32Array(n), y = new Float32Array(n), w = new Float32Array(n);
  for (let i = 0; i < n; i++) { x[i] = pts[i].x; y[i] = pts[i].y; w[i] = weights ? weights[i] : 1; }
  points = { n, x, y, w, vx: new Float32Array(n), vy: new Float32Array(n) };
  resetPositions(mesh); // start the relaxation fresh for the new point set
  fieldDirty = true;
}
function syncPos(): void {
  for (let i = 0; i < N * N; i++) { posArr[2 * i] = mesh.x[i]; posArr[2 * i + 1] = mesh.y[i]; }
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, posArr);
}
function syncZ(): void { gl.bindBuffer(gl.ARRAY_BUFFER, zBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, mesh.z); }
function computeField(): void {
  if (points.n === 0) mesh.z.fill(1);
  // strength is carried per-point (per feature) in points.w, so volume = 1 here
  else if (ui.method === "density") computeDensityField(mesh, points, 1, ui.radius);
  else computeProximityField(mesh, points, 1, ui.radius);
  syncZ(); fieldDirty = false;
}

const tmp: [number, number] = [0, 0];
function buildPoints(): number {
  const s = 0.009; let o = 0;
  for (let i = 0; i < points.n; i++) {
    warpLookup(mesh, points.x[i], points.y[i], tmp);
    const cx = tmp[0], cy = tmp[1];
    ptVerts[o++] = cx; ptVerts[o++] = cy + s * 1.4;
    ptVerts[o++] = cx - s; ptVerts[o++] = cy - s;
    ptVerts[o++] = cx + s; ptVerts[o++] = cy - s;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, ptsBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, ptVerts);
  return points.n * 3;
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
  const maxStr = Math.max(ui.str.eyes, ui.str.brows, ui.str.nose, ui.str.mouth, ui.str.jaw);
  gl.uniform1f(loc.u_maxZ, Math.max(2, 1 + maxStr));
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

function frame(): void {
  if (fieldDirty) computeField();
  if (ui.running && points.n > 0) { for (let k = 0; k < ui.itersPerFrame; k++) diffuseStep(mesh, params, k); syncPos(); }
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
      features = f;
      if (faces === 0) { setPoints([]); setStatus("no face found"); }
      else rebuildPoints(); // composes selected feature groups, sets the status
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
bindSlider("radius", (v) => { ui.radius = v; fieldDirty = true; }, "radiusVal");
for (const key of ["eyes", "brows", "nose", "mouth", "jaw"] as FeatureKey[]) {
  bindSlider(`str-${key}`, (v) => { ui.str[key] = v; rebuildPoints(); }, `str-${key}Val`, (v) => v.toFixed(1));
}
bindSlider("refine", (v) => (params.refineCoeff = v), "refineVal");
bindSlider("clampEps", (v) => (params.clampEps = v), "clampEpsVal");
bindSlider("iters", (v) => (ui.itersPerFrame = Math.round(v)), "itersVal", (v) => String(Math.round(v)));

el<HTMLSelectElement>("method").addEventListener("change", (e) => {
  const t = e.target as HTMLSelectElement; ui.method = t.value as typeof ui.method; t.blur(); fieldDirty = true;
});
el<HTMLInputElement>("showImage").addEventListener("change", (e) => { ui.showImage = (e.target as HTMLInputElement).checked; });
el<HTMLInputElement>("showField").addEventListener("change", (e) => { ui.showField = (e.target as HTMLInputElement).checked; });
el<HTMLInputElement>("showMesh").addEventListener("change", (e) => { ui.showMesh = (e.target as HTMLInputElement).checked; });
el<HTMLInputElement>("showPoints").addEventListener("change", (e) => { ui.showPoints = (e.target as HTMLInputElement).checked; });


const runBtn = el<HTMLButtonElement>("run");
function syncRun() { runBtn.textContent = ui.running ? "Pause" : "Play"; }
runBtn.addEventListener("click", () => { ui.running = !ui.running; syncRun(); });
el<HTMLButtonElement>("resetMesh").addEventListener("click", () => { resetPositions(mesh); syncPos(); });
el<HTMLInputElement>("file").addEventListener("change", (e) => { const f = (e.target as HTMLInputElement).files?.[0]; if (f) loadFace(f); });

// ---------------------------------------------------------------------------
// Go
// ---------------------------------------------------------------------------
syncRun(); syncPos();
loadFace("/sample-face.png");
requestAnimationFrame(frame);
