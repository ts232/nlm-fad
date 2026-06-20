// main.ts — WebGL2 renderer + paint interaction for the area-based diffusion
// magnification prototype. Paint a magnification field; the solver (diffuse.ts)
// relaxes the mesh toward it, frame by frame. Renders the deforming mesh as
// grid lines, optionally textured, with the painted field as a heat overlay.

import {
  makeDiffMesh, diffuseStep, rmsError, paintField, resetPositions, clearField,
  DEFAULT_DIFF, type DiffMesh, type DiffParams,
} from "./diffuse";

const canvas = document.getElementById("gl") as HTMLCanvasElement;
const gl = canvas.getContext("webgl2", { antialias: true, premultipliedAlpha: false })!;
if (!gl) throw new Error("WebGL2 not available");

function compile(type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "compile");
  return s;
}
function program(vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
  return p;
}

const VS = `#version 300 es
in vec2 a_pos; in vec2 a_uv; in float a_z;
out vec2 v_uv; out float v_z;
void main(){ v_uv=a_uv; v_z=a_z; gl_Position=vec4(a_pos,0.0,1.0); }`;

const FS = `#version 300 es
precision highp float;
in vec2 v_uv; in float v_z;
out vec4 frag;
uniform sampler2D u_tex;
uniform int u_mode;          // 0 flat, 1 texture, 2 heatmap
uniform vec4 u_color;
uniform float u_minZ, u_maxZ;
vec4 heat(float z){
  float up = clamp((z-1.0)/max(u_maxZ-1.0,1e-3),0.0,1.0); // magnify -> warm
  float dn = clamp((1.0-z)/max(1.0-u_minZ,1e-3),0.0,1.0); // minify  -> cool
  vec3 warm=vec3(0.80,0.22,0.20), cool=vec3(0.20,0.45,0.72);
  return vec4(warm*up + cool*dn, max(up,dn)*0.7);
}
void main(){
  if(u_mode==1) frag=texture(u_tex,v_uv);
  else if(u_mode==2) frag=heat(v_z);
  else frag=u_color;
}`;

const prog = program(VS, FS);
const loc = {
  a_pos: gl.getAttribLocation(prog, "a_pos"),
  a_uv: gl.getAttribLocation(prog, "a_uv"),
  a_z: gl.getAttribLocation(prog, "a_z"),
  u_tex: gl.getUniformLocation(prog, "u_tex"),
  u_mode: gl.getUniformLocation(prog, "u_mode"),
  u_color: gl.getUniformLocation(prog, "u_color"),
  u_minZ: gl.getUniformLocation(prog, "u_minZ"),
  u_maxZ: gl.getUniformLocation(prog, "u_maxZ"),
};

// ---------------------------------------------------------------------------
// Mesh + buffers
// ---------------------------------------------------------------------------
const N = 64; // grid nodes per side
const mesh: DiffMesh = makeDiffMesh(N);
const posArr = new Float32Array(N * N * 2); // interleaved current positions
const uvArr = new Float32Array(N * N * 2); // home positions -> [0,1]
for (let i = 0; i < N * N; i++) {
  uvArr[2 * i] = (mesh.x0[i] + 1) * 0.5;
  uvArr[2 * i + 1] = (mesh.y0[i] + 1) * 0.5;
}

const vao = gl.createVertexArray()!;
gl.bindVertexArray(vao);

const posBuf = gl.createBuffer()!;
gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
gl.bufferData(gl.ARRAY_BUFFER, posArr, gl.DYNAMIC_DRAW);
gl.enableVertexAttribArray(loc.a_pos);
gl.vertexAttribPointer(loc.a_pos, 2, gl.FLOAT, false, 0, 0);

const uvBuf = gl.createBuffer()!;
gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
gl.bufferData(gl.ARRAY_BUFFER, uvArr, gl.STATIC_DRAW);
gl.enableVertexAttribArray(loc.a_uv);
gl.vertexAttribPointer(loc.a_uv, 2, gl.FLOAT, false, 0, 0);

const zBuf = gl.createBuffer()!;
gl.bindBuffer(gl.ARRAY_BUFFER, zBuf);
gl.bufferData(gl.ARRAY_BUFFER, mesh.z, gl.DYNAMIC_DRAW);
gl.enableVertexAttribArray(loc.a_z);
gl.vertexAttribPointer(loc.a_z, 1, gl.FLOAT, false, 0, 0);

// triangle indices
const triBuf = gl.createBuffer()!;
let triCount = 0;
{
  const idx = new Uint32Array((N - 1) * (N - 1) * 6);
  let k = 0;
  for (let r = 0; r < N - 1; r++)
    for (let c = 0; c < N - 1; c++) {
      const a = r * N + c, b = a + 1, cc = a + N, d = cc + 1;
      idx[k++] = a; idx[k++] = b; idx[k++] = cc;
      idx[k++] = b; idx[k++] = d; idx[k++] = cc;
    }
  triCount = idx.length;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, triBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
}

// grid line indices (subsampled to ~24 lines, each a full polyline)
const lineBuf = gl.createBuffer()!;
let lineCount = 0;
{
  const step = Math.max(1, Math.round(N / 24));
  const seg: number[] = [];
  for (let r = 0; r < N; r += step) for (let c = 0; c < N - 1; c++) seg.push(r * N + c, r * N + c + 1);
  for (let c = 0; c < N; c += step) for (let r = 0; r < N - 1; r++) seg.push(r * N + c, (r + 1) * N + c);
  const idx = new Uint32Array(seg);
  lineCount = idx.length;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
}

// ---------------------------------------------------------------------------
// Texture
// ---------------------------------------------------------------------------
const texture = gl.createTexture()!;
gl.bindTexture(gl.TEXTURE_2D, texture);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([170, 170, 174, 255]));

function uploadImage(src: HTMLImageElement | HTMLCanvasElement): void {
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
}
function loadURL(url: string): void {
  const img = new Image();
  img.onload = () => uploadImage(img);
  img.src = url;
}
function makeTestPattern(size = 512): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const x = c.getContext("2d")!;
  const cells = 16, s = size / cells;
  for (let r = 0; r < cells; r++) for (let cl = 0; cl < cells; cl++) {
    x.fillStyle = (r + cl) % 2 ? "#1c1c1c" : "#e8e8e8";
    x.fillRect(cl * s, r * s, s, s);
  }
  return c;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const params: DiffParams = { ...DEFAULT_DIFF };
const ui = {
  running: true,
  itersPerFrame: 25,
  brushRadius: 0.18,
  brushStrength: 0.15,
  maxZ: 4,
  minZ: 0.35,
  brushMode: "magnify" as "magnify" | "minify" | "erase",
  showImage: true,
  showField: true,
  showMesh: true,
};
let iterCounter = 0;
let painting = false;

function syncPos(): void {
  for (let i = 0; i < N * N; i++) { posArr[2 * i] = mesh.x[i]; posArr[2 * i + 1] = mesh.y[i]; }
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, posArr);
}
function syncZ(): void {
  gl.bindBuffer(gl.ARRAY_BUFFER, zBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, mesh.z);
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

function draw(): void {
  resize();
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.07, 0.07, 0.08, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(prog);
  gl.bindVertexArray(vao);
  gl.uniform1f(loc.u_minZ, ui.minZ);
  gl.uniform1f(loc.u_maxZ, ui.maxZ);

  if (ui.showImage) {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(loc.u_tex, 0);
    gl.uniform1i(loc.u_mode, 1);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, triBuf);
    gl.drawElements(gl.TRIANGLES, triCount, gl.UNSIGNED_INT, 0);
  }
  if (ui.showField) {
    gl.uniform1i(loc.u_mode, 2);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, triBuf);
    gl.drawElements(gl.TRIANGLES, triCount, gl.UNSIGNED_INT, 0);
    gl.disable(gl.BLEND);
  }
  if (ui.showMesh) {
    gl.uniform1i(loc.u_mode, 0);
    gl.uniform4f(loc.u_color, 0.62, 0.64, 0.72, 0.5);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineBuf);
    gl.drawElements(gl.LINES, lineCount, gl.UNSIGNED_INT, 0);
    gl.disable(gl.BLEND);
  }
}

let frameNo = 0;
function frame(): void {
  if (ui.running) {
    for (let k = 0; k < ui.itersPerFrame; k++) diffuseStep(mesh, params, iterCounter++);
    syncPos();
  }
  draw();
  if ((frameNo++ & 15) === 0) el("rms").textContent = rmsError(mesh).toFixed(3);
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------
// Paint interaction
// ---------------------------------------------------------------------------
function paintAt(e: PointerEvent): void {
  const rect = canvas.getBoundingClientRect();
  const cx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const cy = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
  if (paintField(mesh, cx, cy, ui.brushRadius, ui.brushStrength, ui.minZ, ui.maxZ, ui.brushMode))
    syncZ();
}
// Brush-size ring that follows the cursor over the canvas (and is tinted by the
// current mode, so you can see which brush is active before clicking).
const ring = el<HTMLDivElement>("brushring");
let lastClientX = 0, lastClientY = 0;
function ringColor(): string {
  return ui.brushMode === "magnify" ? "rgba(216,84,72,0.95)"   // warm
    : ui.brushMode === "minify" ? "rgba(70,130,205,0.95)"      // cool
    : "rgba(208,208,212,0.9)";                                 // neutral (erase)
}
function updateRing(clientX: number, clientY: number): void {
  lastClientX = clientX; lastClientY = clientY;
  const rect = canvas.getBoundingClientRect();
  // brushRadius is in clip units (fraction of half-width); clip [-1,1] spans the
  // canvas width, so screen diameter = brushRadius * width.
  const d = ui.brushRadius * rect.width;
  ring.style.width = `${d}px`;
  ring.style.height = `${d}px`;
  ring.style.left = `${clientX}px`;
  ring.style.top = `${clientY}px`;
  ring.style.borderColor = ringColor();
  ring.style.display = "block";
}

canvas.addEventListener("pointerdown", (e) => {
  painting = true;
  try { canvas.setPointerCapture(e.pointerId); } catch { /* capture is best-effort */ }
  updateRing(e.clientX, e.clientY);
  paintAt(e); // paint on the very first press, capture or not
});
canvas.addEventListener("pointermove", (e) => {
  updateRing(e.clientX, e.clientY);
  if (painting) paintAt(e);
});
canvas.addEventListener("pointerenter", (e) => updateRing(e.clientX, e.clientY));
canvas.addEventListener("pointerleave", () => { if (!painting) ring.style.display = "none"; });
function endPaint(e: PointerEvent): void {
  if (!painting) return;
  painting = false;
  try { canvas.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
}
canvas.addEventListener("pointerup", endPaint);
canvas.addEventListener("pointercancel", endPaint);

canvas.addEventListener("dragover", (e) => e.preventDefault());
canvas.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = e.dataTransfer?.files?.[0];
  if (!f) return;
  const img = new Image();
  img.onload = () => { uploadImage(img); URL.revokeObjectURL(img.src); };
  img.src = URL.createObjectURL(f);
});

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
function el<T extends HTMLElement>(id: string): T { return document.getElementById(id) as T; }
function bindSlider(id: string, set: (v: number) => void, fmtId: string, fmt?: (v: number) => string) {
  const s = el<HTMLInputElement>(id);
  const update = () => { const v = parseFloat(s.value); set(v); el(fmtId).textContent = fmt ? fmt(v) : v.toFixed(2); };
  s.addEventListener("input", update); update();
}

bindSlider("brushRadius", (v) => {
  ui.brushRadius = v;
  if (ring.style.display === "block") updateRing(lastClientX, lastClientY);
}, "brushRadiusVal");
bindSlider("brushStrength", (v) => (ui.brushStrength = v), "brushStrengthVal");
bindSlider("maxZ", (v) => (ui.maxZ = v), "maxZVal", (v) => `${v.toFixed(1)}×`);
bindSlider("refine", (v) => (params.refineCoeff = v), "refineVal");
bindSlider("iters", (v) => (ui.itersPerFrame = Math.round(v)), "itersVal", (v) => String(Math.round(v)));

el<HTMLSelectElement>("brushMode").addEventListener("change", (e) => {
  const t = e.target as HTMLSelectElement;
  ui.brushMode = t.value as typeof ui.brushMode;
  t.blur(); // drop focus so the next canvas press registers immediately
  if (ring.style.display === "block") ring.style.borderColor = ringColor();
});
el<HTMLSelectElement>("weight").addEventListener("change", (e) => {
  const t = e.target as HTMLSelectElement;
  params.weightByMag = parseInt(t.value, 10);
  t.blur();
});
el<HTMLInputElement>("pin").addEventListener("change", (e) => {
  params.pinBoundary = (e.target as HTMLInputElement).checked;
});
el<HTMLInputElement>("showImage").addEventListener("change", (e) => { ui.showImage = (e.target as HTMLInputElement).checked; });
el<HTMLInputElement>("showField").addEventListener("change", (e) => { ui.showField = (e.target as HTMLInputElement).checked; });
el<HTMLInputElement>("showMesh").addEventListener("change", (e) => { ui.showMesh = (e.target as HTMLInputElement).checked; });

el<HTMLSelectElement>("source").addEventListener("change", (e) => {
  const t = e.target as HTMLSelectElement;
  if (t.value === "metro") loadURL("/dcMetro.png");
  else if (t.value === "test") uploadImage(makeTestPattern());
  t.blur();
});

const runBtn = el<HTMLButtonElement>("run");
function syncRun(): void { runBtn.textContent = ui.running ? "Pause solver" : "Run solver"; }
runBtn.addEventListener("click", () => { ui.running = !ui.running; syncRun(); });
el<HTMLButtonElement>("step").addEventListener("click", () => {
  for (let k = 0; k < ui.itersPerFrame; k++) diffuseStep(mesh, params, iterCounter++);
  syncPos();
});
el<HTMLButtonElement>("resetMesh").addEventListener("click", () => { resetPositions(mesh); iterCounter = 0; syncPos(); });
el<HTMLButtonElement>("clearField").addEventListener("click", () => { clearField(mesh); syncZ(); });

// ---------------------------------------------------------------------------
// Go
// ---------------------------------------------------------------------------
syncRun();
syncPos();
syncZ();
loadURL("/dcMetro.png");
requestAnimationFrame(frame);
