// main.ts — data-driven magnification. Planes fly over a zone; their density (or
// proximity) becomes the magnification field; the diffusion solver deforms the
// mesh to satisfy it; the planes are drawn riding the warp. Reuses the Engine B
// solver (diffuse.ts) verbatim — only the FIELD SOURCE changes (data, not paint).

import { makeDiffMesh, diffuseStep, rmsError, resetPositions, DEFAULT_DIFF, type DiffMesh, type DiffParams } from "./diffuse";
import { makePlanes, stepPlanes, computeDensityField, computeProximityField, warpLookup, type Planes } from "./field";

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
in vec2 a_pos; in vec2 a_uv; in float a_z;
out vec2 v_uv; out float v_z;
void main(){ v_uv=a_uv; v_z=a_z; gl_Position=vec4(a_pos,0.0,1.0); }`;
const FS = `#version 300 es
precision highp float;
in vec2 v_uv; in float v_z; out vec4 frag;
uniform sampler2D u_tex; uniform int u_mode; uniform vec4 u_color; uniform float u_maxZ;
vec4 heat(float z){ float up=clamp((z-1.0)/max(u_maxZ-1.0,1e-3),0.0,1.0);
  return vec4(0.80,0.30,0.16, up*0.55); }
void main(){
  if(u_mode==1) frag=texture(u_tex,v_uv);
  else if(u_mode==2) frag=heat(v_z);
  else frag=u_color;
}`;
const prog = program(VS, FS);
const loc = {
  a_pos: gl.getAttribLocation(prog, "a_pos"), a_uv: gl.getAttribLocation(prog, "a_uv"), a_z: gl.getAttribLocation(prog, "a_z"),
  u_tex: gl.getUniformLocation(prog, "u_tex"), u_mode: gl.getUniformLocation(prog, "u_mode"),
  u_color: gl.getUniformLocation(prog, "u_color"), u_maxZ: gl.getUniformLocation(prog, "u_maxZ"),
};

// ---------------------------------------------------------------------------
// Mesh + buffers (same scheme as the diffusion demo)
// ---------------------------------------------------------------------------
const N = 64;
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
  const step = Math.max(1, Math.round(N / 24)); const seg: number[] = [];
  for (let r = 0; r < N; r += step) for (let c = 0; c < N - 1; c++) seg.push(r * N + c, r * N + c + 1);
  for (let c = 0; c < N; c += step) for (let r = 0; r < N - 1; r++) seg.push(r * N + c, (r + 1) * N + c);
  const idx = new Uint32Array(seg); lineCount = idx.length;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineBuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
}

// plane markers — their own VAO/buffer (a_pos only; flat-colour triangles)
const MAX_PLANES = 600;
const planeVerts = new Float32Array(MAX_PLANES * 3 * 2);
const planesVao = gl.createVertexArray()!;
gl.bindVertexArray(planesVao);
const planesBuf = gl.createBuffer()!;
gl.bindBuffer(gl.ARRAY_BUFFER, planesBuf); gl.bufferData(gl.ARRAY_BUFFER, planeVerts, gl.DYNAMIC_DRAW);
gl.enableVertexAttribArray(loc.a_pos); gl.vertexAttribPointer(loc.a_pos, 2, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

// ---------------------------------------------------------------------------
// Texture (zone background)
// ---------------------------------------------------------------------------
const texture = gl.createTexture()!;
gl.bindTexture(gl.TEXTURE_2D, texture);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([26, 28, 32, 255]));
{ const img = new Image(); img.onload = () => { gl.bindTexture(gl.TEXTURE_2D, texture); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img); gl.generateMipmap(gl.TEXTURE_2D); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); }; img.src = import.meta.env.BASE_URL + "dcMetro.png"; }

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const params: DiffParams = { ...DEFAULT_DIFF };
let planes: Planes = makePlanes(160);
const ui = {
  running: true,
  itersPerFrame: 18,        // the field moves each frame; enough sweeps to track it
  method: "density" as "density" | "proximity",
  volume: 4,
  radius: 0.2,
  speed: 1,
  showImage: true,
  showField: false,
  showMesh: true,
  showPlanes: true,
};

function syncPos(): void {
  for (let i = 0; i < N * N; i++) { posArr[2 * i] = mesh.x[i]; posArr[2 * i + 1] = mesh.y[i]; }
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, posArr);
}
function syncZ(): void { gl.bindBuffer(gl.ARRAY_BUFFER, zBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, mesh.z); }

const tmp: [number, number] = [0, 0];
function buildPlanes(): number {
  const size = 0.016; let o = 0;
  for (let i = 0; i < planes.n; i++) {
    warpLookup(mesh, planes.x[i], planes.y[i], tmp);
    const cx = tmp[0], cy = tmp[1];
    let hx = planes.vx[i], hy = planes.vy[i]; const L = Math.hypot(hx, hy) || 1; hx /= L; hy /= L;
    const px = -hy, py = hx;
    planeVerts[o++] = cx + hx * size * 1.7; planeVerts[o++] = cy + hy * size * 1.7;
    planeVerts[o++] = cx - hx * size + px * size * 0.9; planeVerts[o++] = cy - hy * size + py * size * 0.9;
    planeVerts[o++] = cx - hx * size - px * size * 0.9; planeVerts[o++] = cy - hy * size - py * size * 0.9;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, planesBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, planeVerts);
  return planes.n * 3;
}

function computeField(): void {
  if (ui.method === "density") computeDensityField(mesh, planes, ui.volume, ui.radius);
  else computeProximityField(mesh, planes, ui.volume, ui.radius);
  syncZ();
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
function draw(planeCount: number): void {
  resize();
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.07, 0.07, 0.08, 1); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(prog);
  gl.uniform1f(loc.u_maxZ, Math.max(2, ui.volume));

  gl.bindVertexArray(vao);
  if (ui.showImage) {
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(loc.u_tex, 0); gl.uniform1i(loc.u_mode, 1);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, triBuf); gl.drawElements(gl.TRIANGLES, triCount, gl.UNSIGNED_INT, 0);
  }
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  if (ui.showField) {
    gl.uniform1i(loc.u_mode, 2);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, triBuf); gl.drawElements(gl.TRIANGLES, triCount, gl.UNSIGNED_INT, 0);
  }
  if (ui.showMesh) {
    gl.uniform1i(loc.u_mode, 0); gl.uniform4f(loc.u_color, 0.62, 0.64, 0.72, ui.showImage ? 0.35 : 0.5);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineBuf); gl.drawElements(gl.LINES, lineCount, gl.UNSIGNED_INT, 0);
  }
  if (ui.showPlanes) {
    gl.bindVertexArray(planesVao);
    gl.uniform1i(loc.u_mode, 0); gl.uniform4f(loc.u_color, 0.95, 0.42, 0.26, 0.95);
    gl.drawArrays(gl.TRIANGLES, 0, planeCount);
  }
  gl.disable(gl.BLEND);
}

let lastT = 0, frameNo = 0;
function frame(t: number): void {
  const dt = lastT ? Math.min(0.05, (t - lastT) / 1000) : 0.016; lastT = t;
  if (ui.running) {
    stepPlanes(planes, dt, ui.speed);
    computeField();
    for (let k = 0; k < ui.itersPerFrame; k++) diffuseStep(mesh, params, frameNo + k);
    syncPos();
  }
  const pc = buildPlanes();
  draw(pc);
  if ((frameNo++ & 15) === 0) el("rms").textContent = rmsError(mesh).toFixed(3);
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
function el<T extends HTMLElement>(id: string): T { return document.getElementById(id) as T; }
function bindSlider(id: string, set: (v: number) => void, fmtId: string, fmt?: (v: number) => string) {
  const s = el<HTMLInputElement>(id);
  const update = () => { const v = parseFloat(s.value); set(v); el(fmtId).textContent = fmt ? fmt(v) : v.toFixed(2); };
  s.addEventListener("input", update); update();
}
bindSlider("count", (v) => { const n = Math.round(v); if (n !== planes.n) planes = makePlanes(n); }, "countVal", (v) => String(Math.round(v)));
bindSlider("speed", (v) => (ui.speed = v), "speedVal");
bindSlider("volume", (v) => (ui.volume = v), "volumeVal", (v) => `${v.toFixed(1)}×`);
bindSlider("radius", (v) => (ui.radius = v), "radiusVal");
bindSlider("refine", (v) => (params.refineCoeff = v), "refineVal");
bindSlider("clampEps", (v) => (params.clampEps = v), "clampEpsVal");
bindSlider("iters", (v) => (ui.itersPerFrame = Math.round(v)), "itersVal", (v) => String(Math.round(v)));

el<HTMLSelectElement>("method").addEventListener("change", (e) => {
  const t = e.target as HTMLSelectElement; ui.method = t.value as typeof ui.method; t.blur();
});
el<HTMLInputElement>("showImage").addEventListener("change", (e) => { ui.showImage = (e.target as HTMLInputElement).checked; });
el<HTMLInputElement>("showField").addEventListener("change", (e) => { ui.showField = (e.target as HTMLInputElement).checked; });
el<HTMLInputElement>("showMesh").addEventListener("change", (e) => { ui.showMesh = (e.target as HTMLInputElement).checked; });
el<HTMLInputElement>("showPlanes").addEventListener("change", (e) => { ui.showPlanes = (e.target as HTMLInputElement).checked; });

const runBtn = el<HTMLButtonElement>("run");
function syncRun() { runBtn.textContent = ui.running ? "Pause" : "Play"; }
runBtn.addEventListener("click", () => { ui.running = !ui.running; syncRun(); });
el<HTMLButtonElement>("reseed").addEventListener("click", () => { planes = makePlanes(planes.n, (frameNo % 1000) + 2); });
el<HTMLButtonElement>("resetMesh").addEventListener("click", () => { resetPositions(mesh); syncPos(); });

// ---------------------------------------------------------------------------
// Go
// ---------------------------------------------------------------------------
syncRun(); syncPos(); computeField();
requestAnimationFrame(frame);
