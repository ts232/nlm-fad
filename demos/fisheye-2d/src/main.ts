// main.ts — WebGL2 renderer + interaction for the closed-form 2D fisheye demo.
// The warp math lives entirely in warp.ts; here we just push the warped lattice
// to the GPU and draw it as (a) a textured mesh and/or (b) its grid lines.

import {
  makeLattice,
  warpLattice,
  DEFAULT_PARAMS,
  type FisheyeParams,
  type KernelName,
  type Lattice,
} from "./warp";

// ---------------------------------------------------------------------------
// GL boilerplate
// ---------------------------------------------------------------------------
const canvas = document.getElementById("gl") as HTMLCanvasElement;
const gl = canvas.getContext("webgl2", { antialias: true, premultipliedAlpha: false })!;
if (!gl) throw new Error("WebGL2 not available");

function compile(type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(sh) ?? "shader compile failed");
  return sh;
}
function program(vsrc: string, fsrc: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vsrc));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fsrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(p) ?? "link failed");
  return p;
}

const VS = `#version 300 es
in vec2 a_pos;   // pre-warped clip-space position
in vec2 a_uv;    // home/texture coordinate
out vec2 v_uv;
void main() {
  v_uv = a_uv;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_tex;
uniform int u_mode;     // 0 = flat color, 1 = texture
uniform vec4 u_color;
void main() {
  if (u_mode == 1) frag = texture(u_tex, v_uv);
  else frag = u_color;
}`;

const prog = program(VS, FS);
const loc = {
  a_pos: gl.getAttribLocation(prog, "a_pos"),
  a_uv: gl.getAttribLocation(prog, "a_uv"),
  u_tex: gl.getUniformLocation(prog, "u_tex"),
  u_mode: gl.getUniformLocation(prog, "u_mode"),
  u_color: gl.getUniformLocation(prog, "u_color"),
};

// ---------------------------------------------------------------------------
// Lattice + GPU buffers
// ---------------------------------------------------------------------------
let meshN = 96; // warp mesh cells per side — drives texture fidelity AND faceting
let lat: Lattice = makeLattice(meshN);

const vao = gl.createVertexArray()!;
gl.bindVertexArray(vao);

const posBuf = gl.createBuffer()!;
gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
gl.bufferData(gl.ARRAY_BUFFER, lat.pos, gl.DYNAMIC_DRAW);
gl.enableVertexAttribArray(loc.a_pos);
gl.vertexAttribPointer(loc.a_pos, 2, gl.FLOAT, false, 0, 0);

const uvBuf = gl.createBuffer()!;
gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
gl.bufferData(gl.ARRAY_BUFFER, lat.uv, gl.STATIC_DRAW);
gl.enableVertexAttribArray(loc.a_uv);
gl.vertexAttribPointer(loc.a_uv, 2, gl.FLOAT, false, 0, 0);

// Triangle indices (two tris per cell) for the textured mesh.
function buildTriIndices(n: number): Uint32Array {
  const side = n + 1;
  const idx = new Uint32Array(n * n * 6);
  let k = 0;
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const a = row * side + col;
      const b = a + 1;
      const c = a + side;
      const d = c + 1;
      idx[k++] = a; idx[k++] = b; idx[k++] = c;
      idx[k++] = b; idx[k++] = d; idx[k++] = c;
    }
  }
  return idx;
}
const triBuf = gl.createBuffer()!;
let triCount = 0;

// Line indices: draw every `step`-th row/column as a finely-subdivided polyline
// (connecting consecutive mesh vertices) so warped grid lines render as curves.
const lineBuf = gl.createBuffer()!;
let lineCount = 0;
function buildLineIndices(n: number, step: number): void {
  const side = n + 1;
  const seg: number[] = [];
  for (let row = 0; row <= n; row += step) {
    for (let col = 0; col < n; col++) {
      seg.push(row * side + col, row * side + col + 1);
    }
  }
  for (let col = 0; col <= n; col += step) {
    for (let row = 0; row < n; row++) {
      seg.push(row * side + col, (row + 1) * side + col);
    }
  }
  const idx = new Uint32Array(seg);
  lineCount = idx.length;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
}

// Keep ~24 visible grid lines regardless of mesh resolution, so the overlay
// stays readable while the mesh itself (and the texture) genuinely changes.
function gridLineStep(n: number): number {
  return Math.max(1, Math.round(n / 24));
}

// Rebuild the whole warp mesh at a new resolution: lattice, positions, UVs,
// triangle indices, and the grid-line overlay. The VAO's attribute pointers
// already reference posBuf/uvBuf, so re-uploading their data is enough.
function rebuildMesh(n: number): void {
  meshN = n;
  lat = makeLattice(n);
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
  gl.bufferData(gl.ARRAY_BUFFER, lat.uv, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, lat.pos, gl.DYNAMIC_DRAW);
  const tri = buildTriIndices(n);
  triCount = tri.length;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, triBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, tri, gl.STATIC_DRAW);
  buildLineIndices(n, gridLineStep(n));
  markDirty();
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------
function makeTexture(): WebGLTexture {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  // 1x1 placeholder until an image loads.
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
    new Uint8Array([180, 180, 180, 255]));
  return tex;
}
const texture = makeTexture();

type ImgSrc = HTMLImageElement | HTMLCanvasElement;

// Largest texture this GPU accepts per side (typically 8192 or 16384).
const MAX_TEX = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
// Anisotropic filtering — keeps detail crisp where the warp stretches the texture.
const aniso = gl.getExtension("EXT_texture_filter_anisotropic");
const maxAniso = aniso ? (gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT) as number) : 1;
console.info(`[fisheye] MAX_TEXTURE_SIZE = ${MAX_TEX}px, max anisotropy = ${maxAniso} (larger images are downscaled to fit)`);

// Downscale (preserving aspect) only if the image exceeds the GPU limit.
function fitToMax(src: ImgSrc, w: number, h: number): ImgSrc {
  const m = Math.max(w, h);
  if (m <= MAX_TEX) return src;
  const s = MAX_TEX / m;
  const c = document.createElement("canvas");
  c.width = Math.round(w * s);
  c.height = Math.round(h * s);
  c.getContext("2d")!.drawImage(src, 0, 0, c.width, c.height);
  return c;
}

function uploadImage(src: ImgSrc, w: number, h: number): void {
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, fitToMax(src, w, h));
  // Trilinear mipmaps: a high-res source minified into a small viewport stays
  // clean in the context view, while the magnified focus still pulls full detail.
  // (WebGL2 allows mipmaps on non-power-of-two textures — no size restriction.)
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  if (aniso) gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, maxAniso));
  setAspect(w / h); // canvas adopts the image's true aspect ratio
  draw();
}

function loadURL(url: string): void {
  const img = new Image();
  img.onload = () => uploadImage(img, img.naturalWidth, img.naturalHeight);
  img.src = url; // same-origin (served by Vite) — no crossOrigin needed
}

// Procedural checkerboard + concentric rings — great for *seeing* magnification,
// and (with square cells + true circles) for confirming aspect ratio is honoured.
function makeTestPattern(w = 512, h = 512): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const x = c.getContext("2d")!;
  const s = Math.min(w, h) / 16; // square cells regardless of aspect
  const cols = Math.ceil(w / s), rows = Math.ceil(h / s);
  for (let row = 0; row < rows; row++)
    for (let col = 0; col < cols; col++) {
      x.fillStyle = (row + col) % 2 ? "#1c1c1c" : "#e8e8e8";
      x.fillRect(col * s, row * s, s, s);
    }
  x.strokeStyle = "#c81e3c"; // single muted accent
  x.lineWidth = 2;
  for (let r = s; r < Math.min(w, h) * 0.72; r += s) {
    x.beginPath();
    x.arc(w / 2, h / 2, r, 0, Math.PI * 2);
    x.stroke();
  }
  return c;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const params: FisheyeParams = { ...DEFAULT_PARAMS, focus: { x: 0, y: 0 } };
const ui = {
  showImage: true,
  showGrid: true,
  followMouse: true,
  linearOn: false, // flat (constant-magnification) legible centre
  linearExtent: 0.6, // its width, as a fraction of the feasible max
};
let dirty = true;
function markDirty() { dirty = true; }

rebuildMesh(meshN); // initial mesh (triangles + grid lines + buffers)

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------
// Size the canvas (CSS px) to contain the current aspect ratio within the stage.
const stage = document.getElementById("stage") as HTMLElement;
function layout(): void {
  const pad = 16; // matches #stage padding
  const sw = Math.max(1, stage.clientWidth - pad * 2);
  const sh = Math.max(1, stage.clientHeight - pad * 2);
  const a = params.aspect;
  let w = sw, h = sw / a;
  if (h > sh) { h = sh; w = sh * a; }
  canvas.style.width = `${Math.round(w)}px`;
  canvas.style.height = `${Math.round(h)}px`;
}

function setAspect(a: number): void {
  params.aspect = a; // layout() picks it up on the next frame
  markDirty();
}

function resize(): void {
  layout();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function draw(): void {
  resize();
  if (dirty) {
    warpLattice(lat, params);
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, lat.pos);
    dirty = false;
  }

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.07, 0.07, 0.08, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.useProgram(prog);
  gl.bindVertexArray(vao);

  if (ui.showImage) {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(loc.u_tex, 0);
    gl.uniform1i(loc.u_mode, 1);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, triBuf);
    gl.drawElements(gl.TRIANGLES, triCount, gl.UNSIGNED_INT, 0);
  }

  if (ui.showGrid) {
    gl.uniform1i(loc.u_mode, 0);
    // light lines on the dark backdrop; darker if drawn over the image
    gl.uniform4f(loc.u_color, ui.showImage ? 0.05 : 0.55, ui.showImage ? 0.05 : 0.6,
      ui.showImage ? 0.07 : 0.7, ui.showImage ? 0.55 : 1.0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineBuf);
    gl.drawElements(gl.LINES, lineCount, gl.UNSIGNED_INT, 0);
    gl.disable(gl.BLEND);
  }
}

function frame(): void {
  draw();
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------
// Focus moves only while a pointer button is held (drag), so releasing freezes
// it in place. The "Drag to move focus" checkbox locks it entirely when off.
let dragging = false;

function setFocusFromEvent(e: PointerEvent): void {
  const rect = canvas.getBoundingClientRect();
  // map the (possibly non-square) canvas to clip space [-1,1], y up
  params.focus.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  params.focus.y = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
  markDirty();
}

canvas.addEventListener("pointerdown", (e) => {
  if (!ui.followMouse) return; // focus locked
  dragging = true;
  canvas.setPointerCapture(e.pointerId);
  canvas.style.cursor = "grabbing";
  setFocusFromEvent(e);
});

canvas.addEventListener("pointermove", (e) => {
  if (dragging) setFocusFromEvent(e);
});

function endDrag(e: PointerEvent): void {
  if (!dragging) return;
  dragging = false;
  try { canvas.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
  canvas.style.cursor = "grab";
}
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

// Drag-and-drop an image file onto the canvas.
canvas.addEventListener("dragover", (e) => e.preventDefault());
canvas.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer?.files?.[0];
  if (file) loadFile(file);
});

function loadFile(file: File): void {
  const img = new Image();
  img.onload = () => {
    uploadImage(img, img.naturalWidth, img.naturalHeight);
    ui.showImage = true;
    syncUI();
    URL.revokeObjectURL(img.src);
  };
  img.src = URL.createObjectURL(file);
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
function el<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

function bindSlider(id: string, set: (v: number) => void, fmtId?: string, fmt?: (v: number) => string) {
  const s = el<HTMLInputElement>(id);
  const update = () => {
    const v = parseFloat(s.value);
    set(v);
    if (fmtId) el(fmtId).textContent = fmt ? fmt(v) : v.toFixed(2);
    markDirty();
  };
  s.addEventListener("input", update);
  update();
}

bindSlider("beta", (v) => (params.beta = v), "betaVal");
bindSlider("gamma", (v) => (params.gamma = v), "gammaVal");
bindSlider("filter", (v) => (params.filterWeight = v), "filterVal");
bindSlider("meshN", (v) => rebuildMesh(Math.round(v)), "meshNVal", (v) => String(Math.round(v)));

el<HTMLSelectElement>("kernel").addEventListener("change", (e) => {
  params.kernel = (e.target as HTMLSelectElement).value as KernelName;
  markDirty();
});

// Flat (linear) magnification centre — constant-magnification legible zone.
function applyLinear(): void {
  params.linearCenter = ui.linearOn ? ui.linearExtent : 0;
  markDirty();
}
el<HTMLInputElement>("linearCenter").addEventListener("change", (e) => {
  ui.linearOn = (e.target as HTMLInputElement).checked;
  applyLinear();
});
bindSlider("linExtent", (v) => { ui.linearExtent = v; applyLinear(); }, "linExtentVal");

el<HTMLInputElement>("showImage").addEventListener("change", (e) => {
  ui.showImage = (e.target as HTMLInputElement).checked; markDirty();
});
el<HTMLInputElement>("showGrid").addEventListener("change", (e) => {
  ui.showGrid = (e.target as HTMLInputElement).checked; markDirty();
});
el<HTMLInputElement>("followMouse").addEventListener("change", (e) => {
  ui.followMouse = (e.target as HTMLInputElement).checked;
});

el<HTMLSelectElement>("source").addEventListener("change", (e) => {
  const v = (e.target as HTMLSelectElement).value;
  if (v === "metro") loadURL(import.meta.env.BASE_URL + "dcMetro.png");
  else if (v === "test") { const c = makeTestPattern(512, 512); uploadImage(c, c.width, c.height); }
  else if (v === "wide") { const c = makeTestPattern(1024, 384); uploadImage(c, c.width, c.height); }
  else if (v === "tall") { const c = makeTestPattern(384, 1024); uploadImage(c, c.width, c.height); }
  ui.showImage = true; syncUI();
});

el<HTMLInputElement>("file").addEventListener("change", (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) loadFile(f);
});

el<HTMLButtonElement>("reset").addEventListener("click", () => {
  params.beta = DEFAULT_PARAMS.beta;
  params.gamma = DEFAULT_PARAMS.gamma;
  params.filterWeight = DEFAULT_PARAMS.filterWeight;
  params.kernel = DEFAULT_PARAMS.kernel;
  ui.linearOn = false;
  ui.linearExtent = 0.6;
  params.linearCenter = 0;
  syncUI();
  markDirty();
});

function syncUI(): void {
  el<HTMLInputElement>("beta").value = String(params.beta);
  el("betaVal").textContent = params.beta.toFixed(2);
  el<HTMLInputElement>("gamma").value = String(params.gamma);
  el("gammaVal").textContent = params.gamma.toFixed(2);
  el<HTMLInputElement>("filter").value = String(params.filterWeight);
  el("filterVal").textContent = params.filterWeight.toFixed(2);
  el<HTMLSelectElement>("kernel").value = params.kernel;
  el<HTMLInputElement>("linearCenter").checked = ui.linearOn;
  el<HTMLInputElement>("linExtent").value = String(ui.linearExtent);
  el("linExtentVal").textContent = ui.linearExtent.toFixed(2);
  el<HTMLInputElement>("meshN").value = String(meshN);
  el("meshNVal").textContent = String(meshN);
  el<HTMLInputElement>("showImage").checked = ui.showImage;
  el<HTMLInputElement>("showGrid").checked = ui.showGrid;
  el<HTMLInputElement>("followMouse").checked = ui.followMouse;
}

// ---------------------------------------------------------------------------
// Go
// ---------------------------------------------------------------------------
syncUI();
loadURL(import.meta.env.BASE_URL + "dcMetro.png"); // iconic FAD demo image
requestAnimationFrame(frame);
