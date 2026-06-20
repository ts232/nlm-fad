# Magnify a face from its landmarks — FAD demo

Image features → magnification. A face is detected, its **68 facial landmarks**
become the data points, and the diffusion solver (Engine B) magnifies where the
landmarks cluster — so the **eyes, nose and mouth bulge outward** while the
periphery compresses. The data-driven magnification of
[`../data-flow`](../data-flow/README.md), but the "data" is now *features
extracted from an image* (the thing this whole port was building toward).

## Run

```bash
npm install
npm run dev      # → http://localhost:1998  (pinned port)
```

## How it works

- **Landmarks** (`src/faceApi.ts`) — loads **face-api.js@0.22.2** + TinyFaceDetector
  / 68-landmark weights from local `/public` assets (vendored, no CDN), mirroring
  Entoptica's `src/core/field/faceApi.ts`. Runs on **TensorFlow.js's WebGL (GPU)
  backend** — not WebAssembly. `detectAllFaces(img).withFaceLandmarks()` → 68
  points per face, **grouped by feature** and normalized to mesh space `[-1,1]`.
- **Feature selection** — the 68 landmarks are typed (face-api's
  `getMouth/getLeftEye/getNose/getJawOutline/…`), so checkboxes pick which groups
  (eyes / brows / nose / mouth / jaw) become the data points. Magnify just the
  eyes, just the mouth, etc.
- **Field from the landmarks** — the same `field.ts` used by `data-flow`: a
  **density** splat (feature clusters → magnification) or **proximity** halos.
  The selected points cluster at their features, so those regions get the field.
- **Solve + render** — `diffuse.ts` (Engine B) relaxes the mesh to the field;
  the face image rides the warp, with the landmark points and mesh overlaid.

So the *only* thing different from the planes demo is the field **source**:
`computeDensityField(mesh, landmarks, …)` instead of `…(mesh, planes, …)`.

## Controls

- **Field** — density / proximity, magnification strength, influence radius.
- **Solver** — refine coeff, step cap, iterations/frame; Play/Pause, Reset mesh.
- **Display** — face image / field heat overlay / mesh / landmarks.
- **Load a face** — pick a file, or **drop any front-facing face** onto the canvas.

## Notes

- Detection runs in-browser on load (~1–2 s the first time: model fetch + TF.js WebGL init).
  Status is shown in the panel ("N landmarks", "no face found", etc.).
- Best on a reasonably front-facing, well-lit face (TinyFaceDetector). Profile /
  tiny / occluded faces may not detect.
- The vendored face-api assets (`public/vendor`, `public/models`) are MIT
  (face-api.js) and committed so the demo is self-contained. `sample-face.png`
  is the same test face Entoptica uses.
- This is the template for any feature→field source: swap the landmark detector
  for edges / saliency / corners and nothing downstream changes.
