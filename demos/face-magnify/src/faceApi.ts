// faceApi.ts — load face-api.js + TinyFaceDetector / 68-landmark weights from our
// own /public assets (vendored from face-api.js@0.22.2). Mirrors Entoptica's
// src/core/field/faceApi.ts so this demo's face stage matches the engine's.

const FACEAPI_URL = import.meta.env.BASE_URL + "vendor/face-api.min.js";
const MODEL_URL = import.meta.env.BASE_URL + "models";
let loadP: Promise<unknown> | null = null;

function loadFaceApi(): Promise<any> {
  const w = window as any;
  if (w.faceapi?.nets?.faceLandmark68Net?.params) return Promise.resolve(w.faceapi);
  if (loadP) return loadP as Promise<any>;
  loadP = new Promise<any>((resolve, reject) => {
    if (w.faceapi) return resolve(w.faceapi);
    const s = document.createElement("script");
    s.src = FACEAPI_URL;
    s.onload = () => resolve(w.faceapi);
    s.onerror = () => reject(new Error("failed to load face-api.js"));
    document.head.appendChild(s);
  }).then(async (fa: any) => {
    await fa.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    await fa.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    return fa;
  });
  return loadP as Promise<any>;
}

export interface Landmark { x: number; y: number }

// Detect faces; return all 68-landmark points normalized to material space
// [-1,1] (x right, y up) to match the mesh.
export async function detectLandmarks(
  img: HTMLImageElement | HTMLCanvasElement, inputSize = 512, scoreThreshold = 0.3,
): Promise<Landmark[]> {
  const fa = await loadFaceApi();
  const w = (img as HTMLImageElement).naturalWidth || (img as HTMLCanvasElement).width;
  const h = (img as HTMLImageElement).naturalHeight || (img as HTMLCanvasElement).height;
  const dets = await fa
    .detectAllFaces(img, new fa.TinyFaceDetectorOptions({ inputSize, scoreThreshold }))
    .withFaceLandmarks();
  const pts: Landmark[] = [];
  for (const det of dets) {
    for (const p of det.landmarks.positions) {
      pts.push({ x: (p.x / w) * 2 - 1, y: 1 - (p.y / h) * 2 });
    }
  }
  return pts;
}
