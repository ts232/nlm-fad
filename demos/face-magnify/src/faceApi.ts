// faceApi.ts — load face-api.js + TinyFaceDetector / 68-landmark weights from our
// own /public assets (vendored from face-api.js@0.22.2).

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

// The 68 landmarks are typed by feature (the standard iBUG/dlib groups). face-api
// exposes a getter per group, so we can magnify just the eyes, just the mouth, …
export interface FaceFeatures {
  jaw: Landmark[];      // 17 pts (0–16)
  leftBrow: Landmark[]; // 5  (17–21)
  rightBrow: Landmark[];// 5  (22–26)
  nose: Landmark[];     // 9  (27–35)
  leftEye: Landmark[];  // 6  (36–41)
  rightEye: Landmark[]; // 6  (42–47)
  mouth: Landmark[];    // 20 (48–67)
}

// Detect faces; return the landmark points grouped by feature, each normalized
// to material space [-1,1] (x right, y up) to match the mesh.
export async function detectFaces(
  img: HTMLImageElement | HTMLCanvasElement, inputSize = 512, scoreThreshold = 0.3,
): Promise<{ features: FaceFeatures; faces: number }> {
  const fa = await loadFaceApi();
  const w = (img as HTMLImageElement).naturalWidth || (img as HTMLCanvasElement).width;
  const h = (img as HTMLImageElement).naturalHeight || (img as HTMLCanvasElement).height;
  const norm = (p: { x: number; y: number }): Landmark => ({ x: (p.x / w) * 2 - 1, y: 1 - (p.y / h) * 2 });
  const g: FaceFeatures = { jaw: [], leftBrow: [], rightBrow: [], nose: [], leftEye: [], rightEye: [], mouth: [] };
  const dets = await fa
    .detectAllFaces(img, new fa.TinyFaceDetectorOptions({ inputSize, scoreThreshold }))
    .withFaceLandmarks();
  for (const det of dets) {
    const L = det.landmarks;
    for (const p of L.getJawOutline()) g.jaw.push(norm(p));
    for (const p of L.getLeftEyeBrow()) g.leftBrow.push(norm(p));
    for (const p of L.getRightEyeBrow()) g.rightBrow.push(norm(p));
    for (const p of L.getNose()) g.nose.push(norm(p));
    for (const p of L.getLeftEye()) g.leftEye.push(norm(p));
    for (const p of L.getRightEye()) g.rightEye.push(norm(p));
    for (const p of L.getMouth()) g.mouth.push(norm(p));
  }
  return { features: g, faces: dets.length };
}
