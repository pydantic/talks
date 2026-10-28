import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import skiPrint from '../assets/ski-slope.png';
import { BEND, GRADE, HALF, PERIOD, SKIER_R, centre, fingerVisible, h, init, step, type Sim } from './skiSim';

// World coordinates: x across the slope (u), y up (snow height), z = -s so downhill is away from the camera.

function isPrint() {
  if (typeof window === 'undefined') return false;
  if (window.matchMedia?.('print').matches) return true;
  return /HeadlessChrome/.test(navigator.userAgent);
}

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

const VERT = `
  varying vec3 vWorld;
  varying vec3 vNormal;
  varying float vDist;
  void main() {
    vWorld = position;
    vNormal = normal;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vDist = -mv.z;
    gl_Position = projectionMatrix * mv;
  }
`;

// The snow: plain on the open slope, contour rings only inside the dips (measured against the tilted plane),
// hatched outside the fences, faint cross lines so the motion reads, and fog into the slide background.
const FRAG = `
  uniform vec3 uSnow;
  uniform vec3 uOut;
  uniform vec3 uRing;
  uniform vec3 uDip;
  uniform vec3 uHatch;
  uniform vec3 uFog;
  uniform float uGrade;
  uniform float uBend;
  uniform float uPeriod;
  uniform float uHalf;
  uniform float uFogNear;
  uniform float uFogFar;
  varying vec3 vWorld;
  varying vec3 vNormal;
  varying float vDist;

  float centre(float s) { return uBend * sin(6.2831853 * s / uPeriod); }
  float lineAt(float x, float width) {
    float d = abs(fract(x + 0.5) - 0.5);
    return 1.0 - smoothstep(0.0, fwidth(x) * width, d);
  }

  void main() {
    float s = -vWorld.z;
    float u = vWorld.x;
    float local = vWorld.y + uGrade * s;

    float off = abs(u - centre(s)) - uHalf;
    float inside = 1.0 - smoothstep(-0.01, 0.01, off);
    vec3 col = mix(uOut, uSnow, inside);

    float near = 1.0 - smoothstep(3.0, 11.0, vDist);
    col = mix(col, uHatch, lineAt((u + s) * 1.6, 1.0) * (1.0 - inside) * 0.45 * near);
    col = mix(col, uRing, lineAt(s, 1.0) * inside * 0.10);

    float depth = clamp(-local / 0.34, 0.0, 1.0);
    col = mix(col, uDip, smoothstep(0.0, 1.0, depth) * 0.75);
    float rings = lineAt(local / 0.05, 1.3) * smoothstep(-0.008, -0.03, local);
    col = mix(col, uRing, rings * 0.85);

    float light = 0.82 + 0.18 * dot(normalize(vNormal), normalize(vec3(0.4, 1.0, 0.6)));
    col *= light;
    col = mix(col, uFog, smoothstep(uFogNear, uFogFar, vDist));
    gl_FragColor = vec4(col, 1.0);
    #include <colorspace_fragment>
  }
`;

const NS = 200;
const X_FAR = 16;
const X_NEAR = BEND + HALF + 0.6;

/** Across-slope sample positions: fine where the course and its dips are, coarse on the flat beyond. */
function uSamples(): number[] {
  const out: number[] = [];
  for (let u = -X_FAR; u < -X_NEAR; u += 0.5) out.push(u);
  for (let u = -X_NEAR; u < X_NEAR; u += 0.05) out.push(u);
  for (let u = X_NEAR; u <= X_FAR + 1e-6; u += 0.5) out.push(u);
  return out;
}
const US = uSamples();

function snowGeometry(k: number): THREE.BufferGeometry {
  const s0 = k * PERIOD;
  const nu = US.length;
  const pos = new Float32Array((NS + 1) * nu * 3);
  let i = 0;
  for (let a = 0; a <= NS; a++) {
    const s = s0 + (a / NS) * PERIOD;
    for (const u of US) {
      pos[i++] = u;
      pos[i++] = h(s, u);
      pos[i++] = -s;
    }
  }
  const idx: number[] = [];
  for (let a = 0; a < NS; a++) {
    for (let b = 0; b < nu - 1; b++) {
      const p = a * nu + b;
      idx.push(p, p + 1, p + nu, p + 1, p + nu + 1, p + nu);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

const POST_GAP = 0.5;
const POST_H = 0.26;

function buildChunk(k: number, snowMat: THREE.Material, fenceMat: THREE.Material): THREE.Group {
  const grp = new THREE.Group();
  grp.add(new THREE.Mesh(snowGeometry(k), snowMat));
  const s0 = k * PERIOD;
  const nPosts = Math.round(PERIOD / POST_GAP);
  const posts = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.018, 0.018, POST_H, 8), fenceMat, nPosts * 2);
  const m = new THREE.Matrix4();
  let n = 0;
  for (const side of [-1, 1]) {
    for (let i = 0; i < nPosts; i++) {
      const s = s0 + i * POST_GAP;
      const u = centre(s) + side * HALF;
      m.makeTranslation(u, h(s, u) + POST_H / 2, -s);
      posts.setMatrixAt(n++, m);
    }
    for (const y of [POST_H * 0.55, POST_H]) {
      const pts: THREE.Vector3[] = [];
      for (let i = 0; i <= PERIOD * 4; i++) {
        const s = s0 + i / 4;
        const u = centre(s) + side * HALF;
        pts.push(new THREE.Vector3(u, h(s, u) + y, -s));
      }
      const tube = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), PERIOD * 8, 0.009, 5, false);
      grp.add(new THREE.Mesh(tube, fenceMat));
    }
  }
  grp.add(posts);
  return grp;
}

function makeFinger(): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d')!;
  ctx.font = '96px "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('\u{1F449}', 64, 70);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  sp.scale.set(0.45, 0.45, 1);
  sp.renderOrder = 10;
  return sp;
}

function dispose(o: THREE.Object3D) {
  o.traverse((c) => {
    if (c instanceof THREE.Mesh || c instanceof THREE.InstancedMesh) c.geometry.dispose();
  });
}

const TRAIL_MAX = 240;
const TRAIL_W = 0.022;
const HIT_SHOW = 0.9;

// Full bleed: the scene covers the slide from below the title to the bottom edge.
const FRAME: React.CSSProperties = { position: 'absolute', left: 0, right: 0, bottom: 0, height: '76%', overflow: 'hidden' };

export default function SkiSlope() {
  const host = useRef<HTMLDivElement>(null);
  // Headless Chrome has no WebGL, so the PDF gets a screenshot of the live scene instead.
  const [fallback, setFallback] = useState(isPrint);
  useEffect(() => {
    const el = host.current;
    if (!el || fallback) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      setFallback(true);
      return;
    }
    const pal = {
      bg: cssVar('--bg-slide', '#15101a'),
      surface: cssVar('--surface', '#241a2e'),
      ring: cssVar('--accent-tertiary', '#b388ff'),
      fence: cssVar('--accent-secondary', '#ff8a4c'),
      accent: cssVar('--accent', '#e520a8'),
      aqua: cssVar('--accent-aqua', '#4ad7c5'),
    };
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(new THREE.Color(pal.bg));
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(46, 2.3, 0.1, 40);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x2a2035, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2, 5, 3);
    scene.add(sun);

    const bg = new THREE.Color(pal.bg);
    const snowMat = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uSnow: { value: new THREE.Color(pal.surface).multiplyScalar(1.6) },
        uOut: { value: new THREE.Color(pal.surface).multiplyScalar(1.6) },
        uRing: { value: new THREE.Color(pal.ring) },
        uDip: { value: new THREE.Color(pal.ring).lerp(bg, 0.55) },
        uHatch: { value: new THREE.Color(pal.fence).lerp(bg, 0.55) },
        uFog: { value: bg },
        uGrade: { value: GRADE },
        uBend: { value: BEND },
        uPeriod: { value: PERIOD },
        uHalf: { value: HALF },
        uFogNear: { value: 9 },
        uFogFar: { value: 24 },
      },
    });
    const fenceMat = new THREE.MeshBasicMaterial({ color: pal.fence });

    // skier: a pink body on aqua skis, with a soft shadow
    const skier = new THREE.Group();
    const body = new THREE.Mesh(new THREE.CapsuleGeometry(SKIER_R, 0.1, 6, 12), new THREE.MeshStandardMaterial({ color: pal.accent, roughness: 0.45 }));
    body.position.y = SKIER_R + 0.07;
    skier.add(body);
    const skiMat = new THREE.MeshBasicMaterial({ color: pal.aqua });
    for (const dx of [-0.04, 0.04]) {
      const ski = new THREE.Mesh(new THREE.BoxGeometry(0.025, 0.01, 0.32), skiMat);
      ski.position.set(dx, 0.018, 0);
      skier.add(ski);
    }
    const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.11, 24), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35, depthWrite: false }));
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.y = 0.004;
    skier.add(shadow);
    scene.add(skier);

    const finger = makeFinger();
    finger.visible = false;
    scene.add(finger);

    // trail: fades out behind the skier
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL_MAX * 2 * 3), 3));
    trailGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(TRAIL_MAX * 2 * 4), 4));
    const trailIdx: number[] = [];
    for (let i = 0; i < TRAIL_MAX - 1; i++) trailIdx.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
    trailGeo.setIndex(trailIdx);
    const trail = new THREE.Mesh(trailGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
    trail.frustumCulled = false;
    scene.add(trail);
    const trailColor = new THREE.Color(pal.accent);

    // fence strikes: an orange ring that spreads from the point of contact
    const hitRings: THREE.Mesh[] = [];
    for (let i = 0; i < 6; i++) {
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.07, 0.1, 40), new THREE.MeshBasicMaterial({ color: pal.fence, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
      ring.rotation.x = -Math.PI / 2;
      ring.visible = false;
      scene.add(ring);
      hitRings.push(ring);
    }

    const chunks = new Map<number, THREE.Group>();
    const ensureChunks = (s: number) => {
      const k = Math.floor(s / PERIOD);
      for (let i = k - 1; i <= k + 2; i++) {
        if (!chunks.has(i)) {
          const g = buildChunk(i, snowMat, fenceMat);
          chunks.set(i, g);
          scene.add(g);
        }
      }
      for (const [i, g] of chunks) {
        if (i < k - 1) {
          scene.remove(g);
          dispose(g);
          chunks.delete(i);
        }
      }
    };

    const UP = new THREE.Vector3(0, 1, 0);
    const surfaceNormal = new THREE.Vector3();
    const tilt = new THREE.Quaternion();
    const turn = new THREE.Quaternion();
    const camPos = new THREE.Vector3();
    const camLook = new THREE.Vector3();
    let heading = 0;
    const apply = (sim: Sim, dt: number, snap: boolean) => {
      ensureChunks(sim.p.s);
      const y = h(sim.p.s, sim.p.u);
      skier.position.set(sim.p.u, y, -sim.p.s);
      if (Math.hypot(sim.v.s, sim.v.u) > 0.05) heading = Math.atan2(sim.v.u, sim.v.s);
      // sit the skier on the snow: tilt to the surface normal, then turn to the direction of travel
      const e = 0.02;
      const dhs = (h(sim.p.s + e, sim.p.u) - h(sim.p.s - e, sim.p.u)) / (2 * e);
      const dhu = (h(sim.p.s, sim.p.u + e) - h(sim.p.s, sim.p.u - e)) / (2 * e);
      surfaceNormal.set(-dhu, 1, dhs).normalize();
      tilt.setFromUnitVectors(UP, surfaceNormal);
      turn.setFromAxisAngle(UP, -heading);
      skier.quaternion.copy(tilt).multiply(turn);
      body.rotation.z = THREE.MathUtils.clamp(-sim.v.u * 0.3, -0.4, 0.4);

      finger.visible = fingerVisible(sim);
      if (finger.visible) {
        const d = sim.pokeDir;
        finger.position.set(sim.p.u - d.u * 0.32, y + 0.14, -(sim.p.s - d.s * 0.32));
        finger.material.rotation = Math.atan2(d.s, d.u);
      }

      const pos = trailGeo.getAttribute('position') as THREE.BufferAttribute;
      const col = trailGeo.getAttribute('color') as THREE.BufferAttribute;
      const n = Math.min(sim.trail.length, TRAIL_MAX);
      const base = sim.trail.length - n;
      for (let i = 0; i < n; i++) {
        const p = sim.trail[base + i]!;
        const q = sim.trail[base + Math.min(i + 1, n - 1)]!;
        const r = sim.trail[base + Math.max(i - 1, 0)]!;
        let ds = q.s - r.s;
        let du = q.u - r.u;
        const len = Math.hypot(ds, du) || 1;
        ds /= len;
        du /= len;
        const w = TRAIL_W * (0.4 + 0.6 * (i / n));
        for (const side of [-1, 1]) {
          const ps = p.s + side * du * w;
          const pu = p.u - side * ds * w;
          const j = 2 * i + (side > 0 ? 1 : 0);
          pos.setXYZ(j, pu, h(ps, pu) + 0.008, -ps);
          col.setXYZW(j, trailColor.r, trailColor.g, trailColor.b, 0.9 * ((i + 1) / n) ** 1.3);
        }
      }
      pos.needsUpdate = true;
      col.needsUpdate = true;
      trailGeo.setDrawRange(0, Math.max(0, (n - 1) * 6));

      hitRings.forEach((ring, i) => {
        const hit = sim.hits[sim.hits.length - 1 - i];
        const age = hit ? sim.t - hit.t : Infinity;
        ring.visible = age < HIT_SHOW;
        if (!hit || !ring.visible) return;
        const k = age / HIT_SHOW;
        ring.position.set(hit.u, h(hit.s, hit.u) + 0.012, -hit.s);
        ring.scale.setScalar(1 + 5 * k);
        (ring.material as THREE.MeshBasicMaterial).opacity = 1 - k;
      });

      // chase camera: behind and above, looking a little way down the slope
      const target = new THREE.Vector3(sim.p.u, y + 2.2, -sim.p.s + 3.4);
      const look = new THREE.Vector3(sim.p.u * 0.6 + centre(sim.p.s + 3) * 0.4, y - 1.0, -sim.p.s - 2.6);
      if (snap) {
        camPos.copy(target);
        camLook.copy(look);
      } else {
        const a = 1 - Math.exp(-dt * 2.5);
        camPos.lerp(target, a);
        camLook.lerp(look, a);
      }
      camera.position.copy(camPos);
      camera.lookAt(camLook);
    };

    const resize = () => {
      const w = el.clientWidth;
      const hgt = el.clientHeight;
      if (!w || !hgt) return;
      renderer.setSize(w, hgt, false);
      camera.aspect = w / hgt;
      camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(() => {
      resize();
      renderer.render(scene, camera);
    });
    ro.observe(el);

    const sim = init();
    apply(sim, 0, true);
    let last = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      step(sim, dt);
      apply(sim, dt, false);
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      for (const g of chunks.values()) dispose(g);
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, [fallback]);
  if (fallback) {
    return <img src={skiPrint} alt="A skier on an endless slope between two bending fences, with dips to get stuck in" style={{ ...FRAME, display: 'block', objectFit: 'cover', width: '100%' }} />;
  }
  return <div ref={host} style={FRAME} />;
}
