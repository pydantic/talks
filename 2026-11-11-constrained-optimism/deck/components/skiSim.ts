// Pure simulation for the SkiSlope component: a skier on an endless, flat, tilted slope.
// Coordinates: `s` runs downhill (grows forever), `u` runs across the slope.
// The slope is flat across; the only features are shallow dips (local minima) and the fences (constraints),
// which bend left and right while gravity pulls straight downhill, so the skier runs into them on the bends.
// Kept free of React and three.js so the same code runs in the browser and from a script.

export interface P { s: number; u: number }
export interface Hit { s: number; u: number; t: number }
export interface Dip { s: number; u: number; depth: number; width: number }

export interface Sim {
  t: number;
  p: P;
  v: P;
  sincePoke: number;
  pokeDir: P;
  pokes: number;
  /** seconds the skier has spent within a small patch of snow */
  stuckTime: number;
  stuckAnchor: P;
  touching: boolean;
  hits: Hit[];
  rests: P[];
  trail: P[];
  trailAcc: number;
}

/** Half the width between the fences. */
export const HALF = 1.25;
export const SKIER_R = 0.06;
/** The course repeats every PERIOD units downhill. */
export const PERIOD = 16;
export const GRADE = 0.25;
/** How far the course swings left and right. */
export const BEND = 1.9;

/** Centre line of the course at s. */
export function centre(s: number): number {
  return BEND * Math.sin((2 * Math.PI * s) / PERIOD);
}

/** Dips within one period, placed relative to the centre line: "works, but..." */
const DIP_SPEC: Dip[] = [
  { s: 1.5, u: 0.5, depth: 0.3, width: 0.45 },
  { s: 5.6, u: -0.5, depth: 0.34, width: 0.48 },
  { s: 11.6, u: 0.35, depth: 0.34, width: 0.48 },
];
/** The same dips with absolute u. */
export const DIPS: Dip[] = DIP_SPEC.map((d) => ({ ...d, u: centre(d.s) + d.u }));

function g(ds: number, du: number, w: number): number {
  return Math.exp(-((ds * ds + du * du) / (w * w)));
}

/** Height of the snow at (s, u). */
export function h(s: number, u: number): number {
  let z = -GRADE * s;
  const k0 = Math.floor(s / PERIOD) - 1;
  for (let k = k0; k <= k0 + 2; k++) {
    for (const d of DIPS) {
      const ds = s - (k * PERIOD + d.s);
      if (Math.abs(ds) > 3 * d.width) continue;
      z -= d.depth * g(ds, u - d.u, d.width);
    }
  }
  return z;
}

function grad(p: P): P {
  const e = 1e-4;
  return {
    s: (h(p.s + e, p.u) - h(p.s - e, p.u)) / (2 * e),
    u: (h(p.s, p.u + e) - h(p.s, p.u - e)) / (2 * e),
  };
}

const G = 4.0;
const DAMP = 0.62;
const REST = 0.6;
const DIP_DRAG = 14;
const DT = 1 / 240;
const POKE_SHOW = 1.0;
const STUCK_FOR = 2.5;
const STUCK_R = 0.5;
const TRAIL_EVERY = 1 / 30;
const TRAIL_LEN = 240;

export function init(): Sim {
  const p = { s: 0, u: 0.2 };
  return {
    t: 0, p, v: { s: 0.6, u: 0 }, sincePoke: -1, pokeDir: { s: 1, u: 0 }, pokes: 0,
    stuckTime: 0, stuckAnchor: { ...p }, touching: false, hits: [], rests: [], trail: [], trailAcc: 0,
  };
}

/** One physics step; returns the speed at which the skier struck a fence (0 if it did not). */
function physics(p: P, v: P): number {
  const gr = grad(p);
  // deep snow in the dips: extra drag, so a skier that rolls in stays in
  const depth = Math.max(0, -(h(p.s, p.u) + GRADE * p.s));
  const damp = DAMP + DIP_DRAG * depth;
  v.s += (-G * gr.s - damp * v.s) * DT;
  v.u += (-G * gr.u - damp * v.u) * DT;
  p.s += v.s * DT;
  p.u += v.u * DT;
  // The fences bounce the skier. A fence slants with the course, so its normal has an along-slope
  // component: a skier running straight downhill into a bend is thrown back across the course.
  const lim = HALF - SKIER_R;
  const c = centre(p.s);
  const side = p.u < c - lim ? -1 : p.u > c + lim ? 1 : 0;
  if (side === 0) return 0;
  p.u = c + side * lim;
  const e = 1e-3;
  const slant = (centre(p.s + e) - centre(p.s - e)) / (2 * e);
  const norm = Math.hypot(slant, 1);
  const ns = (side * slant) / norm;
  const nu = -side / norm;
  const vn = v.s * ns + v.u * nu;
  if (vn < 0) {
    v.s -= (1 + REST) * vn * ns;
    v.u -= (1 + REST) * vn * nu;
  }
  return Math.max(-vn, 1e-6);
}

/** Dry-run a nudge for a couple of seconds and report where the skier ends up. */
function tryImpulse(p0: P, dir: P, speed: number): P {
  const p = { ...p0 };
  const v = { s: dir.s * speed, u: dir.u * speed };
  for (let k = 0; k < 2.5 * 240; k++) physics(p, v);
  return p;
}

const N_ANGLES = 16;

/** The gentlest nudge that gets the skier out and somewhere it has not been stuck; prefer straight downhill. */
function poke(s: Sim) {
  s.rests.push({ ...s.p });
  if (s.rests.length > 6) s.rests.shift();
  const angles: number[] = [];
  for (let i = 0; i < N_ANGLES; i++) angles.push((i * 2 * Math.PI) / N_ANGLES);
  const wrap = (a: number) => Math.abs(Math.atan2(Math.sin(a), Math.cos(a)));
  angles.sort((a, b) => wrap(a) - wrap(b));
  for (let speed = 0.4; speed < 3; speed *= 1.15) {
    for (const a of angles) {
      const dir = { s: Math.cos(a), u: Math.sin(a) };
      const r = tryImpulse(s.p, dir, speed);
      if (Math.hypot(r.s - s.p.s, r.u - s.p.u) < 0.9) continue;
      if (s.rests.some((q) => Math.hypot(q.s - r.s, q.u - r.u) < 0.4)) continue;
      apply(s, dir, speed * 1.05);
      return;
    }
  }
  apply(s, { s: 1, u: 0 }, 3);
}

function apply(s: Sim, dir: P, speed: number) {
  s.v = { s: dir.s * speed, u: dir.u * speed };
  s.pokeDir = dir;
  s.sincePoke = 0;
  s.pokes += 1;
  s.stuckTime = 0;
  s.stuckAnchor = { ...s.p };
}

export function step(s: Sim, seconds: number): Sim {
  const n = Math.max(1, Math.round(seconds / DT));
  for (let i = 0; i < n; i++) {
    const impact = physics(s.p, s.v);
    const last = s.hits[s.hits.length - 1];
    if (impact > 0.15 && !s.touching && (!last || s.t - last.t > 0.5)) {
      s.hits.push({ s: s.p.s, u: s.p.u, t: s.t });
      if (s.hits.length > 6) s.hits.shift();
    }
    s.touching = impact > 0;
    s.t += DT;
    if (s.sincePoke >= 0) s.sincePoke += DT;
    if (Math.hypot(s.p.s - s.stuckAnchor.s, s.p.u - s.stuckAnchor.u) < STUCK_R) s.stuckTime += DT;
    else { s.stuckAnchor = { ...s.p }; s.stuckTime = 0; }
    s.trailAcc += DT;
    if (s.trailAcc >= TRAIL_EVERY) {
      s.trailAcc = 0;
      s.trail.push({ ...s.p });
      if (s.trail.length > TRAIL_LEN) s.trail.shift();
    }
    if (s.stuckTime > STUCK_FOR) poke(s);
  }
  return s;
}

export function fingerVisible(s: Sim): boolean {
  return s.sincePoke >= 0 && s.sincePoke < POKE_SHOW;
}
