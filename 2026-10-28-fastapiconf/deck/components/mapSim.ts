// Pure 2-D simulation for the ContourMap component: a ball rolling on a bumpy bowl seen from above.
// Kept free of React so the same code runs in the browser, in print mode, and from a script.

export interface Vec { x: number; y: number }

export interface Zone { c: Vec; r: number; label: string }

export interface Sim {
  t: number;
  p: Vec;
  v: Vec;
  /** seconds since the last poke, or -1 if never poked */
  sincePoke: number;
  /** unit direction of the last poke */
  pokeDir: Vec;
  pokes: number;
  /** seconds the ball has been continuously slow */
  slowTime: number;
  /** seconds spent settled at the goal */
  settled: number;
  done: boolean;
  /** the ball is pressed against a wall or a zone */
  touching: boolean;
  /** places the ball has already been stuck; a poke must not lead back to one */
  rests: Vec[];
  /** recent positions, oldest first */
  trail: Vec[];
  trailAcc: number;
}

/** The map covers x in [0, 1] and y in [0, 0.46] (same aspect as the 1000 x 460 SVG). */
export const MAP = { w: 1, h: 0.46, margin: 0.03, ballR: 0.011 };

/** The bowl is an ellipse inscribed in the map; the ball stays inside it. */
export const BOWL = { cx: MAP.w / 2, cy: MAP.h / 2, a: MAP.w / 2 - MAP.margin, b: MAP.h / 2 - MAP.margin };

export const GOAL: Vec = { x: 0.56, y: 0.23 };
export const START: Vec = { x: 0.11, y: 0.32 };

/** Local minima that "work, but..." */
export const DIPS: Vec[] = [
  { x: 0.2, y: 0.36 },
  { x: 0.82, y: 0.1 },
  { x: 0.47, y: 0.1 },
];

/** Hard constraints: the ball may not enter these. They sit on the slope, not over a minimum. */
export const ZONES: Zone[] = [
  { c: { x: 0.4, y: 0.31 }, r: 0.075, label: 'breaks the spec' },
  { c: { x: 0.7, y: 0.31 }, r: 0.06, label: 'insecure' },
];

function g2(p: Vec, c: Vec, w: number): number {
  return Math.exp(-(((p.x - c.x) ** 2 + (p.y - c.y) ** 2) / (w * w)));
}

export function f(p: Vec): number {
  let z = 1.6 * ((p.x - GOAL.x) ** 2 + (p.y - GOAL.y) ** 2);
  z -= 0.2 * g2(p, GOAL, 0.05);
  z -= 0.26 * g2(p, DIPS[0]!, 0.075);
  z -= 0.26 * g2(p, DIPS[1]!, 0.05);
  z -= 0.2 * g2(p, DIPS[2]!, 0.05);
  // a hill so the paths curve
  z += 0.16 * g2(p, { x: 0.68, y: 0.14 }, 0.05);
  return z;
}

function grad(p: Vec): Vec {
  const h = 1e-4;
  return {
    x: (f({ x: p.x + h, y: p.y }) - f({ x: p.x - h, y: p.y })) / (2 * h),
    y: (f({ x: p.x, y: p.y + h }) - f({ x: p.x, y: p.y - h })) / (2 * h),
  };
}

const G = 0.7;
const DAMP = 2.6;
const DT = 1 / 240;
const POKE_SHOW = 1.0;
const HOLD = 4.0;
const SLOW_V = 0.05;
const SLOW_FOR = 2.0;
const TRAIL_EVERY = 1 / 30;
const TRAIL_LEN = 90;

export function init(): Sim {
  return {
    t: 0, p: { ...START }, v: { x: 0, y: 0 }, sincePoke: -1, pokeDir: { x: 1, y: 0 }, pokes: 0,
    slowTime: 0, settled: 0, done: false, touching: false, rests: [], trail: [], trailAcc: 0,
  };
}

/** One physics step with walls and exclusion zones; returns whether the ball is touching something. */
function physics(p: Vec, v: Vec): boolean {
  const gr = grad(p);
  v.x += (-G * gr.x - DAMP * v.x) * DT;
  v.y += (-G * gr.y - DAMP * v.y) * DT;
  p.x += v.x * DT;
  p.y += v.y * DT;
  let touching = false;
  // the rim of the bowl: an ellipse shrunk by the ball's radius
  const a = BOWL.a - MAP.ballR;
  const b = BOWL.b - MAP.ballR;
  const ex = (p.x - BOWL.cx) / a;
  const ey = (p.y - BOWL.cy) / b;
  const r = Math.hypot(ex, ey);
  if (r > 1) {
    p.x = BOWL.cx + (ex / r) * a;
    p.y = BOWL.cy + (ey / r) * b;
    let nx = (p.x - BOWL.cx) / (a * a);
    let ny = (p.y - BOWL.cy) / (b * b);
    const nl = Math.hypot(nx, ny) || 1;
    nx /= nl;
    ny /= nl;
    const vn = v.x * nx + v.y * ny;
    if (vn > 0) { v.x -= 1.3 * vn * nx; v.y -= 1.3 * vn * ny; }
    touching = true;
  }
  for (const z of ZONES) {
    const dx = p.x - z.c.x;
    const dy = p.y - z.c.y;
    const d = Math.hypot(dx, dy);
    const min = z.r + MAP.ballR;
    if (d < min) {
      const nx = dx / (d || 1);
      const ny = dy / (d || 1);
      p.x = z.c.x + nx * min;
      p.y = z.c.y + ny * min;
      const vn = v.x * nx + v.y * ny;
      if (vn < 0) { v.x -= vn * nx; v.y -= vn * ny; }
      touching = true;
    }
  }
  return touching;
}

/** Dry-run an impulse and report where the ball comes to rest. */
function tryImpulse(p0: Vec, dir: Vec, speed: number): { p: Vec; z: number; touching: boolean; moved: boolean } {
  const p = { ...p0 };
  const v = { x: dir.x * speed, y: dir.y * speed };
  let slow = 0;
  let touching = false;
  for (let k = 0; k < 5 * 240; k++) {
    touching = physics(p, v);
    slow = Math.hypot(v.x, v.y) < SLOW_V ? slow + DT : 0;
    if (slow > 0.4) break;
  }
  return { p, z: f(p), touching, moved: Math.hypot(p.x - p0.x, p.y - p0.y) > 0.05 };
}

/** The finger accepts a nudge that ends somewhere new and lower. */
function acceptable(r: { p: Vec; z: number; touching: boolean; moved: boolean }, here: number, rests: Vec[]): boolean {
  if (!r.moved) return false;
  if (rests.some((q) => Math.hypot(q.x - r.p.x, q.y - r.p.y) < 0.04)) return false;
  return r.z < here - 0.03;
}

const N_ANGLES = 24;

/** The smallest impulse (found by dry-running the physics) that lands the ball somewhere lower than it is now. */
function poke(s: Sim) {
  const here = f(s.p);
  const toGoal = Math.atan2(GOAL.y - s.p.y, GOAL.x - s.p.x);
  const angles: number[] = [];
  for (let i = 0; i < N_ANGLES; i++) angles.push(toGoal + (i * 2 * Math.PI) / N_ANGLES);
  // prefer directions closest to the goal at equal speed
  angles.sort((a, b) => Math.abs(Math.atan2(Math.sin(a - toGoal), Math.cos(a - toGoal))) - Math.abs(Math.atan2(Math.sin(b - toGoal), Math.cos(b - toGoal))));
  s.rests.push({ ...s.p });
  // the gentlest nudge that gets somewhere new; among equal nudges, the one aimed closest to the goal
  for (let speed = 0.25; speed < 1.6; speed *= 1.12) {
    for (const a of angles) {
      const dir = { x: Math.cos(a), y: Math.sin(a) };
      if (acceptable(tryImpulse(s.p, dir, speed), here, s.rests)) {
        s.v = { x: dir.x * speed * 1.05, y: dir.y * speed * 1.05 };
        s.pokeDir = dir;
        s.sincePoke = 0;
        s.pokes += 1;
        s.slowTime = 0;
        return;
      }
    }
  }
  const dir = { x: Math.cos(toGoal), y: Math.sin(toGoal) };
  s.v = { x: dir.x * 1.6, y: dir.y * 1.6 };
  s.pokeDir = dir;
  s.sincePoke = 0;
  s.pokes += 1;
  s.slowTime = 0;
}

export function atGoal(p: Vec): boolean {
  return Math.hypot(p.x - GOAL.x, p.y - GOAL.y) < 0.025;
}

export function step(s: Sim, seconds: number): Sim {
  if (s.done) return s;
  const n = Math.max(1, Math.round(seconds / DT));
  for (let i = 0; i < n; i++) {
    s.touching = physics(s.p, s.v);
    s.t += DT;
    if (s.sincePoke >= 0) s.sincePoke += DT;
    s.slowTime = Math.hypot(s.v.x, s.v.y) < SLOW_V ? s.slowTime + DT : 0;
    s.trailAcc += DT;
    if (s.trailAcc >= TRAIL_EVERY) {
      s.trailAcc = 0;
      s.trail.push({ ...s.p });
      if (s.trail.length > TRAIL_LEN) s.trail.shift();
    }
    if (atGoal(s.p) && s.slowTime > 0) {
      s.settled += DT;
      if (s.settled > HOLD) s.done = true;
    } else if (!atGoal(s.p) && s.slowTime > SLOW_FOR) {
      poke(s);
    }
  }
  return s;
}

export function fingerVisible(s: Sim): boolean {
  return s.sincePoke >= 0 && s.sincePoke < POKE_SHOW;
}

/** Run the simulation from scratch for `seconds`; used for the print frame and for scripts. */
export function simulate(seconds: number): Sim {
  return step(init(), seconds);
}
