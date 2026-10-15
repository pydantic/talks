// Pure simulation for the Landscape component: a ball rolling on a 1-D objective function.
// Kept free of React so the same code runs in the browser, in print mode, and from a script.

export type Mode = 'bowl' | 'slope';

export interface Sim {
  mode: Mode;
  t: number;
  x: number;
  v: number;
  /** seconds since the last poke, or -1 if never poked */
  sincePoke: number;
  pokeDir: number;
  pokes: number;
  /** seconds the ball has been continuously slow */
  slowTime: number;
  /** seconds spent settled at the goal (bowl only) */
  settled: number;
  done: boolean;
}

export function gauss(x: number, c: number, w: number): number {
  return Math.exp(-(((x - c) / w) ** 2));
}

export const BOWL = { lo: 0.08, hi: 0.92, goal: 0.5, start: 0.1 };

export function bowl(x: number): number {
  return 4 * (x - 0.5) ** 2 + 0.45 * gauss(x, 0.33, 0.06) + 0.5 * gauss(x, 0.68, 0.06);
}

export const SLOPE_PERIOD = 1.0;
export const SLOPE_GRADE = 0.55;

export function slope(x: number): number {
  let y = -SLOPE_GRADE * x;
  const k0 = Math.floor(x / SLOPE_PERIOD) - 1;
  for (let k = k0; k <= k0 + 3; k++) {
    y += 0.26 * gauss(x, k * SLOPE_PERIOD + 0.6, 0.06);
  }
  return y;
}

export function f(mode: Mode, x: number): number {
  return mode === 'bowl' ? bowl(x) : slope(x);
}

function df(mode: Mode, x: number): number {
  const h = 1e-4;
  return (f(mode, x + h) - f(mode, x - h)) / (2 * h);
}

const G = 2.5;
const DAMP = 2.0;
const DT = 1 / 240;
const POKE_SHOW = 0.8;
const HOLD = 2.0;
const SLOW_V = 0.06;
const SLOW_FOR = 0.35;

export function init(mode: Mode): Sim {
  return {
    mode, t: 0, x: mode === 'bowl' ? BOWL.start : 0.05, v: 0,
    sincePoke: -1, pokeDir: 1, pokes: 0, slowTime: 0, settled: 0, done: false,
  };
}

function physics(mode: Mode, x: number, v: number): [number, number] {
  const a = -G * df(mode, x) - DAMP * v;
  const v2 = v + a * DT;
  return [x + v2 * DT, v2];
}

/** Position of the highest point of the curve between a and b. */
function peakBetween(mode: Mode, a: number, b: number): number {
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  let best = -Infinity;
  let at = lo;
  for (let x = lo; x <= hi; x += 0.002) {
    const y = f(mode, x);
    if (y > best) { best = y; at = x; }
  }
  return at;
}

/** Smallest impulse (found by dry-running the physics) that carries the ball over the next peak. */
function escapeVelocity(mode: Mode, x0: number, dir: number, target: number): number {
  const peak = peakBetween(mode, x0, target);
  let v0 = 0.2;
  for (let i = 0; i < 40; i++) {
    let x = x0;
    let v = dir * v0;
    let cleared = false;
    for (let k = 0; k < 4 * 240; k++) {
      [x, v] = physics(mode, x, v);
      if ((x - peak) * dir > 0.01) { cleared = true; break; }
      if (v * dir < 0) break;
    }
    if (cleared) return dir * v0 * 1.05;
    v0 *= 1.12;
  }
  return dir * v0;
}

function poke(s: Sim) {
  const target = s.mode === 'bowl' ? BOWL.goal : s.x + SLOPE_PERIOD;
  const dir = Math.sign(target - s.x) || 1;
  s.v = escapeVelocity(s.mode, s.x, dir, target);
  s.sincePoke = 0;
  s.pokeDir = dir;
  s.pokes += 1;
  s.slowTime = 0;
}

export function step(s: Sim, seconds: number): Sim {
  if (s.done) return s;
  const n = Math.max(1, Math.round(seconds / DT));
  for (let i = 0; i < n; i++) {
    [s.x, s.v] = physics(s.mode, s.x, s.v);
    s.t += DT;
    if (s.sincePoke >= 0) s.sincePoke += DT;
    s.slowTime = Math.abs(s.v) < SLOW_V ? s.slowTime + DT : 0;

    if (s.mode === 'bowl') {
      if (s.x < BOWL.lo) { s.x = BOWL.lo; s.v = Math.abs(s.v) * 0.3; }
      if (s.x > BOWL.hi) { s.x = BOWL.hi; s.v = -Math.abs(s.v) * 0.3; }
      const atGoal = Math.abs(s.x - BOWL.goal) < 0.03;
      if (atGoal && s.slowTime > 0) {
        s.settled += DT;
        if (s.settled > HOLD) s.done = true;
      } else if (!atGoal && s.slowTime > SLOW_FOR) {
        poke(s);
      }
    } else if (s.slowTime > SLOW_FOR) {
      poke(s);
    }
  }
  return s;
}

export function fingerVisible(s: Sim): boolean {
  return s.sincePoke >= 0 && s.sincePoke < POKE_SHOW;
}

/** Run the simulation from scratch for `seconds`; used for the print frame and for scripts. */
export function simulate(mode: Mode, seconds: number): Sim {
  const s = init(mode);
  return step(s, seconds);
}
