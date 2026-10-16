import { useEffect, useMemo, useRef, useState } from 'react';
import { DIPS, GOAL, MAP, ZONES, f, fingerVisible, init, simulate, step, type Sim, type Vec } from './mapSim';

const W = 1000;
const H = 400;
const S = W / MAP.w; // px per map unit
const R = MAP.ballR * S;

// Headless Chrome (the PDF build) does not fire `beforeprint`, so detect it up front and render a fixed frame.
function isPrint() {
  if (typeof window === 'undefined') return false;
  if (window.matchMedia?.('print').matches) return true;
  return /HeadlessChrome/.test(navigator.userAgent);
}

// Frozen frame: the ball pressed against the first fence, finger still visible.
const PRINT_T = 5.20;

function useSim(): Sim {
  const ref = useRef<Sim>(isPrint() ? simulate(PRINT_T) : init());
  const [, setTick] = useState(0);
  const [frozen, setFrozen] = useState(isPrint);
  useEffect(() => {
    const freeze = () => {
      ref.current = simulate(PRINT_T);
      setFrozen(true);
      setTick((n) => n + 1);
    };
    window.addEventListener('beforeprint', freeze);
    return () => window.removeEventListener('beforeprint', freeze);
  }, []);
  useEffect(() => {
    if (frozen) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      const s = ref.current;
      if (s.done) ref.current = init();
      else step(s, dt);
      setTick((n) => n + 1);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [frozen]);
  return ref.current;
}

const NX = 200;
const NY = 80;
const LEVELS: number[] = [];
for (let z = -0.45; z <= 0.46; z += 0.05) LEVELS.push(Math.round(z * 100) / 100);

/** Marching squares: one SVG path per contour level. */
function contours(): { z: number; d: string }[] {
  const grid: number[][] = [];
  for (let j = 0; j <= NY; j++) {
    const row: number[] = [];
    for (let i = 0; i <= NX; i++) row.push(f({ x: (i / NX) * MAP.w, y: (j / NY) * MAP.h }));
    grid.push(row);
  }
  const px = (i: number) => (i / NX) * W;
  const py = (j: number) => (j / NY) * H;
  const out: { z: number; d: string }[] = [];
  for (const z of LEVELS) {
    const parts: string[] = [];
    for (let j = 0; j < NY; j++) {
      for (let i = 0; i < NX; i++) {
        const a = grid[j]![i]!; // top-left
        const b = grid[j]![i + 1]!; // top-right
        const c = grid[j + 1]![i + 1]!; // bottom-right
        const d = grid[j + 1]![i]!; // bottom-left
        const idx = (a > z ? 8 : 0) | (b > z ? 4 : 0) | (c > z ? 2 : 0) | (d > z ? 1 : 0);
        if (idx === 0 || idx === 15) continue;
        const lerp = (v0: number, v1: number) => (z - v0) / (v1 - v0);
        const top: Vec = { x: px(i + lerp(a, b)), y: py(j) };
        const right: Vec = { x: px(i + 1), y: py(j + lerp(b, c)) };
        const bottom: Vec = { x: px(i + lerp(d, c)), y: py(j + 1) };
        const left: Vec = { x: px(i), y: py(j + lerp(a, d)) };
        const seg = (p: Vec, q: Vec) => parts.push(`M${p.x.toFixed(1)},${p.y.toFixed(1)}L${q.x.toFixed(1)},${q.y.toFixed(1)}`);
        switch (idx) {
          case 1: case 14: seg(left, bottom); break;
          case 2: case 13: seg(bottom, right); break;
          case 3: case 12: seg(left, right); break;
          case 4: case 11: seg(top, right); break;
          case 5: seg(top, left); seg(bottom, right); break;
          case 6: case 9: seg(top, bottom); break;
          case 7: case 8: seg(top, left); break;
          case 10: seg(top, right); seg(left, bottom); break;
        }
      }
    }
    out.push({ z, d: parts.join('') });
  }
  return out;
}

const mono: React.CSSProperties = { fontFamily: 'var(--font-mono)' };

function Label({ x, y, children, anchor = 'middle', color = 'var(--color-muted)', size = 15 }: {
  x: number; y: number; children: React.ReactNode; anchor?: 'start' | 'middle' | 'end'; color?: string; size?: number;
}) {
  return (
    <text x={x} y={y} textAnchor={anchor} fill={color} fontSize={size} style={mono}>
      {children}
    </text>
  );
}

export default function ContourMap() {
  const s = useSim();
  const lines = useMemo(contours, []);
  const sx = (x: number) => x * S;
  const sy = (y: number) => y * S;
  const bx = sx(s.p.x);
  const by = sy(s.p.y);
  const m = MAP.margin * S;
  const poking = fingerVisible(s);
  const deg = (Math.atan2(s.pokeDir.y, s.pokeDir.x) * 180) / Math.PI;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
      <defs>
        <pattern id="hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1={0} y1={0} x2={0} y2={10} stroke="var(--accent-secondary)" strokeWidth={3} opacity={0.5} />
        </pattern>
      </defs>
      <rect x={0} y={0} width={W} height={H} rx={10} fill="var(--surface)" opacity={0.9} />
      {/* contours: the deeper the level, the brighter and thicker the line, so dips read as dips and hills fade */}
      {lines.map(({ z, d }) => {
        const depth = Math.min(1, Math.max(0, (0.3 - z) / 0.7));
        return <path key={z} d={d} fill="none" stroke="var(--accent-tertiary)" strokeWidth={0.8 + 1.6 * depth} opacity={0.12 + 0.75 * depth} />;
      })}
      {/* the edge of the map */}
      <rect x={m} y={m} width={W - 2 * m} height={H - 2 * m} fill="none" stroke="var(--accent-secondary)" strokeWidth={2} strokeDasharray="6 6" opacity={0.7} />
      {/* constraints: areas the ball may not enter */}
      {ZONES.map((z) => (
        <g key={z.label}>
          <circle cx={sx(z.c.x)} cy={sy(z.c.y)} r={z.r * S} fill="url(#hatch)" stroke="var(--accent-secondary)" strokeWidth={2.5} strokeDasharray="6 6" />
          <Label x={sx(z.c.x)} y={sy(z.c.y) + z.r * S + 18} color="var(--accent-secondary)" size={14}>{z.label}</Label>
        </g>
      ))}
      {/* labels */}
      <Label x={sx(GOAL.x)} y={sy(GOAL.y) + 34} color="var(--accent-aqua)">good enough</Label>
      {DIPS.map((d) => (
        <Label key={`${d.x},${d.y}`} x={sx(d.x)} y={sy(d.y) + 34}>works, but...</Label>
      ))}
      <Label x={m + 10} y={m + 22} anchor="start" size={13}>the map is the code · contours: how bad it is · brighter is lower</Label>
      {/* trail and ball */}
      {s.trail.length > 1 ? (
        <polyline points={s.trail.map((p) => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ')} fill="none" stroke="var(--accent)" strokeWidth={2} opacity={0.45} />
      ) : null}
      <circle cx={bx} cy={by} r={R} fill="var(--accent)" />
      {poking ? (
        <text transform={`translate(${(bx - s.pokeDir.x * (R + 28)).toFixed(1)},${(by - s.pokeDir.y * (R + 28)).toFixed(1)}) rotate(${deg.toFixed(1)})`} textAnchor="middle" dominantBaseline="central" fontSize={36}>
          {'\u{1F449}'}
        </text>
      ) : null}
    </svg>
  );
}
