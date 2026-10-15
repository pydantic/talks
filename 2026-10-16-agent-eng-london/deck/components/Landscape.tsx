import { useEffect, useRef, useState } from 'react';
import { BOWL, SLOPE_PERIOD, f, fingerVisible, init, simulate, step, type Mode, type Sim } from './landscapeSim';

const W = 1000;
const H = 400;
const R = 11;

// Headless Chrome (the PDF build) does not fire `beforeprint`, so detect it up front and render a fixed frame.
function isPrint() {
  if (typeof window === 'undefined') return false;
  if (window.matchMedia?.('print').matches) return true;
  return /HeadlessChrome/.test(navigator.userAgent);
}

// Representative frozen frames: bowl mid-poke in the first dip, slope stuck behind a bump.
const PRINT_T: Record<Mode, number> = { bowl: 3.55, slope: 3.6 };

function useSim(mode: Mode): Sim {
  const ref = useRef<Sim>(isPrint() ? simulate(mode, PRINT_T[mode]) : init(mode));
  const [, setTick] = useState(0);
  const [frozen, setFrozen] = useState(isPrint);
  useEffect(() => {
    const freeze = () => {
      ref.current = simulate(mode, PRINT_T[mode]);
      setFrozen(true);
      setTick((n) => n + 1);
    };
    window.addEventListener('beforeprint', freeze);
    return () => window.removeEventListener('beforeprint', freeze);
  }, [mode]);
  useEffect(() => {
    if (frozen) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      const s = ref.current;
      if (s.done) {
        ref.current = init(mode);
      } else {
        step(s, dt);
      }
      setTick((n) => n + 1);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [frozen, mode]);
  return ref.current;
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

function Finger({ x, y, dir }: { x: number; y: number; dir: number }) {
  // The finger sits on the side it pokes from and points toward the ball.
  const fx = x - dir * (R + 30);
  return (
    <text x={fx} y={y + 12} textAnchor="middle" fontSize={36}>
      {dir < 0 ? '\u{1F448}' : '\u{1F449}'}
    </text>
  );
}

function Bowl({ s }: { s: Sim }) {
  const bottom = H - 36;
  const scale = 330;
  const sx = (x: number) => x * W;
  const sy = (x: number) => bottom - f('bowl', x) * scale;
  const pts: string[] = [];
  for (let x = 0; x <= 1.0001; x += 0.004) pts.push(`${sx(x).toFixed(1)},${sy(x).toFixed(1)}`);
  const curve = pts.join(' ');
  const area = `M0,${H} L${curve.replace(/ /g, ' L')} L${W},${H} Z`;
  const bx = sx(s.x);
  const by = sy(s.x) - R;
  const poking = fingerVisible(s);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
      <path d={area} fill="var(--surface)" opacity={0.9} />
      <polyline points={curve} fill="none" stroke="var(--accent-tertiary)" strokeWidth={3} />
      {/* constraints: the walls */}
      <rect x={0} y={0} width={sx(BOWL.lo)} height={H} fill="var(--accent-secondary)" opacity={0.18} />
      <rect x={sx(BOWL.hi)} y={0} width={W - sx(BOWL.hi)} height={H} fill="var(--accent-secondary)" opacity={0.18} />
      <line x1={sx(BOWL.lo)} y1={0} x2={sx(BOWL.lo)} y2={H} stroke="var(--accent-secondary)" strokeWidth={2} strokeDasharray="6 6" />
      <line x1={sx(BOWL.hi)} y1={0} x2={sx(BOWL.hi)} y2={H} stroke="var(--accent-secondary)" strokeWidth={2} strokeDasharray="6 6" />
      <Label x={sx(BOWL.lo) / 2} y={H / 2} color="var(--accent-secondary)" size={14}>constraint</Label>
      <Label x={(sx(BOWL.hi) + W) / 2} y={H / 2} color="var(--accent-secondary)" size={14}>constraint</Label>
      {/* labels */}
      <Label x={sx(0.5)} y={sy(0.5) + 26} color="var(--accent-aqua)">good enough</Label>
      <Label x={sx(0.26)} y={sy(0.26) + 26}>works, but...</Label>
      <Label x={sx(0.76)} y={sy(0.76) + 26}>works, but...</Label>
      <Label x={W - 20} y={H - 10} anchor="end">the code {'→'}</Label>
      <Label x={sx(BOWL.lo) + 12} y={24} anchor="start">{'↑'} how bad it is</Label>
      {/* ball */}
      <circle cx={bx} cy={by} r={R} fill="var(--accent)" />
      {poking ? <Finger x={bx} y={by} dir={s.pokeDir} /> : null}
    </svg>
  );
}

const FLAGS = ['tests', 'lint', 'types', 'review', 'security', 'fuzzing'];

function Slope({ s }: { s: Sim }) {
  const window = 1.25;
  const left = s.x - 0.28;
  const scale = 300;
  const baseY = H * 0.5;
  const sx = (x: number) => ((x - left) / window) * W;
  const sy = (x: number) => baseY - (f('slope', x) - f('slope', s.x)) * scale;
  const pts: string[] = [];
  for (let x = left - 0.02; x <= left + window + 0.02; x += 0.004) pts.push(`${sx(x).toFixed(1)},${sy(x).toFixed(1)}`);
  const curve = pts.join(' ');
  const area = `M${sx(left - 0.02).toFixed(1)},${H} L${curve.replace(/ /g, ' L')} L${sx(left + window + 0.02).toFixed(1)},${H} Z`;
  const bx = sx(s.x);
  const by = sy(s.x) - R;
  const poking = fingerVisible(s);
  // flags between the bumps
  const k0 = Math.floor(left / SLOPE_PERIOD) - 1;
  const flags: { x: number; label: string }[] = [];
  for (let k = k0; k <= k0 + 3; k++) {
    const x = k * SLOPE_PERIOD + 0.15;
    if (x > left - 0.05 && x < left + window + 0.05) flags.push({ x, label: FLAGS[((k % FLAGS.length) + FLAGS.length) % FLAGS.length] ?? 'tests' });
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
      <path d={area} fill="var(--surface)" opacity={0.9} />
      <polyline points={curve} fill="none" stroke="var(--accent-tertiary)" strokeWidth={3} />
      {flags.map((fl) => {
        const x = sx(fl.x);
        const y = sy(fl.x);
        return (
          <g key={fl.x}>
            <line x1={x} y1={y} x2={x} y2={y - 46} stroke="var(--accent-secondary)" strokeWidth={2} />
            <polygon points={`${x},${y - 46} ${x + 26},${y - 38} ${x},${y - 30}`} fill="var(--accent-secondary)" />
            <Label x={x + 4} y={y - 52} anchor="start" color="var(--accent-secondary)" size={13}>{fl.label}</Label>
          </g>
        );
      })}
      <Label x={W - 20} y={H - 12} anchor="end">there is no bottom {'→'}</Label>
      <Label x={20} y={24} anchor="start">{'↑'} how bad it is</Label>
      <circle cx={bx} cy={by} r={R} fill="var(--accent)" />
      {poking ? <Finger x={bx} y={by} dir={s.pokeDir} /> : null}
    </svg>
  );
}

export default function Landscape({ mode }: { mode: Mode }) {
  const s = useSim(mode);
  return mode === 'bowl' ? <Bowl s={s} /> : <Slope s={s} />;
}
