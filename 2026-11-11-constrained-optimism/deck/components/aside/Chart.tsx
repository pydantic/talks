import type { ReactNode } from 'react';

export const W = 1000;
export const H = 470;
export const PAD = { left: 78, right: 36, top: 22, bottom: 56 };

export interface Scale {
  (v: number): number;
}

export function linearScale(d0: number, d1: number, r0: number, r1: number): Scale {
  return (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
}

export const mono: React.CSSProperties = { fontFamily: 'var(--font-mono)' };
export const AXIS = 'rgba(245, 238, 248, 0.25)';
export const GRID = 'rgba(245, 238, 248, 0.07)';
export const LABEL = 'var(--color-muted)';

export interface Tick {
  v: number;
  label: string;
}

/** Shared SVG frame: axes, gridlines, tick labels and axis titles. Children draw in plot coordinates. */
export default function Chart({
  xTicks,
  yTicks,
  x,
  y,
  xLabel,
  yLabel,
  children,
}: {
  xTicks: Tick[];
  yTicks: Tick[];
  x: Scale;
  y: Scale;
  xLabel: string;
  yLabel: string;
  children: ReactNode;
}) {
  const x0 = PAD.left;
  const x1 = W - PAD.right;
  const y0 = H - PAD.bottom;
  const y1 = PAD.top;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', flex: 1, minHeight: 0, maxHeight: '100%' }}>
      {yTicks.map((t) => (
        <g key={`y${t.v}`}>
          <line x1={x0} x2={x1} y1={y(t.v)} y2={y(t.v)} stroke={GRID} />
          <text x={x0 - 10} y={y(t.v)} textAnchor="end" dominantBaseline="middle" fill={LABEL} fontSize={16} style={mono}>
            {t.label}
          </text>
        </g>
      ))}
      {xTicks.map((t) => (
        <g key={`x${t.v}`}>
          <line x1={x(t.v)} x2={x(t.v)} y1={y0} y2={y1} stroke={GRID} />
          <text x={x(t.v)} y={y0 + 22} textAnchor="middle" fill={LABEL} fontSize={16} style={mono}>
            {t.label}
          </text>
        </g>
      ))}
      <line x1={x0} x2={x1} y1={y0} y2={y0} stroke={AXIS} />
      <line x1={x0} x2={x0} y1={y0} y2={y1} stroke={AXIS} />
      <text x={(x0 + x1) / 2} y={H - 8} textAnchor="middle" fill={LABEL} fontSize={16}>
        {xLabel}
      </text>
      <text transform={`translate(18 ${(y0 + y1) / 2}) rotate(-90)`} textAnchor="middle" fill={LABEL} fontSize={16}>
        {yLabel}
      </text>
      {children}
    </svg>
  );
}

export function Legend({ items, x: px, y: py }: { items: { color: string; label: string; dashed?: boolean }[]; x: number; y: number }) {
  return (
    <g>
      {items.map((it, i) => (
        <g key={it.label} transform={`translate(${px} ${py + i * 24})`}>
          {it.dashed ? (
            <line x1={0} x2={22} y1={0} y2={0} stroke={it.color} strokeWidth={2.5} strokeDasharray="7 5" />
          ) : (
            <circle cx={11} cy={0} r={6} fill={it.color} />
          )}
          <text x={32} y={0} dominantBaseline="middle" fill="var(--color-text)" fontSize={16}>
            {it.label}
          </text>
        </g>
      ))}
    </g>
  );
}
