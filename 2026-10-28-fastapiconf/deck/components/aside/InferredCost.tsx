import Chart, { H, LABEL, PAD, W, linearScale, mono } from './Chart';
import { COST_GROWTH, COST_ONLY, DEV_COST_MULTIPLIER, EPOCH, FLAGSHIPS, LAB_COLOR, OTHERS, OTHER_ESTIMATES, US_GDP, costOnTrend, devCost, toYear } from './data';

const X0 = 2023;
const X1 = 2032;
/** log10 dollars. */
const Y0 = 7;
const Y1 = 13;
const TODAY = toYear('2026-10-28');

const REFERENCES = [0.001, 0.01, 0.1].map((f) => ({
  v: Math.log10(f * US_GDP),
  label: f === 1 ? 'US GDP' : `${f * 100}% of US GDP`,
}));

function money(log: number): string {
  const v = 10 ** log;
  if (v >= 1e12) return `$${(v / 1e12).toFixed(0)}T`;
  if (v >= 1e9) return `$${(v / 1e9).toFixed(0)}B`;
  return `$${(v / 1e6).toFixed(0)}M`;
}

export default function InferredCost() {
  const x = linearScale(X0, X1, PAD.left, W - PAD.right);
  const y = linearScale(Y0, Y1, H - PAD.bottom, PAD.top);
  const xTicks = Array.from({ length: X1 - X0 + 1 }, (_, i) => ({ v: X0 + i, label: String(X0 + i) }));
  const yTicks = Array.from({ length: Y1 - Y0 + 1 }, (_, i) => ({ v: Y0 + i, label: money(Y0 + i) }));
  const years = Array.from({ length: (X1 - X0) * 10 + 1 }, (_, i) => X0 + i * 0.1);
  const path = (from: number, to: number) =>
    years
      .filter((yr) => yr >= from - 1e-9 && yr <= to + 1e-9)
      .map((yr, i) => `${i === 0 ? 'M' : 'L'}${x(yr).toFixed(1)} ${y(Math.log10(costOnTrend(yr))).toFixed(1)}`)
      .join(' ');
  const epoch = [...FLAGSHIPS, ...OTHERS, ...COST_ONLY].filter((m) => m.cost);
  const lx = W - PAD.right - 540;
  const ly = H - PAD.bottom - 76;
  return (
    <Chart xTicks={xTicks} yTicks={yTicks} x={x} y={y} xLabel="Year" yLabel="Cost to develop a frontier model (log scale)">
      {REFERENCES.map((r) => (
        <g key={r.label}>
          <line x1={PAD.left} x2={W - PAD.right} y1={y(r.v)} y2={y(r.v)} stroke={LABEL} strokeDasharray="3 5" />
          <text x={PAD.left + 8} y={y(r.v) - 7} textAnchor="start" fill={LABEL} fontSize={14} style={mono}>
            {r.label}
          </text>
        </g>
      ))}
      <path d={path(X0, TODAY)} fill="none" stroke="var(--accent)" strokeWidth={2.5} />
      <path d={path(TODAY, X1)} fill="none" stroke="var(--accent)" strokeWidth={2.5} strokeDasharray="7 5" />
      <line x1={x(TODAY)} x2={x(TODAY)} y1={PAD.top} y2={H - PAD.bottom} stroke={LABEL} strokeDasharray="3 5" />
      <text x={x(TODAY) + 8} y={PAD.top + 14} fill={LABEL} fontSize={15} style={mono}>
        today
      </text>
      {OTHER_ESTIMATES.map((e, i) => (
        <rect
          key={i}
          x={x(toYear(e.date)) - 6}
          y={y(Math.log10(e.cost)) - 6}
          width={12}
          height={12}
          transform={`rotate(45 ${x(toYear(e.date))} ${y(Math.log10(e.cost))})`}
          fill="none"
          stroke="var(--color-text)"
          strokeWidth={1.5}
        />
      ))}
      {epoch.map((m) => (
        <circle key={m.name} cx={x(toYear(m.date))} cy={y(Math.log10(devCost(m)))} r={6.5} fill={LAB_COLOR[m.lab]} stroke="var(--bg-slide)" strokeWidth={1.5} />
      ))}
      <g transform={`translate(${lx} ${ly})`}>
        <circle cx={11} cy={0} r={6} fill={LAB_COLOR.OpenAI} />
        <text x={32} y={0} dominantBaseline="middle" fill="var(--color-text)" fontSize={16}>
          Epoch final-run compute cost x {DEV_COST_MULTIPLIER} for staff and experiments
        </text>
        <rect x={5} y={18} width={12} height={12} transform="rotate(45 11 24)" fill="none" stroke="var(--color-text)" strokeWidth={1.5} />
        <text x={32} y={24} dominantBaseline="middle" fill="var(--color-text)" fontSize={16}>
          Stanford AI Index, Altman, Amodei
        </text>
        <line x1={0} x2={22} y1={48} y2={48} stroke="var(--accent)" strokeWidth={2.5} strokeDasharray="7 5" />
        <text x={32} y={48} dominantBaseline="middle" fill="var(--color-text)" fontSize={16}>
          {EPOCH.computeGrowth}x compute / {EPOCH.flopPerDollarGrowth}x cheaper FLOP = {COST_GROWTH.toFixed(1)}x per year
        </text>
      </g>
    </Chart>
  );
}
