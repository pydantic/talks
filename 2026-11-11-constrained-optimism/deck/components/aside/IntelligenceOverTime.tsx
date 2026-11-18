import Chart, { H, LABEL, Legend, PAD, W, linearScale, mono } from './Chart';
import { ECI_VS_TIME, FLAGSHIPS, LAB_COLOR, toYear } from './data';

const X0 = 2023;
const X1 = 2030;
const Y0 = 110;
const Y1 = 230;
const TODAY = toYear('2026-10-28');

/** Named so the dots mean something from the back of the room; the rest stay unlabelled. */
const LABELLED: Record<string, [number, number]> = {
  'GPT-4': [12, 12],
  'Claude 3.5 Sonnet': [14, 10],
  o1: [-12, -10],
  'GPT-5': [14, 14],
  'Gemini 3 Pro': [-14, -12],
  'GPT-6 Astra': [-14, -14],
  'Claude Opus 5.5': [14, 12],
};

export default function IntelligenceOverTime() {
  const x = linearScale(X0, X1, PAD.left, W - PAD.right);
  const y = linearScale(Y0, Y1, H - PAD.bottom, PAD.top);
  const xTicks = Array.from({ length: X1 - X0 + 1 }, (_, i) => ({ v: X0 + i, label: String(X0 + i) }));
  const yTicks = Array.from({ length: 7 }, (_, i) => ({ v: Y0 + i * 20, label: String(Y0 + i * 20) }));
  const fit = ECI_VS_TIME;
  return (
    <Chart xTicks={xTicks} yTicks={yTicks} x={x} y={y} xLabel="Release date" yLabel="Epoch Capabilities Index">
      <line x1={x(X0)} y1={y(fit.at(X0))} x2={x(TODAY)} y2={y(fit.at(TODAY))} stroke="var(--accent)" strokeWidth={2.5} />
      <line x1={x(TODAY)} y1={y(fit.at(TODAY))} x2={x(X1)} y2={y(fit.at(X1))} stroke="var(--accent)" strokeWidth={2.5} strokeDasharray="7 5" />
      <line x1={x(TODAY)} x2={x(TODAY)} y1={PAD.top} y2={H - PAD.bottom} stroke={LABEL} strokeDasharray="3 5" />
      <text x={x(TODAY) + 8} y={PAD.top + 14} fill={LABEL} fontSize={15} style={mono}>
        today
      </text>
      {FLAGSHIPS.map((m) => {
        const cx = x(toYear(m.date));
        const cy = y(m.eci);
        const off = LABELLED[m.name];
        return (
          <g key={m.name}>
            <circle cx={cx} cy={cy} r={6.5} fill={LAB_COLOR[m.lab]} stroke="var(--bg-slide)" strokeWidth={1.5} />
            {off && (
              <text x={cx + off[0]} y={cy + off[1]} fill="var(--color-text)" fontSize={14} textAnchor={off[0] < 0 ? 'end' : 'start'} dominantBaseline="middle">
                {m.name}
              </text>
            )}
          </g>
        );
      })}
      <Legend
        x={PAD.left + 24}
        y={PAD.top + 24}
        items={[
          { color: LAB_COLOR.OpenAI, label: 'OpenAI' },
          { color: LAB_COLOR.Anthropic, label: 'Anthropic' },
          { color: LAB_COLOR.Google, label: 'Google' },
          { color: 'var(--accent)', label: `Linear fit, +${fit.slope.toFixed(0)} points per year`, dashed: true },
        ]}
      />
    </Chart>
  );
}
