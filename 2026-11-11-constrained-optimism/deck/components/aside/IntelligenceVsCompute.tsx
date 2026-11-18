import Chart, { H, Legend, PAD, W, linearScale } from './Chart';
import { ECI_VS_LOG_COMPUTE, FLAGSHIPS, LAB_COLOR, OTHERS } from './data';

const X0 = 24;
const X1 = 27.5;
const Y0 = 110;
const Y1 = 180;

const LABELLED: Record<string, [number, number]> = {
  'GPT-4': [-12, 0],
  'Claude 2': [12, 4],
  'Claude 3.5 Sonnet': [12, -8],
  'GPT-4.5': [-12, 14],
  'GPT-5': [-12, -4],
  'GPT-6 Astra': [-12, -12],
  'DeepSeek-R1': [12, 0],
  'Grok 4': [12, 10],
};

function sup(exp: number) {
  return (
    <>
      10<tspan dy={-7} fontSize={12}>{exp}</tspan>
    </>
  );
}

export default function IntelligenceVsCompute() {
  const x = linearScale(X0, X1, PAD.left, W - PAD.right);
  const y = linearScale(Y0, Y1, H - PAD.bottom, PAD.top);
  const xTicks = [24, 25, 26, 27].map((e) => ({ v: e, label: `1e${e}` }));
  const yTicks = Array.from({ length: 8 }, (_, i) => ({ v: Y0 + i * 10, label: String(Y0 + i * 10) }));
  const fit = ECI_VS_LOG_COMPUTE;
  const points = [...OTHERS, ...FLAGSHIPS].filter((m) => m.compute);
  return (
    <Chart xTicks={xTicks} yTicks={yTicks} x={x} y={y} xLabel="Training compute, FLOP (log scale)" yLabel="Epoch Capabilities Index">
      <line x1={x(X0)} y1={y(fit.at(X0))} x2={x(X1)} y2={y(fit.at(X1))} stroke="var(--accent)" strokeWidth={2.5} strokeDasharray="7 5" />
      {points.map((m) => {
        const cx = x(Math.log10(m.compute!));
        const cy = y(m.eci);
        const off = LABELLED[m.name];
        return (
          <g key={m.name}>
            <circle cx={cx} cy={cy} r={7} fill={LAB_COLOR[m.lab]} stroke="var(--bg-slide)" strokeWidth={1.5} />
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
          { color: LAB_COLOR.Other, label: 'xAI, DeepSeek, Meta' },
          { color: 'var(--accent)', label: `Fit, +${fit.slope.toFixed(0)} points per 10x compute`, dashed: true },
        ]}
      />
    </Chart>
  );
}
