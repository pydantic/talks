// Coordinate system: 1000 x 560
const X = {
  toolCalling: 150,
  monty: 360,
  sandbox: 560,
  coding: 760,
  full: 940,
};

// Plot vertical bounds (legend sits above; x-labels sit below)
const Y_HIGH = 90;
const Y_LOW = 480;

// X-axis horizontal extent for axes/grid
const X_LEFT = 100;
const X_RIGHT = 980;

const COLORS = {
  control: '#4dabe8',
  capabilities: '#e07a3a',
  llm: '#cc79a7',
  price: '#e6b300',
  setup: '#7fc8f0',
};

const labelSize = 18;
const legendSize = 16;

export default function SandboxContinuum() {
  return (
    <svg
      viewBox="0 0 1000 560"
      preserveAspectRatio="xMidYMid meet"
      style={{
        fontFamily: 'var(--font-body)',
        display: 'block',
        width: '100%',
        height: 'auto',
      }}
    >
      {/* Axes */}
      <line x1={X_LEFT} y1={Y_HIGH} x2={X_LEFT} y2={Y_LOW} stroke="rgba(255,255,255,0.18)" strokeWidth="1" />
      <line x1={X_LEFT} y1={Y_LOW} x2={X_RIGHT} y2={Y_LOW} stroke="rgba(255,255,255,0.18)" strokeWidth="1" />

      {/* Grid lines */}
      {[0, 0.25, 0.5, 0.75].map((p) => (
        <line
          key={p}
          x1={X_LEFT}
          y1={Y_HIGH + p * (Y_LOW - Y_HIGH)}
          x2={X_RIGHT}
          y2={Y_HIGH + p * (Y_LOW - Y_HIGH)}
          stroke="rgba(255,255,255,0.05)"
          strokeWidth="0.5"
        />
      ))}

      {/* Y-axis labels */}
      <text x={X_LEFT - 10} y={Y_HIGH + 5} textAnchor="end" fontSize={labelSize} fill="var(--color-muted)">
        High
      </text>
      <text x={X_LEFT - 10} y={Y_LOW + 5} textAnchor="end" fontSize={labelSize} fill="var(--color-muted)">
        Low
      </text>

      {/* Vertical guide dashes at each solution */}
      {Object.values(X).map((x) => (
        <line
          key={x}
          x1={x}
          y1={Y_HIGH}
          x2={x}
          y2={Y_LOW}
          stroke="rgba(255,255,255,0.08)"
          strokeWidth="0.5"
          strokeDasharray="4,4"
        />
      ))}

      {/* Tick marks */}
      {Object.values(X).map((x) => (
        <line
          key={`tick-${x}`}
          x1={x}
          y1={Y_LOW}
          x2={x}
          y2={Y_LOW + 8}
          stroke="rgba(255,255,255,0.3)"
          strokeWidth="1.5"
        />
      ))}

      {/* X-axis solution labels */}
      <SolutionLabel x={X.toolCalling} lines={['Tool', 'Calling']} />
      <SolutionLabel x={X.monty} lines={['Monty']} highlight />
      <SolutionLabel x={X.sandbox} lines={['Sandbox', 'Services']} />
      <SolutionLabel x={X.coding} lines={['Coding', 'Agents']} />
      <SolutionLabel x={X.full} lines={['Full', 'Computer Use']} />

      {/* Lines */}
      {/* Control: high left -> low right */}
      <Line x1={X.toolCalling} y1={110} x2={X.full} y2={460} color={COLORS.control} />
      {/* Capabilities: low left -> high right */}
      <Line x1={X.toolCalling} y1={420} x2={X.full} y2={100} color={COLORS.capabilities} />
      {/* LLM Complexity: low left -> high right (long dash) */}
      <Line x1={X.toolCalling} y1={440} x2={X.full} y2={125} color={COLORS.llm} dash="14,5" thin />
      {/* Setup: low left -> high right (dotted) */}
      <Line x1={X.toolCalling} y1={465} x2={X.full} y2={150} color={COLORS.setup} dash="3,6" thin />

      {/* Price: zero except spike at Sandbox Services */}
      <polyline
        points={`${X.toolCalling},465 ${X.monty},465 ${X.sandbox},115 ${X.coding},465 ${X.full},465`}
        fill="none"
        stroke={COLORS.price}
        strokeWidth="2.5"
        strokeDasharray="8,4"
      />
      <Dot x={X.toolCalling} y={465} color={COLORS.price} />
      <Dot x={X.monty} y={465} color={COLORS.price} />
      <Dot x={X.sandbox} y={115} color={COLORS.price} />
      <Dot x={X.coding} y={465} color={COLORS.price} />
      <Dot x={X.full} y={465} color={COLORS.price} />

      {/* End-point dots for straight lines */}
      <Dot x={X.toolCalling} y={110} color={COLORS.control} />
      <Dot x={X.full} y={460} color={COLORS.control} />
      <Dot x={X.toolCalling} y={420} color={COLORS.capabilities} />
      <Dot x={X.full} y={100} color={COLORS.capabilities} />
      <Dot x={X.toolCalling} y={440} color={COLORS.llm} />
      <Dot x={X.full} y={125} color={COLORS.llm} />
      <Dot x={X.toolCalling} y={465} color={COLORS.setup} />
      <Dot x={X.full} y={150} color={COLORS.setup} />

      <Legend />
    </svg>
  );
}

function SolutionLabel({ x, lines, highlight }: { x: number; lines: string[]; highlight?: boolean }) {
  return (
    <g>
      {lines.map((line, i) => (
        <text
          key={i}
          x={x}
          y={Y_LOW + 30 + i * 22}
          textAnchor="middle"
          fontSize={labelSize}
          fontWeight={highlight ? 700 : 600}
          fill={highlight ? 'var(--accent)' : 'var(--color-text)'}
        >
          {line}
        </text>
      ))}
    </g>
  );
}

function Line({
  x1, y1, x2, y2, color, dash, thin,
}: {
  x1: number; y1: number; x2: number; y2: number; color: string; dash?: string; thin?: boolean;
}) {
  return (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke={color}
      strokeWidth={thin ? 3 : 3.5}
      strokeDasharray={dash}
      strokeLinecap="round"
    />
  );
}

function Dot({ x, y, color }: { x: number; y: number; color: string }) {
  return <circle cx={x} cy={y} r="6" fill={color} />;
}

function Legend() {
  const items: { color: string; label: string; dash?: string }[] = [
    { color: COLORS.control, label: 'Control' },
    { color: COLORS.capabilities, label: 'Capabilities' },
    { color: COLORS.llm, label: 'LLM Complexity', dash: '14,5' },
    { color: COLORS.price, label: 'Price', dash: '8,4' },
    { color: COLORS.setup, label: 'Setup', dash: '3,6' },
  ];
  const y = 40;
  const colWidth = 180;
  const totalWidth = items.length * colWidth;
  const startX = (1000 - totalWidth) / 2 + 10;
  return (
    <g>
      {items.map((it, i) => {
        const cx = startX + i * colWidth;
        return (
          <g key={it.label}>
            <line
              x1={cx}
              y1={y}
              x2={cx + 36}
              y2={y}
              stroke={it.color}
              strokeWidth="3.5"
              strokeDasharray={it.dash}
              strokeLinecap="round"
            />
            <text x={cx + 44} y={y + 6} fontSize={legendSize} fill="var(--color-text)">
              {it.label}
            </text>
          </g>
        );
      })}
    </g>
  );
}
