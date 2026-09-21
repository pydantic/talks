const W = 900;
const H = 230;

const mono: React.CSSProperties = { fontFamily: 'var(--font-mono)', fontSize: 15, fill: 'var(--color-muted)' };
const label: React.CSSProperties = { fontFamily: 'var(--font-body)', fontSize: 17, fill: 'var(--color-heading)' };

/** A real span bar with the pending twin emitted at its start. */
export default function PendingSpan() {
  const x0 = 140;
  const x1 = 760;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxHeight: '15rem', display: 'block', margin: '0.6rem auto 0' }}>
      {/* time axis */}
      <line x1={x0} y1={200} x2={x1 + 60} y2={200} stroke="rgba(255,255,255,0.25)" strokeWidth={1.5} />
      <text x={x0} y={222} style={mono}>t=0 · span starts</text>
      <text x={x1} y={222} style={mono} textAnchor="middle">span ends</text>

      {/* pending span: zero width dot at start, exported immediately */}
      <text x={20} y={62} style={label}>pending_span</text>
      <circle cx={x0} cy={56} r={9} fill="var(--accent)" />
      <line x1={x0} y1={70} x2={x0} y2={110} stroke="var(--accent)" strokeWidth={2} strokeDasharray="4 4" />
      <text x={x0 + 18} y={50} style={mono}>start_time == end_time</text>
      <text x={x0 + 18} y={70} style={mono}>logfire.span_type = "pending_span"</text>
      <text x={x0 + 18} y={90} style={mono}>exported now, as a child of the real span</text>

      {/* real span */}
      <text x={20} y={142} style={label}>span</text>
      <rect x={x0} y={122} width={x1 - x0} height={30} rx={6} fill="var(--accent-tertiary)" opacity={0.85} />
      <text x={x1 + 12} y={143} style={mono}>exported on end</text>

      {/* backend */}
      <text x={x0} y={178} style={mono} fill="var(--accent-aqua)">
        backend: show the pending row live, swap it for the real span when it arrives
      </text>
    </svg>
  );
}
