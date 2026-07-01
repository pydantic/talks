// Coordinate system: 1000 x 540
// Two 100%-stacked bars — Usage and Spend — joined by Sankey ribbons.
// The same two task types swap dominance between the bars: the heavy (costly)
// flow balloons as you move from usage -> spend.
const Y_TOP = 50;
const Y_BOT = 460;
const H = Y_BOT - Y_TOP;

const BAR_W = 120;
const L0 = 30; // left (Usage) bar
const L1 = L0 + BAR_W;
const R1 = 970; // right (Spend) bar
const R0 = R1 - BAR_W;

const COLORS = {
  light: '#4ad7c5', // --accent-aqua
  heavy: '#e520a8', // --accent
};
const INK = '#15101a'; // --bg-slide, for text on solid fills

// Heavy tasks sit on top of both bars; the divider sweeps from 30% -> 70%.
const yL = Y_TOP + 0.3 * H; // Usage divider (heavy 30% top / light 70% bottom)
const yR = Y_TOP + 0.7 * H; // Spend divider (heavy 70% top / light 30% bottom)

const midL = L1;
const midR = R0;
const cx = (midL + midR) / 2;

// Ribbon between the two bars for a band bounded by topL..topR (upper edge)
// and botL..botR (lower edge).
const ribbon = (topL: number, topR: number, botL: number, botR: number) =>
  `M ${midL} ${topL} C ${cx} ${topL} ${cx} ${topR} ${midR} ${topR} ` +
  `L ${midR} ${botR} C ${cx} ${botR} ${cx} ${botL} ${midL} ${botL} Z`;

// Vertical centre of a band at the midpoint of the ribbon, for label placement.
const heavyMidY = (Y_TOP + (yL + yR) / 2) / 2;
const lightMidY = ((yL + yR) / 2 + Y_BOT) / 2;

export default function SandboxInvocations() {
  return (
    <svg
      viewBox="0 0 1000 520"
      preserveAspectRatio="xMidYMid meet"
      style={{
        fontFamily: 'var(--font-body)',
        display: 'block',
        width: '100%',
        height: 'auto',
      }}
    >
      {/* Ribbons — offset from the divider so a small gap mirrors the bar gap */}
      <path d={ribbon(Y_TOP, Y_TOP, yL - 3, yR - 3)} fill={`${COLORS.heavy}2e`} />
      <path d={ribbon(yL + 3, yR + 3, Y_BOT, Y_BOT)} fill={`${COLORS.light}2e`} />

      {/* Ribbon labels */}
      <text x={cx} y={heavyMidY + 12} textAnchor="middle" fontSize={38} fontWeight={700} fill={COLORS.heavy}>
        Heavy tasks
      </text>
      <text x={cx} y={lightMidY + 12} textAnchor="middle" fontSize={38} fontWeight={700} fill={COLORS.light}>
        Light tasks
      </text>

      {/* Usage bar (left) */}
      <Segment x={L0} y0={Y_TOP} y1={yL} color={COLORS.heavy} pct="30%" side="left" />
      <Segment x={L0} y0={yL} y1={Y_BOT} color={COLORS.light} pct="70%" side="left" />

      {/* Spend bar (right) */}
      <Segment x={R0} y0={Y_TOP} y1={yR} color={COLORS.heavy} pct="70%" side="right" />
      <Segment x={R0} y0={yR} y1={Y_BOT} color={COLORS.light} pct="30%" side="right" />

      {/* Bar titles */}
      <text x={(L0 + L1) / 2} y={Y_BOT + 50} textAnchor="middle" fontSize={30} fontWeight={700} fill="var(--color-text)">
        Usage
      </text>
      <text x={(R0 + R1) / 2} y={Y_BOT + 50} textAnchor="middle" fontSize={30} fontWeight={700} fill="var(--color-text)">
        Spend
      </text>
    </svg>
  );
}

// Rounded rect with independent corner radii (tl, tr, br, bl).
function roundedRectPath(x: number, y: number, w: number, h: number, tl: number, tr: number, br: number, bl: number) {
  return (
    `M ${x + tl} ${y} H ${x + w - tr} ` +
    `A ${tr} ${tr} 0 0 1 ${x + w} ${y + tr} V ${y + h - br} ` +
    `A ${br} ${br} 0 0 1 ${x + w - br} ${y + h} H ${x + bl} ` +
    `A ${bl} ${bl} 0 0 1 ${x} ${y + h - bl} V ${y + tl} ` +
    `A ${tl} ${tl} 0 0 1 ${x + tl} ${y} Z`
  );
}

function Segment({
  x,
  y0,
  y1,
  color,
  pct,
  side,
}: {
  x: number;
  y0: number;
  y1: number;
  color: string;
  pct: string;
  side: 'left' | 'right';
}) {
  const cy = (y0 + y1) / 2;
  const r = 12;
  // Round only the outer corners; the inner corners (facing the ribbon) stay square.
  const d =
    side === 'left'
      ? roundedRectPath(x, y0 + 3, BAR_W, y1 - y0 - 6, r, 0, 0, r)
      : roundedRectPath(x, y0 + 3, BAR_W, y1 - y0 - 6, 0, r, r, 0);
  return (
    <g>
      <path d={d} fill={color} />
      <text
        x={x + BAR_W / 2}
        y={cy + 14}
        textAnchor="middle"
        fontSize={42}
        fontWeight={800}
        fill={INK}
        fontFamily="var(--font-mono)"
      >
        {pct}
      </text>
    </g>
  );
}
