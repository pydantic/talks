/** A Python function whose body is a cloud: the developer owns the signature and the return, the agent owns the middle. */

const mono: React.CSSProperties = { fontFamily: 'var(--font-mono)' };

const KW = 'var(--accent)';
const TYPE = 'var(--accent-aqua)';
const NAME = 'var(--accent-secondary)';
const TEXT = 'var(--color-heading)';

/** Top-left quarter of the cloud as circles in a 600 x 260 box; mirrored about x = 300 and y = 130 so it is symmetric both ways. */
const QUARTER: [number, number, number][] = [
  [70, 130, 40],
  [120, 110, 58],
  [185, 85, 72],
  [260, 70, 84],
  [300, 65, 90],
  [300, 130, 100],
  [180, 130, 70],
];
const CX = 300;
const CY = 130;
const PUFFS: [number, number, number][] = [];
for (const [cx, cy, r] of QUARTER) {
  const xs = cx === CX ? [cx] : [cx, 2 * CX - cx];
  const ys = cy === CY ? [cy] : [cy, 2 * CY - cy];
  for (const x of xs) for (const y of ys) PUFFS.push([x, y, r]);
}

function Cloud() {
  return (
    <svg viewBox="0 0 600 260" width="34rem" style={{ display: 'block', overflow: 'visible' }}>
      <defs>
        <radialGradient id="puff" gradientUnits="userSpaceOnUse" cx={CX} cy={CY} r={300}>
          <stop offset="0%" stopColor="#ffffff" stopOpacity={0.95} />
          <stop offset="45%" stopColor="#efe6f8" stopOpacity={0.85} />
          <stop offset="100%" stopColor="#b388ff" stopOpacity={0.35} />
        </radialGradient>
        <filter id="fluff" x="-15%" y="-15%" width="130%" height="130%">
          <feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves={3} seed={7} result="noise" />
          <feDisplacementMap in="SourceGraphic" in2="noise" scale={18} xChannelSelector="R" yChannelSelector="G" />
          <feGaussianBlur stdDeviation="2.5" />
        </filter>
        <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="18" />
        </filter>
      </defs>
      <g filter="url(#glow)" opacity={0.45}>
        {PUFFS.map(([cx, cy, r], i) => (
          <circle key={i} cx={cx} cy={cy} r={r} fill="var(--accent-tertiary)" />
        ))}
      </g>
      <g filter="url(#fluff)">
        {PUFFS.map(([cx, cy, r], i) => (
          <circle key={i} cx={cx} cy={cy} r={r} fill="url(#puff)" />
        ))}
      </g>
    </svg>
  );
}

export default function FluffyFunction() {
  return (
    <div style={{ ...mono, fontSize: '1.7rem', lineHeight: 1.5, color: TEXT, display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, justifyContent: 'center', paddingLeft: '2rem' }}>
      <div>
        <span style={{ color: KW }}>def</span> <span style={{ color: NAME }}>implementation</span>(a: <span style={{ color: TYPE }}>int</span>, b: <span style={{ color: TYPE }}>str</span>, c: <span style={{ color: TYPE }}>float</span>) -&gt; <span style={{ color: TYPE }}>Foo</span>:
      </div>
      <div style={{ margin: '0.8rem 0 0.4rem 3.5rem' }}>
        <Cloud />
      </div>
      <div style={{ marginLeft: '3.5rem' }}>
        <span style={{ color: KW }}>return</span> foo
      </div>
    </div>
  );
}
