const STEPS: Array<{ n: string; title: string; sub: string }> = [
  { n: '1', title: 'Parse', sub: 'ruff → AST' },
  { n: '2', title: 'Resolve names', sub: 'strings → integer slot ids' },
  { n: '3', title: 'Compile', sub: 'AST → bytecode' },
  { n: '4', title: 'Run', sub: 'heap + stack VM' },
  { n: '5', title: 'Pause / Resume', sub: 'snapshot ↔ host' },
];

const wrap: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'stretch',
  gap: '0.35rem',
  fontFamily: 'var(--font-mono)',
  margin: '0 auto',
  maxWidth: '32rem',
};

const card: React.CSSProperties = {
  border: '1px solid rgba(255,255,255,0.14)',
  borderRadius: '10px',
  padding: '0.7rem 1rem',
  background: 'var(--surface)',
  display: 'grid',
  gridTemplateColumns: 'auto 1fr auto',
  alignItems: 'center',
  gap: '1rem',
};

const num: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: '1.4rem',
  fontWeight: 700,
  color: 'var(--accent)',
  width: '2rem',
  textAlign: 'center',
};

const title: React.CSSProperties = {
  fontWeight: 600,
  color: 'var(--color-heading)',
  fontSize: '1.05rem',
};

const sub: React.CSSProperties = {
  color: 'var(--color-muted)',
  fontSize: '0.8rem',
  textAlign: 'right',
};

const arrow: React.CSSProperties = {
  color: 'var(--accent)',
  fontFamily: 'var(--font-mono)',
  fontSize: '1rem',
  textAlign: 'center',
  userSelect: 'none',
  lineHeight: 1,
};

export default function HowOverview() {
  return (
    <div style={wrap}>
      {STEPS.map((s, i) => (
        <div key={s.n}>
          <div style={card}>
            <div style={num}>{s.n}</div>
            <div style={title}>{s.title}</div>
            <div style={sub}>{s.sub}</div>
          </div>
          {i < STEPS.length - 1 && <div style={arrow}>↓</div>}
        </div>
      ))}
    </div>
  );
}
