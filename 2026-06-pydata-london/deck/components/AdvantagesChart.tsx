type Row = {
  label: string;
  monty: string;
  sandbox: string;
};

const ROWS: Row[] = [
  { label: 'Startup latency', monty: '5µs', sandbox: '1s' },
  { label: 'Snapshot size', monty: '1KB', sandbox: '1GB' },
  { label: 'Cost per run', monty: '$0', sandbox: '$?' },
];

const wrap: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '0.9rem',
  marginTop: '1rem',
  alignItems: 'center',
};

const row: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1fr auto auto auto',
  alignItems: 'center',
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: '14px',
  padding: '0.8rem 1.6rem',
  background: 'var(--surface)',
  gap: '2rem',
  width: 'fit-content',
  minWidth: '36rem',
};

const labelStyle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: '0.95rem',
  letterSpacing: '0.10em',
  textTransform: 'uppercase',
  color: 'var(--color-muted)',
};

const valueCell: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  gap: '0.2rem',
  minWidth: '6rem',
};

const montyValue: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: '2.4rem',
  fontWeight: 700,
  color: 'var(--accent)',
  lineHeight: 1,
};

const sandboxValue: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: '1.9rem',
  color: 'var(--color-muted)',
  textDecoration: 'line-through',
  textDecorationColor: 'rgba(255,255,255,0.25)',
  lineHeight: 1,
};

const tag: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: '0.65rem',
  letterSpacing: '0.14em',
  textTransform: 'uppercase',
  color: 'var(--color-muted)',
};

const vs: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: '1rem',
  color: 'var(--color-muted)',
  fontStyle: 'italic',
  alignSelf: 'center',
};

export default function AdvantagesChart() {
  return (
    <div style={wrap}>
      {ROWS.map((r) => (
        <div key={r.label} style={row}>
          <div style={labelStyle}>{r.label}</div>
          <div style={valueCell}>
            <div style={montyValue}>{r.monty}</div>
            <span style={{ ...tag, color: 'var(--accent)' }}>Monty</span>
          </div>
          <span style={vs}>vs.</span>
          <div style={valueCell}>
            <div style={sandboxValue}>{r.sandbox}</div>
            <span style={tag}>full sandbox</span>
          </div>
        </div>
      ))}
    </div>
  );
}
