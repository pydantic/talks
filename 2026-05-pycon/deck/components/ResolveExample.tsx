type Row = {
  name: string;
  kind: string;
  slot: string;
  kindColor?: string;
};

const ROWS: Row[] = [
  { name: "'Hello'", kind: 'const', slot: 'consts[0]' },
  { name: 'get_who', kind: 'LocalUnassigned', slot: '— (host)', kindColor: 'var(--accent)' },
  { name: 'greeting', kind: 'local', slot: 'locals[0]' },
  { name: 'who', kind: 'local', slot: 'locals[1]' },
];

const wrap: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: '0.9rem',
  background: 'var(--surface)',
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: '10px',
  padding: '0.8rem 1rem',
};

const head: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1.1fr 1.3fr 1fr',
  gap: '0.8rem',
  fontSize: '0.65rem',
  letterSpacing: '0.14em',
  textTransform: 'uppercase',
  color: 'var(--color-muted)',
  paddingBottom: '0.45rem',
  borderBottom: '1px solid rgba(255,255,255,0.08)',
  marginBottom: '0.4rem',
};

const row: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1.1fr 1.3fr 1fr',
  gap: '0.8rem',
  padding: '0.35rem 0',
  alignItems: 'center',
};

export default function ResolveExample() {
  return (
    <div style={wrap}>
      <div style={head}>
        <div>name</div>
        <div>classification</div>
        <div>slot</div>
      </div>
      {ROWS.map((r) => (
        <div key={r.name} style={row}>
          <div style={{ color: 'var(--accent-aqua)' }}>{r.name}</div>
          <div style={{ color: r.kindColor ?? 'var(--color-text)' }}>{r.kind}</div>
          <div style={{ color: 'var(--color-muted)' }}>{r.slot}</div>
        </div>
      ))}
    </div>
  );
}
