const wrap: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  margin: '1.5rem 0 0.5rem',
  fontFamily: 'var(--font-mono)',
  fontSize: '1.6rem',
  lineHeight: 1.4,
  letterSpacing: '0.01em',
  background: 'var(--surface)',
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: '12px',
  padding: '1.1rem 1.4rem',
  whiteSpace: 'nowrap',
};

const cmd: React.CSSProperties = {
  color: 'var(--accent)',
  fontWeight: 700,
};

const arg: React.CSSProperties = {
  color: 'var(--color-heading)',
};

const sep: React.CSSProperties = {
  color: 'var(--color-muted)',
  margin: '0 1rem',
};

export default function InstallLine() {
  return (
    <div style={wrap}>
      <span style={cmd}>uv</span>
      <span>&nbsp;</span>
      <span style={arg}>add pydantic-monty</span>
      <span style={sep}>/</span>
      <span style={cmd}>npm</span>
      <span>&nbsp;</span>
      <span style={arg}>i @pydantic/monty</span>
    </div>
  );
}
