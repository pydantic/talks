const wrap: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-start',
  gap: '0.4rem',
  width: 'fit-content',
  margin: '0.6rem auto 0',
  fontFamily: 'var(--font-mono)',
  fontSize: '1.6rem',
  lineHeight: 1.3,
  letterSpacing: '0.01em',
  background: 'var(--surface)',
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: '12px',
  padding: '0.9rem 1.6rem',
  whiteSpace: 'nowrap',
};

const cmd: React.CSSProperties = {
  color: 'var(--accent)',
  fontWeight: 700,
};

const arg: React.CSSProperties = {
  color: 'var(--color-heading)',
};

const prompt: React.CSSProperties = {
  color: 'var(--color-muted)',
  marginRight: '0.6rem',
};

export default function InstallLine() {
  return (
    <div style={wrap}>
      <div>
        <span style={prompt}>➤</span>
        <span style={cmd}>uv</span> <span style={arg}>add pydantic-monty</span>
      </div>
      <div>
        <span style={prompt}>➤</span>
        <span style={cmd}>npm</span> <span style={arg}>i @pydantic/monty</span>
      </div>
      <div>
        <span style={prompt}>➤</span>
        <span style={cmd}>cargo</span> <span style={arg}>add monty</span>
      </div>
    </div>
  );
}
