const wrap: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-start',
  gap: '0.4rem',
  width: 'fit-content',
  margin: '0.6rem auto 0',
  fontFamily: 'var(--font-mono)',
  fontSize: '1.5rem',
  lineHeight: 1.3,
  letterSpacing: '0.01em',
  background: 'var(--surface)',
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: '12px',
  padding: '0.9rem 1.6rem',
  whiteSpace: 'nowrap',
};

const cmd: React.CSSProperties = { color: 'var(--accent)', fontWeight: 700 };
const arg: React.CSSProperties = { color: 'var(--color-heading)' };
const note: React.CSSProperties = { color: 'var(--color-muted)', marginLeft: '1.2rem', fontSize: '0.8em' };
const prompt: React.CSSProperties = { color: 'var(--color-muted)', marginRight: '0.6rem' };

export interface Line {
  cmd: string;
  args: string;
  note?: string;
}

export default function InstallLine({ lines }: { lines: Line[] }) {
  return (
    <div style={wrap}>
      {lines.map((l) => (
        <div key={l.cmd + l.args}>
          <span style={prompt}>➤</span>
          <span style={cmd}>{l.cmd}</span> <span style={arg}>{l.args}</span>
          {l.note ? <span style={note}># {l.note}</span> : null}
        </div>
      ))}
    </div>
  );
}
