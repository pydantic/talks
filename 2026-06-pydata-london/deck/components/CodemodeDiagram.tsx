const TOOLS = [
  'get_weather',
  'convert_temp',
  'sql_query',
  'send_email',
  'read_file',
  'write_file',
  'load_skill',
  'create_chart',
];

const wrap: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'stretch',
  gap: '0.3rem',
  fontFamily: 'var(--font-mono)',
  fontSize: '0.9rem',
  margin: 0,
};

const colTitle: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: '0.65rem',
  letterSpacing: '0.14em',
  textTransform: 'uppercase',
  color: 'var(--color-muted)',
  textAlign: 'center',
  marginBottom: '0.35rem',
};

const card: React.CSSProperties = {
  border: '1px solid rgba(255,255,255,0.14)',
  borderRadius: '10px',
  padding: '0.6rem 0.7rem',
  background: 'var(--surface)',
  textAlign: 'center',
};

const accentCard: React.CSSProperties = {
  ...card,
  border: '1px solid var(--accent)',
  background: 'rgba(229, 32, 168, 0.10)',
  boxShadow: '0 0 0 3px rgba(229, 32, 168, 0.08)',
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: '0.5rem',
};

const userToolPill: React.CSSProperties = {
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: '6px',
  padding: '0.25rem 0.4rem',
  background: 'rgba(255,255,255,0.03)',
  color: 'var(--color-text)',
  fontSize: '0.75rem',
  textAlign: 'center',
};

const wrapperPill: React.CSSProperties = {
  ...userToolPill,
  border: '1px solid var(--accent)',
  background: 'rgba(229, 32, 168, 0.18)',
  color: 'var(--color-heading)',
  fontWeight: 600,
  fontSize: '0.85rem',
};

const arrow: React.CSSProperties = {
  color: 'var(--accent)',
  fontFamily: 'var(--font-mono)',
  fontSize: '1.1rem',
  textAlign: 'center',
  userSelect: 'none',
  lineHeight: 1,
};

const toolGrid: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: '0.3rem',
};

export default function CodemodeDiagram() {
  return (
    <div style={wrap}>
      <div>
        <div style={colTitle}>Agent</div>
        <div style={card}>
          <div style={{ fontWeight: 600, color: 'var(--color-heading)' }}>LLM</div>
          <div style={{ fontSize: '0.72rem', color: 'var(--color-muted)', marginTop: '0.15rem' }}>
            writes Python
          </div>
        </div>
      </div>

      <div style={arrow}>↓</div>

      <div>
        <div style={colTitle}>Codemode capability</div>
        <div style={accentCard}>
          <div style={wrapperPill}>run_code</div>
          <div style={wrapperPill}>find_tools</div>
        </div>
      </div>

      <div style={arrow}>↓</div>

      <div>
        <div style={colTitle}>Your tools</div>
        <div style={card}>
          <div style={toolGrid}>
            {TOOLS.map((t) => (
              <div key={t} style={userToolPill}>
                {t}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
