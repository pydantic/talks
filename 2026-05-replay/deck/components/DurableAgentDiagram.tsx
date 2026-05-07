const ACTIVITIES = [
  { name: 'model request', sub: 'OpenAI · Anthropic · …' },
  { name: 'tool call', sub: 'your @agent.tool fns' },
  { name: 'MCP call', sub: 'remote tools' },
];

const wrap: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'stretch',
  gap: '0.4rem',
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

const workflowCard: React.CSSProperties = {
  border: '1px solid rgba(255,255,255,0.18)',
  borderRadius: '12px',
  padding: '0.9rem 1rem',
  background: 'var(--surface)',
  textAlign: 'center',
  position: 'relative',
};

const agentLoopCard: React.CSSProperties = {
  border: '1px dashed rgba(255,255,255,0.22)',
  borderRadius: '8px',
  padding: '0.55rem 0.7rem',
  marginTop: '0.6rem',
  background: 'rgba(255,255,255,0.03)',
  color: 'var(--color-heading)',
  fontWeight: 600,
};

const arrow: React.CSSProperties = {
  color: 'var(--accent)',
  fontFamily: 'var(--font-mono)',
  fontSize: '1.1rem',
  textAlign: 'center',
  userSelect: 'none',
  lineHeight: 1,
};

const activitiesGrid: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '1fr 1fr 1fr',
  gap: '0.5rem',
};

const activityCard: React.CSSProperties = {
  border: '1px solid var(--accent)',
  background: 'rgba(229, 32, 168, 0.14)',
  boxShadow: '0 0 0 3px rgba(229, 32, 168, 0.06)',
  borderRadius: '8px',
  padding: '0.55rem 0.5rem',
  textAlign: 'center',
  color: 'var(--color-heading)',
};

const activityName: React.CSSProperties = {
  fontWeight: 600,
  fontSize: '0.9rem',
};

const activitySub: React.CSSProperties = {
  fontSize: '0.7rem',
  color: 'var(--color-muted)',
  marginTop: '0.2rem',
};

const tag: React.CSSProperties = {
  position: 'absolute',
  top: '-0.55rem',
  left: '0.8rem',
  fontFamily: 'var(--font-mono)',
  fontSize: '0.6rem',
  letterSpacing: '0.14em',
  textTransform: 'uppercase',
  color: 'var(--color-muted)',
  background: 'var(--bg-slide)',
  padding: '0 0.4rem',
};

const footer: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: '0.7rem',
  color: 'var(--color-muted)',
  textAlign: 'center',
  marginTop: '0.3rem',
  letterSpacing: '0.06em',
};

export default function DurableAgentDiagram() {
  return (
    <div style={wrap}>
      <div style={workflowCard}>
        <span style={tag}>Temporal Workflow</span>
        <div style={{ fontSize: '0.72rem', color: 'var(--color-muted)' }}>
          deterministic · replayable
        </div>
        <div style={agentLoopCard}>Pydantic AI agent loop</div>
      </div>

      <div style={arrow}>↓</div>

      <div>
        <div style={colTitle}>Activities — every external call</div>
        <div style={activitiesGrid}>
          {ACTIVITIES.map((a) => (
            <div key={a.name} style={activityCard}>
              <div style={activityName}>{a.name}</div>
              <div style={activitySub}>{a.sub}</div>
            </div>
          ))}
        </div>
        <div style={footer}>durable · retryable · checkpointed</div>
      </div>
    </div>
  );
}
