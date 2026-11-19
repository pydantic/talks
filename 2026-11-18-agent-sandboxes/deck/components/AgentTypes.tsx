interface AgentType {
  title: string;
  text: string;
  tools: string[];
  more?: boolean;
  color: string;
}

const TYPES: AgentType[] = [
  {
    title: 'Developer agents',
    text: 'Like a human developer, they need a whole machine.',
    tools: ['bash', 'git', 'npm', 'pip', 'gcc', 'curl', 'ssh', 'make', 'apt-get', 'grep', 'sed', 'awk', 'tar', 'cron', 'vim', 'wget'],
    more: true,
    color: 'var(--accent-secondary)',
  },
  {
    title: 'Knowledge work agents',
    text: 'Like a human knowledge worker, they need a handful of tools.',
    tools: ['sql_query', 'search_docs', 'create_chart', 'send_email'],
    color: 'var(--accent-aqua)',
  },
];

const mono: React.CSSProperties = { fontFamily: 'var(--font-mono)' };

export default function AgentTypes() {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.4rem', marginTop: '0.6rem' }}>
      {TYPES.map((t) => (
        <div
          key={t.title}
          style={{
            border: `1.5px solid ${t.color}`,
            borderRadius: 12,
            background: `color-mix(in srgb, ${t.color} 6%, transparent)`,
            padding: '1.1rem 1.2rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.7rem',
          }}
        >
          <div style={{ color: t.color, fontWeight: 700, fontSize: '1.6rem', lineHeight: 1.2 }}>{t.title}</div>
          <div style={{ fontSize: '1.1rem', lineHeight: 1.4 }}>{t.text}</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem', marginTop: '0.3rem' }}>
            {t.tools.map((tool) => (
              <span
                key={tool}
                style={{ ...mono, fontSize: '0.85rem', padding: '3px 10px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.18)', color: 'var(--color-text)' }}
              >
                {tool}
              </span>
            ))}
            {t.more && <span style={{ ...mono, fontSize: '0.85rem', padding: '3px 4px', color: 'var(--color-muted)' }}>…</span>}
          </div>
        </div>
      ))}
    </div>
  );
}
