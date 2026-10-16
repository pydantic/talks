interface Step {
  title: string;
  text: string;
  examples: string[];
  color: string;
}

const STEPS: Step[] = [
  {
    title: 'Pick one or two metrics',
    text: 'What gets better. If there seem to be ten, combine them into one composite score.',
    examples: ['p50 latency', 'CPython compat %', 'bytes allocated'],
    color: 'var(--accent-aqua)',
  },
  {
    title: 'Define hard constraints',
    text: 'What must never break. Deterministic where you can, AI review where you can’t.',
    examples: ['cargo test', 'pyright --strict', 'clippy -D warnings', 'review agent'],
    color: 'var(--accent-secondary)',
  },
  {
    title: 'Close the feedback loop',
    text: 'The agent reads the result itself: no human copying errors between windows.',
    examples: ['gh run view', 'compiler errors', 'CodSpeed MCP', 'Logfire MCP'],
    color: 'var(--accent-tertiary)',
  },
];

const mono: React.CSSProperties = { fontFamily: 'var(--font-mono)' };

export default function Challenge() {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.2rem', marginTop: '0.6rem' }}>
      {STEPS.map((s, i) => (
        <div
          key={s.title}
          style={{
            border: `1.5px solid ${s.color}`,
            borderRadius: 12,
            background: `color-mix(in srgb, ${s.color} 6%, transparent)`,
            padding: '1rem 1.1rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.7rem',
          }}
        >
          <div style={{ ...mono, color: s.color, fontSize: '0.85rem', fontWeight: 700 }}>{i + 1}</div>
          <div style={{ color: 'var(--color-heading)', fontWeight: 700, fontSize: '1.35rem', lineHeight: 1.2 }}>{s.title}</div>
          <div style={{ fontSize: '1rem', lineHeight: 1.4 }}>{s.text}</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: 'auto' }}>
            {s.examples.map((e) => (
              <span key={e} style={{ ...mono, fontSize: '0.75rem', padding: '2px 8px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.15)', color: 'var(--color-text)' }}>
                {e}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
