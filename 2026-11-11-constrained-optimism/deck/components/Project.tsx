interface Props {
  summary: React.ReactNode;
  optimize: React.ReactNode[];
  constraints: React.ReactNode[];
}

function Column({ title, color, items }: { title: string; color: string; items: React.ReactNode[] }) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ color, fontWeight: 700, fontSize: '1.2rem', borderBottom: `2px solid ${color}`, paddingBottom: '0.3rem', marginBottom: '0.7rem' }}>{title}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.55rem' }}>
        {items.map((it, i) => (
          <div key={i} style={{ fontSize: '1.15rem', lineHeight: 1.35, color: 'var(--color-heading)' }}>
            {it}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Same layout for every example: one-line summary, then what we optimize and what must hold. */
export default function Project({ summary, optimize, constraints }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.4rem', marginTop: '0.2rem' }}>
      <div style={{ fontSize: '1.25rem', lineHeight: 1.4 }}>{summary}</div>
      <div style={{ display: 'flex', gap: '2rem', alignItems: 'flex-start' }}>
        <Column title="Optimize" color="var(--accent-aqua)" items={optimize} />
        <Column title="Constraints" color="var(--accent-secondary)" items={constraints} />
      </div>
    </div>
  );
}
