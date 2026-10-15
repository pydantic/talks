export interface Item {
  what: string;
  how: string;
}

interface Props {
  constraints: Item[];
  metrics: Item[];
  footer?: string;
  fontSize?: string;
}

const mono: React.CSSProperties = { fontFamily: 'var(--font-mono)' };

function Column({ title, sub, color, items }: { title: string; sub: string; color: string; items: Item[] }) {
  return (
    <div style={{ minWidth: 0, flex: 1 }}>
      <div style={{ borderBottom: `2px solid ${color}`, paddingBottom: '0.3rem', marginBottom: '0.5rem' }}>
        <span style={{ color, fontWeight: 700, fontSize: '1.05em' }}>{title}</span>
        <span style={{ ...mono, color: 'var(--color-muted)', fontSize: '0.8em', marginLeft: '0.7em' }}>{sub}</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.55rem' }}>
        {items.map((it) => (
          <div key={it.what} style={{ lineHeight: 1.35 }}>
            <span style={{ color: 'var(--color-heading)', fontWeight: 600 }}>{it.what}</span>
            <span style={{ color: 'var(--color-muted)' }}> {'·'} </span>
            <span>{it.how}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ConstraintsMetrics({ constraints, metrics, footer, fontSize = '0.95rem' }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize, marginTop: '0.2rem' }}>
      <div style={{ display: 'flex', gap: '1.6rem', alignItems: 'flex-start' }}>
        <Column title="Constraints" sub="how we enforce them" color="var(--accent-secondary)" items={constraints} />
        <Column title="Metrics" sub="how the model measures them" color="var(--accent-aqua)" items={metrics} />
      </div>
      {footer ? (
        <div style={{ ...mono, color: 'var(--color-muted)', fontSize: '0.9em', borderTop: '1px solid rgba(255,255,255,0.12)', paddingTop: '0.4rem' }}>
          {footer}
        </div>
      ) : null}
    </div>
  );
}
