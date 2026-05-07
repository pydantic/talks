type Pill = {
  name: string;
  blurb: string;
};

const PILLS: Pill[] = [
  { name: 'Pydantic', blurb: 'data validation' },
  { name: 'Pydantic AI', blurb: 'agent framework' },
  { name: 'Logfire', blurb: 'observability' },
  { name: 'AI Gateway', blurb: 'routing & guardrails' },
];

const wrap: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(4, 1fr)',
  gap: '0.6rem',
  marginTop: '1rem',
};

const pill: React.CSSProperties = {
  border: '1px solid rgba(255,255,255,0.14)',
  borderRadius: '10px',
  padding: '0.7rem 0.8rem',
  background: 'var(--surface)',
  textAlign: 'center',
};

const name: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontWeight: 700,
  color: 'var(--color-heading)',
  fontSize: '0.95rem',
};

const blurb: React.CSSProperties = {
  fontSize: '0.7rem',
  color: 'var(--color-muted)',
  marginTop: '0.25rem',
  letterSpacing: '0.04em',
};

export default function PydanticStackPills() {
  return (
    <div style={wrap}>
      {PILLS.map((p) => (
        <div key={p.name} style={pill}>
          <div style={name}>{p.name}</div>
          <div style={blurb}>{p.blurb}</div>
        </div>
      ))}
    </div>
  );
}
