/**
 * Two-panel product map for the "Pydantic today" slide: open source tools on the
 * left, commercial Pydantic Logfire services on the right.
 */

interface Product {
  name: string
  /** Plain-language description, for a non-technical audience. */
  detail: string
  /** Optional tag shown after the name, e.g. "soon". */
  tag?: string
}

const OPEN_SOURCE: Product[] = [
  { name: 'Pydantic Validation', detail: 'Contracts for data, 1 billion downloads a month' },
  { name: 'Pydantic AI', detail: 'Production grade agent framework' },
  { name: 'Pydantic Monty', detail: 'Python sandbox for code written by agents, starts in 1ms' },
]

const LOGFIRE: Product[] = [
  { name: 'AI Observability', detail: 'Traces, evals and prompt management for agents' },
  { name: 'Infra Observability', detail: 'Logs, metrics and traces for the whole application' },
  { name: 'AI Gateway', detail: 'One endpoint, every model' },
  { name: 'Agent Sandboxes', detail: 'Hosted code execution, built on Monty', tag: 'soon' },
]

/** One bordered panel with a heading, a subheading and a list of products. */
function Panel({
  title,
  subtitle,
  color,
  products,
}: {
  title: string
  subtitle: string
  color: string
  products: Product[]
}) {
  return (
    <div
      style={{
        flex: 1,
        minWidth: 0,
        border: `1.5px solid ${color}`,
        borderRadius: 12,
        background: `color-mix(in srgb, ${color} 6%, transparent)`,
        /* Clip the header band to the panel's rounded corners. */
        overflow: 'hidden',
      }}
    >
      {/* Header band: filled with the panel color so the title stands apart from the product list */}
      <div style={{ background: color, padding: '0.6rem 1.3rem 0.65rem' }}>
        <div
          style={{
            fontFamily: 'var(--font-heading)',
            fontWeight: 700,
            fontSize: '1.7rem',
            lineHeight: 1.15,
            color: 'var(--bg-slide)',
          }}
        >
          {title}
        </div>
        <div
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.72rem',
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            color: 'var(--bg-slide)',
            opacity: 0.75,
          }}
        >
          {subtitle}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', padding: '0.9rem 1.3rem 1rem' }}>
        {products.map((p) => (
          <div key={p.name} style={{ lineHeight: 1.25 }}>
            <div style={{ color: 'var(--color-heading)', fontWeight: 600, fontSize: '1.05rem' }}>
              {p.name}
              {p.tag && (
                <span
                  style={{
                    marginLeft: '0.6em',
                    padding: '1px 7px',
                    borderRadius: 6,
                    border: '1px solid var(--accent-secondary)',
                    color: 'var(--accent-secondary)',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.7rem',
                    fontWeight: 400,
                    verticalAlign: 'middle',
                  }}
                >
                  {p.tag}
                </span>
              )}
            </div>
            <div style={{ color: 'var(--color-text)', opacity: 0.8, fontSize: '0.9rem' }}>{p.detail}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

/** Renders the open source and Pydantic Logfire panels side by side. */
export default function PydanticToday() {
  return (
    <div style={{ display: 'flex', gap: '1.5rem', width: '100%', marginTop: '0.8rem', alignItems: 'stretch' }}>
      <Panel
        title="Open source"
        subtitle="Free - what you build with"
        color="var(--accent)"
        products={OPEN_SOURCE}
      />
      <Panel
        title="Pydantic Logfire"
        subtitle="Commercial Platform"
        color="var(--accent-aqua)"
        products={LOGFIRE}
      />
    </div>
  )
}
