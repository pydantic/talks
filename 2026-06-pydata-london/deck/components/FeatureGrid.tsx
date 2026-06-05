/**
 * Feature grid for Logfire free tier / pricing slides.
 */

interface Feature {
  number: string
  title: string
  description: string
}

export default function FeatureGrid({ features, columns = 2 }: { features: Feature[]; columns?: number }) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gap: '1.5rem',
        marginTop: '1rem',
      }}
    >
      {features.map((f, i) => (
        <div
          key={i}
          style={{
            borderTop: '2px solid var(--accent-secondary)',
            paddingTop: '0.8rem',
          }}
        >
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.8rem',
              color: 'var(--accent-secondary)',
              marginRight: '0.5rem',
            }}
          >
            {f.number}
          </span>
          <span style={{ fontWeight: 700, fontSize: '1.1rem', color: 'var(--color-heading)' }}>
            {f.title}
          </span>
          <p style={{ color: 'var(--color-text)', fontSize: '0.9rem', marginTop: '0.4rem', lineHeight: 1.4 }}>
            {f.description}
          </p>
        </div>
      ))}
    </div>
  )
}
