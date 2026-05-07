interface Step {
  label: string
  /** Center x as a percentage of the container width. */
  cx: number
  /** Center y as a percentage of the container height. */
  cy: number
}

const STEPS: Step[] = [
  { label: 'Structured Output', cx: 16, cy: 86 },
  { label: 'Observability', cx: 33, cy: 68 },
  { label: 'Durable Execution', cx: 50, cy: 50 },
  { label: 'Online & offline Evals', cx: 67, cy: 32 },
  { label: 'Continuous Learning', cx: 84, cy: 14 },
]

const HIGHLIGHTED_LABEL = 'Durable Execution'

const ASPECT_W = 1000
const ASPECT_H = 460

export default function LLMOpsSteps() {
  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        aspectRatio: `${ASPECT_W} / ${ASPECT_H}`,
        margin: '0 auto',
      }}
    >
      {/* Step cards */}
      {STEPS.map((s, i) => {
        const isHighlighted = s.label === HIGHLIGHTED_LABEL
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: `${s.cx}%`,
              top: `${s.cy}%`,
              transform: 'translate(-50%, -50%)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.6em',
              padding: '0.35em 1em 0.35em 0.35em',
              borderRadius: '999px',
              background: 'var(--bg-slide)',
              border: `2px solid ${isHighlighted ? 'var(--accent-aqua)' : 'var(--accent)'}`,
              boxShadow: isHighlighted
                ? '0 0 18px color-mix(in srgb, var(--accent-aqua) 30%, transparent), 0 0 0 4px color-mix(in srgb, var(--accent-aqua) 12%, transparent)'
                : '0 0 0 3px color-mix(in srgb, var(--accent) 8%, transparent)',
              whiteSpace: 'nowrap',
              fontSize: '1.25em',
            }}
          >
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '1.7em',
                height: '1.7em',
                borderRadius: '999px',
                background: isHighlighted ? 'var(--accent-aqua)' : 'var(--accent)',
                color: 'var(--bg-slide)',
                fontFamily: 'var(--font-mono)',
                fontWeight: 700,
                fontSize: '0.95em',
                flexShrink: 0,
              }}
            >
              {i + 1}
            </span>
            <span
              style={{
                color: 'var(--color-heading)',
                fontFamily: 'var(--font-body)',
                fontWeight: 600,
                letterSpacing: '0.01em',
              }}
            >
              {s.label}
            </span>
          </div>
        )
      })}
    </div>
  )
}
