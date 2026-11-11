interface Step {
  label: string;
  product: string;
  /** Center x as a percentage of the container width. */
  cx: number;
  /** Center y as a percentage of the container height. */
  cy: number;
}

const STEPS: Step[] = [
  { label: 'Agent loop', product: 'Pydantic AI', cx: 10, cy: 92 },
  { label: 'Code execution', product: 'Agent Sandboxes', cx: 28.75, cy: 72 },
  { label: 'Observability', product: 'Logfire', cx: 47.5, cy: 52 },
  { label: 'Evals', product: 'Logfire', cx: 66.25, cy: 32 },
  { label: 'Continuous learning', product: 'Logfire + Gateway', cx: 85, cy: 12 },
];

const ASPECT_W = 1000;
const ASPECT_H = 460;

export default function LLMOpsSteps() {
  return (
    <div style={{ position: 'relative', width: '82%', aspectRatio: `${ASPECT_W} / ${ASPECT_H}`, margin: '0 auto' }}>
      {STEPS.map((s, i) => {
        const isLast = i === STEPS.length - 1;
        const color = isLast ? 'var(--accent-aqua)' : 'var(--accent)';
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
              border: `2px solid ${color}`,
              boxShadow: isLast
                ? '0 0 18px color-mix(in srgb, var(--accent-aqua) 30%, transparent), 0 0 0 4px color-mix(in srgb, var(--accent-aqua) 12%, transparent)'
                : '0 0 0 3px color-mix(in srgb, var(--accent) 8%, transparent)',
              whiteSpace: 'nowrap',
              fontSize: '1em',
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
                background: color,
                color: 'var(--bg-slide)',
                fontFamily: 'var(--font-mono)',
                fontWeight: 700,
                fontSize: '0.95em',
                flexShrink: 0,
              }}
            >
              {i + 1}
            </span>
            <span style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.15 }}>
              <span style={{ color: 'var(--color-heading)', fontWeight: 600, letterSpacing: '0.01em' }}>{s.label}</span>
              <span style={{ color, fontFamily: 'var(--font-mono)', fontSize: '0.7em' }}>{s.product}</span>
            </span>
          </div>
        );
      })}
    </div>
  );
}
