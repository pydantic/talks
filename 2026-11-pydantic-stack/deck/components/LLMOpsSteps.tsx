interface Step {
  label: string;
  product: string;
  /** Center x as a percentage of the container width. */
  cx: number;
  /** Center y as a percentage of the container height. */
  cy: number;
}

const STEPS: Step[] = [
  { label: 'Validation', product: 'Pydantic', cx: 10, cy: 90.5 },
  { label: 'Agent loop', product: 'Pydantic AI', cx: 25, cy: 74 },
  { label: 'Code execution', product: 'Agent Sandboxes', cx: 40, cy: 57.5 },
  { label: 'Observability', product: 'Logfire', cx: 55, cy: 41 },
  { label: 'Evals', product: 'Logfire', cx: 70, cy: 24.5 },
  { label: 'Continuous learning', product: 'Logfire + Gateway', cx: 85, cy: 8 },
];

const ASPECT_W = 1000;
const ASPECT_H = 540;

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
