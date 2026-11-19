/**
 * Cyclic flow for the "Continuous learning" slide.
 * Four pill cards arranged as a diamond (top/right/bottom/left), connected by
 * curved arrows running clockwise to form a closed loop.
 */

interface Step {
  label: string
  cx: number
  cy: number
}

const STEPS: Step[] = [
  { label: 'Agent state updated', cx: 50, cy: 12 },
  { label: 'Production traces', cx: 88, cy: 50 },
  { label: 'Online evals', cx: 50, cy: 88 },
  { label: 'Optimize', cx: 12, cy: 50 },
]

const ASPECT_W = 1000
const ASPECT_H = 900

const CENTER_X = 50
const CENTER_Y = 50
/** Radius of the arrow arc, as a percentage of viewBox. */
const ARROW_R = 30
const ARROW_GAP_DEG = 18

function polar(angleDeg: number, r: number) {
  const a = (angleDeg - 90) * (Math.PI / 180)
  return {
    x: CENTER_X + r * Math.cos(a),
    y: CENTER_Y + r * Math.sin(a),
  }
}

function Arrow({ fromAngle, toAngle }: { fromAngle: number; toAngle: number }) {
  const start = polar(fromAngle + ARROW_GAP_DEG, ARROW_R)
  const end = polar(toAngle - ARROW_GAP_DEG, ARROW_R)
  return (
    <path
      d={`M ${start.x} ${start.y} A ${ARROW_R} ${ARROW_R} 0 0 1 ${end.x} ${end.y}`}
      fill="none"
      stroke="var(--accent)"
      strokeWidth={0.6}
      strokeLinecap="round"
      markerEnd="url(#cl-arrow-head)"
      opacity={0.85}
    />
  )
}

export default function ContinuousLearningLoop() {
  const angles = [0, 90, 180, 270]
  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        aspectRatio: `${ASPECT_W} / ${ASPECT_H}`,
        margin: '0 auto',
      }}
    >
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
      >
        <defs>
          <marker
            id="cl-arrow-head"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--accent)" />
          </marker>
        </defs>
        {angles.map((a, i) => (
          <Arrow key={i} fromAngle={a} toAngle={angles[(i + 1) % angles.length] || 360} />
        ))}
      </svg>

      {STEPS.map((s, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            left: `${s.cx}%`,
            top: `${s.cy}%`,
            transform: 'translate(-50%, -50%)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.55em',
            padding: '0.5em 1em 0.5em 0.5em',
            borderRadius: '999px',
            background: 'var(--bg-slide)',
            border: '2px solid var(--accent)',
            boxShadow: '0 0 0 4px color-mix(in srgb, var(--accent) 9%, transparent)',
            whiteSpace: 'nowrap',
            fontSize: '0.7em',
          }}
        >
          <span
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '1.8em',
              height: '1.8em',
              borderRadius: '999px',
              background: 'var(--accent)',
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
      ))}
    </div>
  )
}
