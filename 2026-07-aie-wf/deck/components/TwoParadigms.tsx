const LIGHT = '#4ad7c5'; // --accent-aqua
const HEAVY = '#e520a8'; // --accent
const DANGER = '#ff4d4d';

const LIGHT_TASKS = ['Calculations', 'Run queries', 'Pass tool outputs to next tool', 'Render a chart'];
const HEAVY_TASKS = ['Clone a repo', 'Run a coding agent', 'Heavy computation', 'Compilation'];

const ENV_OPTIONS = ['VM', 'Container', 'gVisor', 'Firecracker'];

type BoxProps = {
  title: string;
  color: string;
  intro?: string;
  bullets: string[];
  footerLabel: string;
  footerValue: string;
};

function Box({ title, color, intro, bullets, footerLabel, footerValue }: BoxProps) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        border: `2px solid ${color}`,
        borderRadius: '18px',
        background: `${color}14`,
        padding: '1.1rem 1.6rem',
        height: '100%',
        boxSizing: 'border-box',
      }}
    >
      <div
        style={{
          fontSize: '1.8rem',
          fontWeight: 700,
          color,
          textAlign: 'center',
          marginBottom: '0.8rem',
        }}
      >
        {title}
      </div>

      <div style={{ flex: 1 }}>
        {intro ? (
          <div style={{ fontSize: '1.25rem', marginBottom: '0.6rem' }}>{intro}</div>
        ) : null}
        <ul style={{ margin: 0, paddingLeft: '1.3rem', fontSize: '1.25rem', lineHeight: intro ? 1.45 : 1.7 }}>
          {bullets.map((b) => (
            <li key={b}>{b}</li>
          ))}
        </ul>
      </div>

      <div style={{ marginTop: '0.9rem', paddingTop: '0.7rem', borderTop: `1px solid ${color}44` }}>
        <div
          style={{
            fontSize: '0.78rem',
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            color: 'var(--color-muted)',
          }}
        >
          {footerLabel}
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.5rem', fontWeight: 700, color }}>
          {footerValue}
        </div>
      </div>
    </div>
  );
}

export default function TwoParadigms({ variant }: { variant: 'tasks' | 'env' }) {
  const lightColor = variant === 'env' ? DANGER : LIGHT;
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '2.5rem',
        marginTop: '1rem',
      }}
    >
      <Box
        title="Light tasks"
        color={lightColor}
        intro={variant === 'env' ? 'Execution environments:' : undefined}
        bullets={variant === 'tasks' ? LIGHT_TASKS : ENV_OPTIONS}
        footerLabel={variant === 'env' ? 'typical boot time' : 'typical duration'}
        footerValue={variant === 'env' ? '1-5s' : 'milliseconds'}
      />
      <Box
        title="Heavy tasks"
        color={HEAVY}
        intro={variant === 'env' ? 'Execution environments:' : undefined}
        bullets={variant === 'tasks' ? HEAVY_TASKS : ENV_OPTIONS}
        footerLabel={variant === 'env' ? 'typical boot time' : 'typical duration'}
        footerValue={variant === 'env' ? '1-5s' : 'minutes'}
      />
    </div>
  );
}
