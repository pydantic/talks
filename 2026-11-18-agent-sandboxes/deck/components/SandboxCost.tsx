import type { CSSProperties, ReactNode } from 'react';

const mono: CSSProperties = { fontFamily: 'var(--font-mono)' };

const OTHERS = 'var(--accent-secondary)';
const OURS = 'var(--accent-aqua)';

interface Need {
  title: string;
  text: ReactNode;
}

const NEEDS: Need[] = [
  { title: 'A new sandbox', text: 'on every agent step' },
  { title: 'A REPL', text: <>so state survives between <code>run_code</code> calls</> },
  { title: 'External function calls', text: 'back into your process, in the middle of execution' },
];

interface Metric {
  label: string;
  unit: string;
  others: number;
  ours: number;
  /** e.g. "faster", follows the computed multiple */
  gain: string;
}

const METRICS: Metric[] = [
  { label: 'Time to a new sandbox', unit: 'ms', others: 1000, ours: 20, gain: 'faster' },
  { label: 'Memory per sandbox', unit: 'MB', others: 128, ours: 2, gain: 'less memory' },
];

function Bar({ name, value, max, unit, color }: { name: string; value: number; max: number; unit: string; color: string }) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.25rem' }}>
        <span style={{ color: 'var(--color-heading)', fontWeight: 600, fontSize: '1.05rem' }}>{name}</span>
        <span style={{ ...mono, color, fontWeight: 700, fontSize: '1.2rem', whiteSpace: 'nowrap' }}>
          {value} {unit}
        </span>
      </div>
      <div style={{ height: '1.15rem', borderRadius: 7, background: 'rgba(255,255,255,0.07)', overflow: 'hidden' }}>
        <div style={{ width: `${(value / max) * 100}%`, minWidth: 7, height: '100%', borderRadius: 8, background: color }} />
      </div>
    </div>
  );
}

function MetricCard({ m }: { m: Metric }) {
  return (
    <div
      style={{
        background: 'var(--surface)',
        border: '1px solid rgba(255,255,255,0.12)',
        borderRadius: 14,
        padding: '0.9rem 1.2rem 0.85rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.6rem',
      }}
    >
      <div style={{ ...mono, color: 'var(--color-muted)', fontSize: '0.8rem', letterSpacing: '0.08em', textTransform: 'uppercase' }}>{m.label}</div>
      <Bar name="Most sandboxes" value={m.others} max={m.others} unit={m.unit} color={OTHERS} />
      <Bar name="Agent Sandboxes" value={m.ours} max={m.others} unit={m.unit} color={OURS} />
      <div style={{ marginTop: 'auto', paddingTop: '0.3rem', display: 'flex', alignItems: 'baseline', gap: '0.55rem' }}>
        <span style={{ fontSize: '2.4rem', fontWeight: 800, color: 'var(--accent)', lineHeight: 1, letterSpacing: '-0.02em' }}>{Math.round(m.others / m.ours)}x</span>
        <span style={{ fontSize: '1.35rem', fontWeight: 700, color: 'var(--color-heading)' }}>{m.gain}</span>
      </div>
    </div>
  );
}

export default function SandboxCost() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem', marginTop: '0.3rem' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '0.92fr 1fr 1fr', gap: '1.3rem' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.7rem', paddingTop: '0.2rem' }}>
          <div style={{ ...mono, color: 'var(--color-muted)', fontSize: '0.8rem', letterSpacing: '0.08em', textTransform: 'uppercase' }}>Code mode needs</div>
          {NEEDS.map((n, i) => (
            <div key={n.title} style={{ display: 'flex', gap: '0.8rem', alignItems: 'flex-start' }}>
              <span
                style={{
                  ...mono,
                  flex: '0 0 auto',
                  width: '1.7rem',
                  height: '1.7rem',
                  borderRadius: '50%',
                  border: `1.5px solid ${OURS}`,
                  color: OURS,
                  fontSize: '0.85rem',
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginTop: '0.1rem',
                }}
              >
                {i + 1}
              </span>
              <div style={{ lineHeight: 1.3 }}>
                <div style={{ color: 'var(--color-heading)', fontWeight: 700, fontSize: '1.2rem' }}>{n.title}</div>
                <div style={{ fontSize: '1rem' }}>{n.text}</div>
              </div>
            </div>
          ))}
        </div>
        {METRICS.map((m) => (
          <MetricCard key={m.label} m={m} />
        ))}
      </div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1.3rem',
          border: '1.5px solid var(--accent)',
          borderRadius: 14,
          background: 'color-mix(in srgb, var(--accent) 9%, transparent)',
          padding: '0.65rem 1.3rem',
        }}
      >
        <span style={{ ...mono, fontSize: '2.3rem', fontWeight: 800, color: 'var(--accent)', whiteSpace: 'nowrap', lineHeight: 1 }}>&lt;1 ms</span>
        <span style={{ fontSize: '1.15rem', lineHeight: 1.35 }}>
          <strong>to run the code an agent writes.</strong> Everything else is sandbox overhead, and with a VM you pay for the machine while it sits idle.
        </span>
      </div>
    </div>
  );
}
