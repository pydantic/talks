import { useEffect, useState } from 'react';

interface Span {
  name: string;
  start: number;
  end: number;
  depth: number;
}

// The spans from the demo request, in seconds from the start of the request.
const SPANS: Span[] = [
  { name: 'GET /weather', start: 0, end: 8, depth: 0 },
  { name: 'weather_agent run', start: 0.2, end: 7.8, depth: 1 },
  { name: 'chat gpt-5.6', start: 0.4, end: 3.6, depth: 2 },
  { name: 'run_code', start: 3.7, end: 4.3, depth: 2 },
  { name: 'chat gpt-5.6', start: 4.4, end: 7.6, depth: 2 },
];
const TOTAL = 8;
const HOLD = 2;

// Headless Chrome (the PDF build) does not fire `beforeprint`, so detect it up front and render the final frame.
function isPrint() {
  if (typeof window === 'undefined') return false;
  if (window.matchMedia?.('print').matches) return true;
  return /HeadlessChrome/.test(navigator.userAgent);
}

function useClock() {
  const [t, setT] = useState(() => (isPrint() ? TOTAL : 0));
  const [frozen, setFrozen] = useState(isPrint);
  useEffect(() => {
    const freeze = () => {
      setFrozen(true);
      setT(TOTAL);
    };
    window.addEventListener('beforeprint', freeze);
    return () => window.removeEventListener('beforeprint', freeze);
  }, []);
  useEffect(() => {
    if (frozen) return;
    let raf = 0;
    let t0 = performance.now();
    const tick = (now: number) => {
      const elapsed = (now - t0) / 1000;
      if (elapsed > TOTAL + HOLD) {
        t0 = now;
        setT(0);
      } else {
        setT(Math.min(elapsed, TOTAL));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [frozen]);
  return t;
}

const mono: React.CSSProperties = { fontFamily: 'var(--font-mono)' };

const panel: React.CSSProperties = {
  flex: 1,
  minWidth: 0,
  background: 'var(--surface)',
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: '10px',
  padding: '0.6rem 0.9rem',
  fontSize: '0.95rem',
  minHeight: '10.5rem',
};

const panelTitle: React.CSSProperties = {
  ...mono,
  fontSize: '0.8rem',
  letterSpacing: '0.06em',
  textTransform: 'uppercase',
  color: 'var(--color-muted)',
  marginBottom: '0.4rem',
};

function Row({ span, t, pending }: { span: Span; t: number; pending: boolean }) {
  const dur = Math.min(t, span.end) - span.start;
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        paddingLeft: `${span.depth * 1.1}rem`,
        lineHeight: 1.75,
        opacity: pending ? 0.75 : 1,
        color: pending ? 'var(--accent)' : 'var(--color-heading)',
      }}
    >
      <span
        className={pending ? 'ps-dot ps-dot--pending' : 'ps-dot'}
        style={{ background: pending ? 'var(--accent)' : 'var(--accent-tertiary)' }}
      />
      <span style={{ ...mono, flex: '0 0 11rem', whiteSpace: 'nowrap' }}>{span.name}</span>
      <span
        style={{
          height: '0.5rem',
          width: `${(dur / TOTAL) * 9}rem`,
          background: pending ? 'var(--accent)' : 'var(--accent-tertiary)',
          borderRadius: '3px',
          opacity: 0.8,
        }}
      />
      <span style={{ ...mono, color: 'var(--color-muted)', fontSize: '0.85em' }}>{dur.toFixed(1)}s</span>
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return <div style={{ ...mono, color: 'var(--color-muted)', lineHeight: 1.75 }}>{text}</div>;
}

export default function PendingSpans() {
  const t = useClock();
  const finished = SPANS.filter((s) => s.end <= t);
  const started = SPANS.filter((s) => s.start <= t);
  const width = 100;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', marginTop: '0.4rem' }}>
      <style>{`
        @keyframes ps-pulse { 0%, 100% { transform: scale(1); opacity: 1 } 50% { transform: scale(1.6); opacity: 0.5 } }
        .ps-dot { display: inline-block; width: 0.55rem; height: 0.55rem; border-radius: 50%; flex: none }
        .ps-dot--pending { animation: ps-pulse 1s ease-in-out infinite }
        @media print { .ps-dot--pending { animation: none } }
      `}</style>

      {/* timeline */}
      <svg viewBox={`0 0 ${width} 22`} width="100%" style={{ display: 'block', height: '5.2rem' }} preserveAspectRatio="none">
        {SPANS.map((s, i) => {
          const x = (s.start / TOTAL) * width;
          const w = ((Math.min(t, s.end) - s.start) / TOTAL) * width;
          const y = 1 + i * 3.6;
          const done = s.end <= t;
          return s.start <= t ? (
            <rect
              key={i}
              x={x}
              y={y}
              width={Math.max(w, 0.3)}
              height={2.6}
              rx={0.4}
              fill={done ? 'var(--accent-tertiary)' : 'var(--accent)'}
              opacity={done ? 0.85 : 0.6}
            />
          ) : null;
        })}
        <line x1={(t / TOTAL) * width} y1={0} x2={(t / TOTAL) * width} y2={22} stroke="var(--accent-aqua)" strokeWidth={0.3} />
      </svg>
      <div style={{ ...mono, display: 'flex', justifyContent: 'space-between', color: 'var(--color-muted)', fontSize: '0.8rem', marginTop: '-0.5rem' }}>
        <span>request starts</span>
        <span>t = {t.toFixed(1)}s</span>
        <span>response sent</span>
      </div>

      {/* panels */}
      <div style={{ display: 'flex', gap: '1rem' }}>
        <div style={panel}>
          <div style={panelTitle}>Plain OTLP: spans arrive when they end</div>
          {finished.length === 0 ? <Empty text="nothing yet..." /> : null}
          {[...finished]
            .sort((a, b) => a.end - b.end)
            .map((s, i) => (
              <Row key={i} span={s} t={t} pending={false} />
            ))}
        </div>
        <div style={panel}>
          <div style={panelTitle}>With pending spans: rows appear when they start</div>
          {started.length === 0 ? <Empty text="nothing yet..." /> : null}
          {started.map((s, i) => (
            <Row key={i} span={s} t={t} pending={s.end > t} />
          ))}
        </div>
      </div>
    </div>
  );
}
