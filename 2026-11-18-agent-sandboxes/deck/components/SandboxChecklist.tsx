import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';

interface Item {
  label: string;
  text: string;
  icon: ReactNode;
  /** What provides this, shown with a tick on the second frame. */
  how: string;
}

const ITEMS: Item[] = [
  {
    label: 'Runtime',
    how: 'Monty',
    text: 'Starts in milliseconds',
    icon: (
      <>
        <polyline points="4 17 10 11 4 5" />
        <line x1="12" y1="19" x2="20" y2="19" />
      </>
    ),
  },
  {
    label: 'Session identity',
    how: 'External Functions',
    text: 'Which agent, user and run',
    icon: (
      <>
        <rect x="2" y="4" width="20" height="16" rx="2" />
        <circle cx="8.5" cy="11" r="2.5" />
        <path d="M4.5 17c.6-2 2.1-3 4-3s3.400 1 4 3" />
        <line x1="15" y1="9.500" x2="19" y2="9.500" />
        <line x1="15" y1="13.500" x2="19" y2="13.500" />
      </>
    ),
  },
  {
    label: 'Policy engine',
    how: 'External Functions',
    text: 'What this run may do',
    icon: (
      <>
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        <path d="m9 12 2 2 4-4" />
      </>
    ),
  },
  {
    label: 'Network filters',
    how: 'External Functions',
    text: 'Which hosts it can reach',
    icon: <polygon points="22 3 2 3 10 12.460 10 19 14 21 14 12.460 22 3" />,
  },
  {
    label: 'Network mutation',
    how: 'External Functions',
    text: 'Rewrite outgoing requests',
    icon: (
      <>
        <path d="M16 3h5v5" />
        <path d="M4 20 21 3" />
        <path d="M21 16v5h-5" />
        <path d="m15 15 6 6" />
        <path d="m4 4 5 5" />
      </>
    ),
  },
  {
    label: 'Secrets injection',
    how: 'External Functions',
    text: 'Keys the code never sees',
    icon: (
      <>
        <circle cx="7.500" cy="15.500" r="5.500" />
        <path d="m21 2-9.600 9.600" />
        <path d="m15.500 7.500 3 3L22 7l-3-3" />
      </>
    ),
  },
  {
    label: 'Sys call + fs filters',
    how: 'External Functions',
    text: 'Allowed files and OS calls',
    icon: (
      <>
        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
        <rect x="9" y="13" width="6" height="4" rx="1" />
        <path d="M10.500 13v-1.500a1.500 1.500 0 0 1 3 0V13" />
      </>
    ),
  },
  {
    label: 'Snapshots',
    how: 'Monty Snapshots',
    text: 'A paused run, as bytes',
    icon: (
      <>
        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
        <circle cx="12" cy="13" r="4" />
      </>
    ),
  },
  {
    label: 'Resume',
    how: 'Monty Snapshots',
    text: 'Pick it up later, anywhere',
    icon: (
      <>
        <circle cx="12" cy="12" r="10" />
        <polygon points="10 8 16 12 10 16 10 8" />
      </>
    ),
  },
  {
    label: 'Observability',
    how: 'Logfire',
    text: 'A trace for every run',
    icon: <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />,
  },
  {
    label: 'Audit',
    how: 'Logfire',
    text: 'What ran, and for whom',
    icon: (
      <>
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
      </>
    ),
  },
  {
    label: 'Resource limits',
    how: 'Monty + infra limits',
    text: 'Caps on time and memory',
    icon: (
      <>
        <path d="m12 14 4-4" />
        <path d="M3.340 19a10 10 0 1 1 17.320 0" />
      </>
    ),
  },
];

const ICON_COLORS = ['var(--accent-aqua)', 'var(--accent-tertiary)', 'var(--accent-secondary)'];

const GREEN = '#3ddc84';

const fade = 'opacity 0.45s ease, border-color 0.45s ease, transform 0.45s ease';

/**
 * The twelve parts of a sandbox. Press `.` (the `>` key) to tick every box and show what
 * provides it, `,` (the `<` key) to go back.
 */
export default function SandboxChecklist() {
  const [solved, setSolved] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // only react while this slide is the one on screen
      if (!ref.current?.closest('.slide')?.classList.contains('slide--active')) return;
      if (e.key === '.' || e.key === '>') setSolved(true);
      else if (e.key === ',' || e.key === '<') setSolved(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <div ref={ref} style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', marginTop: '0.75rem' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.35rem 0.9rem' }}>
        {ITEMS.map((item, i) => {
          const iconColor = ICON_COLORS[i % ICON_COLORS.length]!;
          return (
            <div
              key={item.label}
              style={{
                position: 'relative',
                background: 'var(--surface)',
                border: `1.5px solid ${solved ? `color-mix(in srgb, ${GREEN} 55%, transparent)` : 'rgba(255,255,255,0.14)'}`,
                borderRadius: 14,
                padding: '1rem 0.95rem 0.9rem',
                transition: fade,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <span
                  style={{
                    flex: '0 0 auto',
                    width: '2.35rem',
                    height: '2.35rem',
                    borderRadius: 10,
                    background: `color-mix(in srgb, ${iconColor} 16%, transparent)`,
                    color: iconColor,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    {item.icon}
                  </svg>
                </span>
                <span style={{ color: 'var(--color-heading)', fontWeight: 700, fontSize: '1.05rem', lineHeight: 1.15, whiteSpace: 'nowrap' }}>{item.label}</span>
              </div>
              <div style={{ marginTop: '0.5rem', fontSize: '0.92rem', lineHeight: 1.3 }}>{item.text}</div>
              {/* tick + what provides it, sitting over the top right corner of the box */}
              <span
                style={{
                  position: 'absolute',
                  top: '-0.8rem',
                  right: '0.6rem',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                  padding: '0.18rem 0.6rem 0.18rem 0.4rem',
                  borderRadius: 999,
                  background: GREEN,
                  color: '#0b1f13',
                  fontWeight: 700,
                  fontSize: '0.82rem',
                  lineHeight: 1.2,
                  whiteSpace: 'nowrap',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.35)',
                  opacity: solved ? 1 : 0,
                  transform: solved ? 'none' : 'translateY(6px) scale(0.9)',
                  transition: fade,
                  transitionDelay: solved ? `${i * 45}ms` : '0ms',
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
                {item.how}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
