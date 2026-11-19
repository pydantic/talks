/**
 * Stylised terminal mock showing an agent investigating production traces
 * via the Logfire MCP. Used on the "Aggregate observability" slide.
 */

const Dot = ({ color }: { color: string }) => (
  <span style={{ width: '0.7em', height: '0.7em', borderRadius: '50%', background: color, display: 'inline-block' }} />
)

export default function AgentTerminal() {
  return (
    <>
      <style>{`
        @keyframes agent-term-blink { 0%, 49% { opacity: 1 } 50%, 100% { opacity: 0 } }
        @keyframes agent-term-spin {
          0%   { content: '⠋' }
          12%  { content: '⠙' }
          25%  { content: '⠹' }
          37%  { content: '⠸' }
          50%  { content: '⠼' }
          62%  { content: '⠴' }
          75%  { content: '⠦' }
          87%  { content: '⠧' }
          100% { content: '⠇' }
        }
        .agent-term-spinner::before {
          content: '⠋';
          animation: agent-term-spin 0.9s steps(8) infinite;
        }
        .agent-term-cursor {
          animation: agent-term-blink 1s step-end infinite;
        }
      `}</style>

      <div
        style={{
          width: '100%',
          background: '#06080c',
          border: '1px solid color-mix(in srgb, var(--color-heading) 12%, transparent)',
          borderRadius: '0.6em',
          boxShadow: '0 0 0 4px color-mix(in srgb, var(--accent) 5%, transparent)',
          overflow: 'hidden',
          fontFamily: 'var(--font-mono)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.45em',
            padding: '0.65em 0.9em',
            borderBottom: '1px solid color-mix(in srgb, var(--color-heading) 10%, transparent)',
            background: '#0b0e14',
          }}
        >
          <Dot color="#ff5f57" />
          <Dot color="#febc2e" />
          <Dot color="#28c840" />
          <span
            style={{
              marginLeft: '0.9em',
              fontSize: '0.62em',
              color: 'color-mix(in srgb, var(--color-heading) 55%, transparent)',
            }}
          >
            ~/support-agent — claude
          </span>
        </div>

        <div
          style={{
            padding: '1em 1.1em',
            fontSize: '0.78em',
            lineHeight: 1.55,
            color: 'var(--color-heading)',
          }}
        >
          <div>
            <span style={{ color: 'var(--accent-aqua)' }}>{'❯'}</span>{' '}
            why is the support agent failing to show shipping status sometimes?
          </div>

          <div style={{ height: '0.9em' }} />

          <div style={{ color: 'color-mix(in srgb, var(--color-heading) 55%, transparent)', fontStyle: 'italic' }}>
            thinking: I'll use logfire MCP to find spans for the shipping agent
          </div>

          <div style={{ height: '0.9em' }} />

          <div>
            <span style={{ color: 'var(--accent)' }}>⏺</span>{' '}
            <span style={{ color: 'var(--accent-aqua)' }}>MCP</span> logfire_query(
            <span style={{ color: 'color-mix(in srgb, var(--accent-aqua) 65%, transparent)' }}>
              {"'select … from records where … group by …'"}
            </span>
            )
          </div>

          <div
            style={{
              marginTop: '0.5em',
              padding: '0.5em 0.7em',
              borderLeft: '2px solid color-mix(in srgb, var(--accent-aqua) 50%, transparent)',
              background: 'color-mix(in srgb, var(--accent-aqua) 5%, transparent)',
              color: 'color-mix(in srgb, var(--color-heading) 75%, transparent)',
              fontSize: '0.88em',
              whiteSpace: 'pre',
              overflow: 'hidden',
            }}
          >
{`[
  {...},
  {...},
  {...},
  {...}
]`}
          </div>

          <div style={{ height: '0.9em' }} />

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6em' }}>
            <span
              className="agent-term-spinner"
              style={{ color: 'var(--accent-aqua)', fontSize: '1.05em', display: 'inline-block', width: '1em' }}
            />
            <span style={{ color: 'color-mix(in srgb, var(--color-heading) 65%, transparent)', fontStyle: 'italic' }}>
              Identifying issues…
            </span>
            <span
              className="agent-term-cursor"
              style={{
                display: 'inline-block',
                width: '0.55em',
                height: '1em',
                background: 'var(--color-heading)',
                marginLeft: '0.3em',
                verticalAlign: 'middle',
              }}
            />
          </div>
        </div>
      </div>
    </>
  )
}
