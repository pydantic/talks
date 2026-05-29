/**
 * Stylised mock of a Logfire trace view for the "Trace observability" slide.
 * Left: hierarchical span list with OTel scope badges, waterfall, and durations.
 * Right: detail panel showing the LLM exchange for the highlighted span.
 */

type Scope = 'http' | 'pg' | 'pydantic_ai' | 'logfire'

interface Span {
  depth: number
  scope: Scope
  label: string
  /** Start offset along the trace, percent of total duration. */
  start: number
  /** Bar width, percent of total duration. */
  width: number
  /** Pre-formatted duration label (e.g. "1.32s", "964µs"). */
  duration: string
  /** Whether this row is the "selected" span shown in the right panel. */
  selected?: boolean
}

const SPANS: Span[] = [
  { depth: 0, scope: 'http', label: 'POST /support/ticket/45af-3sda3 200', start: 0, width: 100, duration: '5.50s' },
  { depth: 1, scope: 'pg', label: 'SELECT * FROM tickets WHERE id = …', start: 1, width: 1, duration: '42ms' },
  { depth: 1, scope: 'pydantic_ai', label: 'support_agent run', start: 2, width: 98, duration: '5.39s', selected: true },
  { depth: 2, scope: 'pydantic_ai', label: 'chat claude-haiku-4-5', start: 2, width: 24, duration: '1.32s' },
  { depth: 2, scope: 'pydantic_ai', label: 'running tool: lookup_order', start: 26, width: 0.6, duration: '964µs' },
  { depth: 2, scope: 'pydantic_ai', label: 'chat claude-haiku-4-5', start: 27, width: 35, duration: '1.92s' },
  { depth: 2, scope: 'pydantic_ai', label: 'running tool: query_shipment_status', start: 62, width: 0.6, duration: '1.20ms' },
  { depth: 2, scope: 'pydantic_ai', label: 'chat claude-haiku-4-5', start: 63, width: 37, duration: '2.14s' },
  { depth: 2, scope: 'logfire', label: 'result.output="Great news! I found your order. …"', start: 99.5, width: 0.4, duration: '—' },
]

const SCOPE_COLOR: Record<Scope, string> = {
  http: 'var(--accent)',
  pg: 'var(--accent-tertiary)',
  pydantic_ai: 'var(--accent-aqua)',
  logfire: 'var(--accent-secondary)',
}

const SCOPE_LABEL: Record<Scope, string> = {
  http: 'http',
  pg: 'pg',
  pydantic_ai: 'pydantic-ai',
  logfire: 'logfire',
}

function ScopeBadge({ scope }: { scope: Scope }) {
  const color = SCOPE_COLOR[scope]
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '0.1em 0.55em',
        borderRadius: '0.4em',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.62em',
        color,
        background: `color-mix(in srgb, ${color} 16%, transparent)`,
        border: `1px solid color-mix(in srgb, ${color} 40%, transparent)`,
        whiteSpace: 'nowrap',
        justifySelf: 'end',
      }}
    >
      {SCOPE_LABEL[scope]}
    </span>
  )
}

function Waterfall({ start, width, color }: { start: number; width: number; color: string }) {
  const w = Math.max(width, 1.4)
  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: '0.55em',
        borderRadius: '0.2em',
        background: 'color-mix(in srgb, var(--color-heading) 6%, transparent)',
      }}
    >
      <div
        style={{
          position: 'absolute',
          left: `${start}%`,
          width: `${w}%`,
          top: 0,
          bottom: 0,
          background: color,
          borderRadius: '0.2em',
          boxShadow: `0 0 12px color-mix(in srgb, ${color} 70%, transparent), 0 0 4px color-mix(in srgb, ${color} 90%, transparent)`,
        }}
      />
    </div>
  )
}

function TreeLabel({ depth, label }: { depth: number; label: string }) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.68em',
        color: 'var(--color-heading)',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
      }}
    >
      <span
        aria-hidden
        style={{
          display: 'inline-block',
          width: `${depth * 1.2}em`,
          flexShrink: 0,
          color: 'color-mix(in srgb, var(--color-heading) 30%, transparent)',
        }}
      >
        {depth > 0 ? '└─' : ''}
      </span>
      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</span>
    </span>
  )
}

const WATERFALL_COLOR = 'var(--accent)'

function TraceRow({ span }: { span: Span }) {
  const color = SCOPE_COLOR[span.scope]
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 2.6fr) 6em minmax(0, 1.9fr) 5em',
        gap: '0.85em',
        alignItems: 'center',
        padding: '0.18em 0.7em',
        borderLeft: `2px solid ${span.selected ? color : 'transparent'}`,
        background: span.selected ? `color-mix(in srgb, ${color} 8%, transparent)` : 'transparent',
        borderRadius: '0.25em',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', minWidth: 0 }}>
        <TreeLabel depth={span.depth} label={span.label} />
      </div>
      <ScopeBadge scope={span.scope} />
      <Waterfall start={span.start} width={span.width} color={WATERFALL_COLOR} />
      <span
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.62em',
          color: 'color-mix(in srgb, var(--color-heading) 55%, transparent)',
          textAlign: 'right',
        }}
      >
        {span.duration}
      </span>
    </div>
  )
}

function Panel({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div
      style={{
        flex: 1,
        minWidth: 0,
        display: 'flex',
        flexDirection: 'column',
        background: '#06080c',
        border: '1px solid color-mix(in srgb, var(--color-heading) 12%, transparent)',
        borderRadius: '0.6em',
        boxShadow: '0 0 0 4px color-mix(in srgb, var(--accent) 5%, transparent)',
        overflow: 'hidden',
        ...style,
      }}
    >
      {children}
    </div>
  )
}

function PanelHeader({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0.55em 0.9em',
        borderBottom: '1px solid color-mix(in srgb, var(--color-heading) 10%, transparent)',
        background: '#0b0e14',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.62em',
        color: 'color-mix(in srgb, var(--color-heading) 65%, transparent)',
      }}
    >
      {children}
    </div>
  )
}

function CodeBlock({ json }: { json: string }) {
  return (
    <pre
      style={{
        margin: '0.4em 0 0 0',
        padding: '0.45em 0.7em',
        borderRadius: '0.35em',
        background: 'color-mix(in srgb, var(--color-heading) 5%, transparent)',
        border: '1px solid color-mix(in srgb, var(--color-heading) 8%, transparent)',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.78em',
        lineHeight: 1.4,
        color: 'var(--color-heading)',
        whiteSpace: 'pre',
        overflow: 'auto',
      }}
    >
      {json}
    </pre>
  )
}

function ToolCallBlock({ name, json }: { name: string; json: string }) {
  return (
    <div style={{ marginTop: '0.55em' }}>
      <div
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.85em',
          color: 'color-mix(in srgb, var(--color-heading) 60%, transparent)',
          marginBottom: '0.15em',
        }}
      >
        Tool call · <span style={{ color: 'var(--accent-aqua)' }}>{name}</span>
      </div>
      <CodeBlock json={json} />
    </div>
  )
}

function ConversationSection({
  kind,
  role,
  roleSuffix,
  body,
}: {
  kind: 'Input' | 'Output'
  role: string
  roleSuffix?: string
  body: React.ReactNode
}) {
  return (
    <div style={{ padding: '0.7em 0.9em', borderBottom: '1px solid color-mix(in srgb, var(--color-heading) 6%, transparent)' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.55em',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.58em',
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
          color: 'color-mix(in srgb, var(--color-heading) 45%, transparent)',
          marginBottom: '0.35em',
        }}
      >
        <span>▸ {kind}</span>
        <span
          style={{
            padding: '0.05em 0.45em',
            borderRadius: '0.3em',
            background: 'color-mix(in srgb, var(--accent-aqua) 14%, transparent)',
            color: 'var(--accent-aqua)',
            textTransform: 'lowercase',
            letterSpacing: 0,
          }}
        >
          {role}
        </span>
        {roleSuffix && (
          <span
            style={{
              padding: '0.05em 0.45em',
              borderRadius: '0.3em',
              background: 'color-mix(in srgb, var(--color-heading) 8%, transparent)',
              color: 'color-mix(in srgb, var(--color-heading) 65%, transparent)',
              textTransform: 'none',
              letterSpacing: 0,
            }}
          >
            {roleSuffix}
          </span>
        )}
      </div>
      <div
        style={{
          fontFamily: 'var(--font-body)',
          fontSize: '0.68em',
          lineHeight: 1.45,
          color: 'var(--color-text)',
        }}
      >
        {body}
      </div>
    </div>
  )
}

export default function TraceObservability() {
  return (
    <div
      style={{
        display: 'flex',
        gap: '1.2em',
        width: '100%',
        fontSize: '0.95em',
        transformOrigin: 'top center',
      }}
    >
      <Panel style={{ flex: '1.35' }}>
        <PanelHeader>
          <span>Trace · 5.50s · 9 spans</span>
          <span style={{ display: 'flex', gap: '0.6em' }}>
            <span style={{ color: 'var(--accent-aqua)' }}>● live</span>
            <span>filters · search</span>
          </span>
        </PanelHeader>
        <div style={{ padding: '0.4em 0.2em', display: 'flex', flexDirection: 'column', gap: '0.05em' }}>
          {SPANS.map((s, i) => (
            <TraceRow key={i} span={s} />
          ))}
        </div>
      </Panel>

      <Panel style={{ flex: '1' }}>
        <PanelHeader>
          <span style={{ color: 'var(--color-heading)', fontWeight: 600 }}>support_agent run</span>
          <span style={{ display: 'flex', gap: '0.8em' }}>
            <span style={{ color: 'var(--accent-aqua)' }}>Run</span>
            <span>Details</span>
            <span>Raw</span>
          </span>
        </PanelHeader>
        <div style={{ overflow: 'auto', flex: 1 }}>
          <ConversationSection
            kind="Input"
            role="system"
            body={
              <>
                You are a support agent for an online retailer. Use the available tools to look up
                orders and shipments before responding.{' '}
                <span style={{ color: 'var(--color-muted)' }}>… show more</span>
              </>
            }
          />
          <ConversationSection
            kind="Input"
            role="user"
            body={<>My order #4521 is missing — can you tell me where it is?</>}
          />
          <ConversationSection
            kind="Output"
            role="assistant"
            body={
              <>
                I'll look up your order and check the shipment status for you.
                <ToolCallBlock
                  name="lookup_order"
                  json={`{
  "order_id": 4521
}`}
                />
              </>
            }
          />
          <ConversationSection
            kind="Output"
            role="tool"
            roleSuffix="lookup_order"
            body={
              <CodeBlock
                json={`{
  "order_id": 4521,
  "customer": "jean@example.com",
  "shipped_on": "2026-04-26",
  "carrier": "DHL",
  "tracking": "JD0123456789"
}`}
              />
            }
          />
        </div>
      </Panel>
    </div>
  )
}
