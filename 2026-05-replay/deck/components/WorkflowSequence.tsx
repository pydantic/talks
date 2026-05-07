/**
 * Vertical flowchart of an agent run.
 * Left:   workflow steps (LLM → decision → Tool → … → return).
 * Middle: a Temporal box that mediates every external call. It only spans
 *         rows that participate (first activity → last activity).
 * Right:  activity cards reached via Temporal, with bidirectional arrows.
 *
 * The "spine" connecting the workflow boxes is a single absolutely-positioned
 * vertical line behind everything; opaque step backgrounds mask it where they
 * sit, so the line is only visible in the gaps between boxes.
 */

import { Fragment } from 'react'

type StepKind = 'llm' | 'decision' | 'tool' | 'return'
type ActivityKind = 'model' | 'tool' | 'mcp'

interface Step {
  kind: StepKind
  label: string
  activity?: {
    kind: ActivityKind
    name: string
    detail: string
    meta: string
  }
}

const STEPS: Step[] = [
  {
    kind: 'llm',
    label: 'LLM',
    activity: {
      kind: 'model',
      name: 'model.chat',
      detail: 'claude-sonnet-4-5',
      meta: '1.92s · 12k tokens',
    },
  },
  { kind: 'decision', label: 'decision' },
  {
    kind: 'tool',
    label: 'Tool',
    activity: {
      kind: 'tool',
      name: '@agent.tool',
      detail: 'search_docs',
      meta: 'local · 842µs',
    },
  },
  { kind: 'decision', label: 'decision' },
  {
    kind: 'llm',
    label: 'LLM',
    activity: {
      kind: 'model',
      name: 'model.chat',
      detail: 'claude-sonnet-4-5',
      meta: '1.04s · 18k tokens',
    },
  },
  { kind: 'decision', label: 'decision' },
  {
    kind: 'tool',
    label: 'Tool',
    activity: {
      kind: 'mcp',
      name: 'MCP server',
      detail: 'github.fetch_repo',
      meta: '1.32s · retried ×2',
    },
  },
  { kind: 'decision', label: 'decision' },
  { kind: 'return', label: 'return result' },
]

const STEP_WIDTH = '11rem'
const TEMPORAL_WIDTH = '8rem'
const ACTIVITY_WIDTH = '14rem'
const ARROW_WIDTH = '5rem'
const ROW_GAP = '0.55em'

const KIND_COLOR: Record<ActivityKind, string> = {
  model: 'var(--accent-aqua)',
  tool: 'var(--accent-tertiary)',
  mcp: 'var(--accent-secondary)',
}

const KIND_LABEL: Record<ActivityKind, string> = {
  model: 'model',
  tool: 'tool',
  mcp: 'mcp',
}

const TEMPORAL_COLOR = 'var(--accent)'

function StepBox({ step }: { step: Step }) {
  if (step.kind === 'decision') {
    return (
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          width: '100%',
        }}
      >
        <div
          style={{
            padding: '0.25em 0.9em',
            borderRadius: '999px',
            border: '1px dashed color-mix(in srgb, var(--color-heading) 30%, transparent)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.78em',
            color: 'color-mix(in srgb, var(--color-heading) 60%, transparent)',
            letterSpacing: '0.04em',
            background: 'var(--bg-slide)',
          }}
        >
          {step.label}
        </div>
      </div>
    )
  }

  const isReturn = step.kind === 'return'
  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        boxSizing: 'border-box',
        padding: '0.4em 1em',
        borderRadius: isReturn ? '999px' : '0.5em',
        border: `1px solid color-mix(in srgb, var(--color-heading) ${isReturn ? 50 : 35}%, transparent)`,
        // opaque base + tint, so the spine line behind is fully masked
        background: `linear-gradient(color-mix(in srgb, var(--color-heading) ${isReturn ? 12 : 5}%, var(--bg-slide)), color-mix(in srgb, var(--color-heading) ${isReturn ? 12 : 5}%, var(--bg-slide)))`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.95em',
        fontWeight: 600,
        color: 'var(--color-heading)',
      }}
    >
      {step.label}
    </div>
  )
}

function BiArrow({ color }: { color: string }) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.45em',
        width: '100%',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', height: '0.9em' }}>
        <span
          style={{
            flex: 1,
            height: '2px',
            background: `color-mix(in srgb, ${color} 75%, transparent)`,
          }}
        />
        <span
          style={{
            color: `color-mix(in srgb, ${color} 90%, transparent)`,
            fontFamily: 'var(--font-mono)',
            fontSize: '0.95em',
            lineHeight: 1,
            marginLeft: '-0.1em',
          }}
        >
          ▶
        </span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', height: '0.9em' }}>
        <span
          style={{
            color: `color-mix(in srgb, ${color} 65%, transparent)`,
            fontFamily: 'var(--font-mono)',
            fontSize: '0.95em',
            lineHeight: 1,
            marginRight: '-0.1em',
          }}
        >
          ◀
        </span>
        <span
          style={{
            flex: 1,
            height: '2px',
            background: `color-mix(in srgb, ${color} 50%, transparent)`,
          }}
        />
      </div>
    </div>
  )
}

function KindBadge({ kind }: { kind: ActivityKind }) {
  const color = KIND_COLOR[kind]
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
      }}
    >
      {KIND_LABEL[kind]}
    </span>
  )
}

function ActivityCard({ activity }: { activity: NonNullable<Step['activity']> }) {
  const color = KIND_COLOR[activity.kind]
  return (
    <div
      style={{
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.1em',
        padding: '0.4em 0.8em',
        boxSizing: 'border-box',
        borderRadius: '0.45em',
        border: `1px solid color-mix(in srgb, ${color} 45%, transparent)`,
        background: `color-mix(in srgb, ${color} 7%, transparent)`,
        boxShadow: `0 0 14px color-mix(in srgb, ${color} 12%, transparent)`,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5em' }}>
        <KindBadge kind={activity.kind} />
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.78em',
            fontWeight: 600,
            color: 'var(--color-heading)',
          }}
        >
          {activity.name}
        </span>
      </div>
      <div
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.75em',
          color: 'color-mix(in srgb, var(--color-heading) 80%, transparent)',
        }}
      >
        {activity.detail}
      </div>
      <div
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.65em',
          color: 'color-mix(in srgb, var(--color-heading) 50%, transparent)',
        }}
      >
        {activity.meta}
      </div>
    </div>
  )
}

export default function WorkflowSequence() {
  const lastActivityIdx = STEPS.reduce(
    (last, s, i) => (s.activity ? i : last),
    -1,
  )
  const firstActivityIdx = STEPS.findIndex((s) => s.activity !== undefined)

  return (
    <div
      style={{
        position: 'relative',
        display: 'grid',
        gridTemplateColumns: `${STEP_WIDTH} ${ARROW_WIDTH} ${TEMPORAL_WIDTH} ${ARROW_WIDTH} ${ACTIVITY_WIDTH}`,
        margin: '0 auto',
        width: 'fit-content',
        fontSize: '0.78em',
        rowGap: ROW_GAP,
        columnGap: 0,
      }}
    >
      {/* Spine: continuous vertical line behind the workflow column */}
      <div
        aria-hidden
        style={{
          position: 'absolute',
          top: 0,
          bottom: 0,
          left: `calc(${STEP_WIDTH} / 2 - 1px)`,
          width: '2px',
          background: 'color-mix(in srgb, var(--color-heading) 30%, transparent)',
          zIndex: 0,
        }}
      />

      {/* Temporal box: spans first activity row to last activity row */}
      <div
        style={{
          gridColumn: 3,
          gridRow: `${firstActivityIdx + 1} / ${lastActivityIdx + 2}`,
          width: '100%',
          height: '100%',
          borderRadius: '0.6em',
          border: `1px solid color-mix(in srgb, ${TEMPORAL_COLOR} 60%, transparent)`,
          background: `color-mix(in srgb, ${TEMPORAL_COLOR} 6%, transparent)`,
          boxShadow: `0 0 22px color-mix(in srgb, ${TEMPORAL_COLOR} 14%, transparent)`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1,
        }}
      >
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '1.3em',
            fontWeight: 700,
            color: TEMPORAL_COLOR,
            letterSpacing: '0.08em',
          }}
        >
          Temporal
        </span>
      </div>

      {STEPS.map((step, i) => {
        const color = step.activity ? KIND_COLOR[step.activity.kind] : null
        return (
          <Fragment key={i}>
            {/* Col 1: step (stretches to fill row when active) */}
            <div
              style={{
                gridColumn: 1,
                gridRow: i + 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative',
                zIndex: 2,
              }}
            >
              <StepBox step={step} />
            </div>

            {step.activity && color && (
              <>
                {/* Col 2: outgoing arrows, vertically centered against the step */}
                <div
                  style={{
                    gridColumn: 2,
                    gridRow: i + 1,
                    display: 'flex',
                    alignItems: 'center',
                    paddingLeft: '0.3em',
                    paddingRight: '0.3em',
                    boxSizing: 'border-box',
                    zIndex: 2,
                  }}
                >
                  <BiArrow color={color} />
                </div>
                {/* Col 4: incoming arrows */}
                <div
                  style={{
                    gridColumn: 4,
                    gridRow: i + 1,
                    display: 'flex',
                    alignItems: 'center',
                    paddingLeft: '0.3em',
                    paddingRight: '0.3em',
                    boxSizing: 'border-box',
                    zIndex: 2,
                  }}
                >
                  <BiArrow color={color} />
                </div>
                {/* Col 5: activity card */}
                <div
                  style={{
                    gridColumn: 5,
                    gridRow: i + 1,
                    display: 'flex',
                    alignItems: 'center',
                    width: '100%',
                    zIndex: 2,
                  }}
                >
                  <ActivityCard activity={step.activity} />
                </div>
              </>
            )}
          </Fragment>
        )
      })}
    </div>
  )
}
