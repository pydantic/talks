/**
 * Progressive architecture diagram showing how agents get to production.
 *
 * Stages:
 *  - temporal: 3 large boxes (User / Agent / Temporal) stacked on the right half, large + centered
 *  - six:      adds LLM Provider (far right). Cost Controls & Optimization hang off the
 *              midpoint of the Agent<->LLM arrow (above and below).
 *  - eight:    adds AI Guardrails and Audit Logging hanging off the same midpoint,
 *              NOT pointing at the LLM provider.
 *  - logfire:  all arrows become dashed, central Logfire label appears.
 */

interface BoxProps {
  label: string
  subtitle: string
  x: number
  y: number
  w: number
  h: number
  highlight?: boolean
  accent?: boolean
}

function Box({ label, subtitle, x, y, w, h, highlight, accent }: BoxProps) {
  const stroke = accent ? 'var(--accent)' : highlight ? 'var(--accent)' : 'var(--accent-aqua)'
  const sw = highlight || accent ? 2.5 : 1.5
  const fSize = w > 180 ? 16 : 14
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={10} fill="#0d1f22" stroke={stroke} strokeWidth={sw} />
      <text x={x + w / 2} y={y + h / 2 - 7} textAnchor="middle" fontSize={fSize} fontWeight={700} fill={highlight ? 'var(--accent)' : 'var(--color-heading)'}>
        {label}
      </text>
      <text x={x + w / 2} y={y + h / 2 + 12} textAnchor="middle" fontSize={11.5} fill="var(--color-muted)">
        {subtitle}
      </text>
    </g>
  )
}

/** Single directed line with optional double-head */
function Line({ x1, y1, x2, y2, dashed, bidir }: { x1: number; y1: number; x2: number; y2: number; dashed?: boolean; bidir?: boolean }) {
  const dash = dashed ? '7 5' : undefined
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--accent)" strokeWidth={2} strokeDasharray={dash} markerEnd="url(#ah)" />
      {bidir && <line x1={x2} y1={y2} x2={x1} y2={y1} stroke="var(--accent)" strokeWidth={2} strokeDasharray={dash} markerEnd="url(#ah)" />}
    </g>
  )
}

export default function ArchitectureDiagram({ stage }: { stage: 'temporal' | 'six' | 'eight' | 'logfire' }) {
  const showLLM   = stage === 'six' || stage === 'eight' || stage === 'logfire'
  const showSix   = stage === 'six' || stage === 'eight' || stage === 'logfire'
  const showEight = stage === 'eight' || stage === 'logfire'
  const showFire  = stage === 'logfire'

  const W = 1100
  const H = 520

  // === TEMPORAL stage: 3 large boxes, RIGHT-aligned and large ===
  // For later stages, we compact the left stack to make room for more
  const stackX   = showLLM ? 90  : 280   // center-x of the left stack
  const bw       = showLLM ? 210 : 270   // box width
  const bh       = showLLM ? 70  : 90    // box height
  const vGap     = showLLM ? 85  : 100   // vertical gap between box centers
  const startY   = showLLM ? 80  : 60    // y center of top box

  const userCx   = stackX
  const agentCx  = stackX
  const tempCx   = stackX

  const userY    = startY
  const agentY   = startY + vGap
  const tempY    = startY + vGap * 2

  // LLM Provider
  const llmX   = 720
  const llmW   = 260
  const llmH   = 75
  const llmCx  = llmX + llmW / 2
  const llmCy  = agentY + bh / 2  // same vertical center as agent

  // Midpoint of the Agent<->LLM horizontal connector
  const midX   = (agentCx + bw / 2 + llmX) / 2
  const midY   = llmCy

  // Boxes hanging off the midpoint
  const hangW  = 195
  const hangH  = 60

  // Six-step: Cost above, Optimization below
  const costX  = midX - hangW / 2
  const costY  = midY - 105 - hangH
  const optimX = midX - hangW / 2
  const optimY = midY + 105

  // Eight-step: Guardrails above (between cost and LLM), Audit below
  const guardX = midX + 120 - hangW / 2
  const guardY = costY
  const auditX = midX + 120 - hangW / 2
  const auditY = optimY

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', maxHeight: '70vh', marginTop: '0.4rem' }}>
      <defs>
        <marker id="ah" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
          <polygon points="0 0, 9 3.5, 0 7" fill="var(--accent)" />
        </marker>
      </defs>

      {/* App group outline when external boxes are shown */}
      {showLLM && (
        <>
          <rect x={20} y={30} width={bw + 80} height={vGap * 2 + bh + 60} rx={14}
            fill="rgba(251,255,234,0.025)" stroke="rgba(255,255,255,0.07)" strokeWidth={1} />
          <text x={stackX + 5} y={vGap * 2 + bh + 110} textAnchor="middle" fontSize={13} fill="var(--color-muted)">Your App</text>
        </>
      )}

      {/* Core 3 boxes */}
      <Box label="User" subtitle="Submits tasks, expects results"
        x={userCx - bw / 2} y={userY} w={bw} h={bh} />
      <Box label="Your Agent" subtitle="Tool loop - code - SQL"
        x={agentCx - bw / 2} y={agentY} w={bw} h={bh} highlight />
      <Box label="Temporal" subtitle="Durable Execution"
        x={tempCx - bw / 2} y={tempY} w={bw} h={bh} />

      {/* Core vertical arrows */}
      <Line x1={userCx}  y1={userY + bh}  x2={agentCx} y2={agentY}      bidir dashed={showFire} />
      <Line x1={agentCx} y1={agentY + bh} x2={tempCx}  y2={tempY}       bidir dashed={showFire} />

      {/* LLM Provider */}
      {showLLM && (
        <>
          <Box label="LLM Provider" subtitle="Anthropic - OpenAI - AWS - Azure - Google"
            x={llmX} y={agentY} w={llmW} h={llmH} accent />
          <Line x1={agentCx + bw / 2} y1={llmCy} x2={llmX} y2={llmCy} bidir dashed={showFire} />
        </>
      )}

      {/* Six-step: Cost Controls and Optimization hang off the midpoint */}
      {showSix && (
        <>
          {/* vertical stem from midpoint up to cost */}
          <Line x1={midX} y1={midY - 8} x2={midX} y2={costY + hangH} dashed={showFire} />
          <Box label="Cost Controls" subtitle="Monitor and Enforce spend limits"
            x={costX} y={costY} w={hangW} h={hangH} />

          {/* vertical stem from midpoint down to optimization */}
          <Line x1={midX} y1={midY + 8} x2={midX} y2={optimY} dashed={showFire} />
          <Box label="Optimization" subtitle="Evals - datasets - tuning"
            x={optimX} y={optimY} w={hangW} h={hangH} />
        </>
      )}

      {/* Eight-step: Guardrails and Audit also hang off the connection (shifted right) */}
      {showEight && (
        <>
          <Line x1={guardX + hangW / 2} y1={midY - 8} x2={guardX + hangW / 2} y2={guardY + hangH} dashed={showFire} />
          <Box label="AI Guardrails" subtitle="Detect key and data exfiltration"
            x={guardX} y={guardY} w={hangW} h={hangH} />

          <Line x1={auditX + hangW / 2} y1={midY + 8} x2={auditX + hangW / 2} y2={auditY} dashed={showFire} />
          <Box label="Audit Logging" subtitle="Full conversations & tool calls"
            x={auditX} y={auditY} w={hangW} h={hangH} />
        </>
      )}

      {/* Logfire label */}
      {showFire && (
        <text x={midX + 100} y={midY + 5} textAnchor="middle" fontSize={21} fontWeight={700} fill="var(--accent)">
          Pydantic Logfire
        </text>
      )}
    </svg>
  )
}
