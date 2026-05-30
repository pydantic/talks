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
  subtitle2?: string
  x: number
  y: number
  w: number
  h: number
  highlight?: boolean
  accent?: boolean
  large?: boolean
  medium?: boolean
}

function Box({ label, subtitle, subtitle2, x, y, w, h, highlight, accent, large, medium }: BoxProps) {
  const stroke = accent ? 'var(--accent)' : highlight ? 'var(--accent)' : 'var(--accent-aqua)'
  const sw = highlight || accent ? 2.5 : 1.5
  const fSize = large ? 26 : medium ? 28 : w > 180 ? 16 : 14
  const subSize = large ? 16 : medium ? 19 : 11.5
  const textColor = highlight ? 'var(--accent)' : 'var(--color-heading)'
  const subColor = highlight ? 'var(--accent)' : 'rgba(255,255,255,0.65)'
  const hasTwo = !!subtitle2
  const labelOff = hasTwo ? -14 : (large ? -10 : -7)
  const sub1Off = hasTwo ? 9 : (large ? 16 : 12)
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={10} fill="#0d1f22" stroke={stroke} strokeWidth={sw} />
      <text x={x + w / 2} y={y + h / 2 + labelOff} textAnchor="middle" fontSize={fSize} fontWeight={700} fill={textColor}>
        {label}
      </text>
      <text x={x + w / 2} y={y + h / 2 + sub1Off} textAnchor="middle" fontSize={subSize} fill={subColor}>
        {subtitle}
      </text>
      {subtitle2 && (
        <text x={x + w / 2} y={y + h / 2 + sub1Off + 22} textAnchor="middle" fontSize={subSize} fill={subColor}>
          {subtitle2}
        </text>
      )}
    </g>
  )
}

/** Single directed line with optional double-head */
function Line({ x1, y1, x2, y2, dashed, bidir, green }: { x1: number; y1: number; x2: number; y2: number; dashed?: boolean; bidir?: boolean; green?: boolean }) {
  const stroke = green ? 'var(--accent-aqua)' : 'var(--accent)'
  const marker = green ? 'url(#ah-green)' : 'url(#ah)'
  const dash = dashed ? '7 5' : undefined
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={stroke} strokeWidth={2} strokeDasharray={dash} markerEnd={marker} />
      {bidir && <line x1={x2} y1={y2} x2={x1} y2={y1} stroke={stroke} strokeWidth={2} strokeDasharray={dash} markerEnd={marker} />}
    </g>
  )
}

export default function ArchitectureDiagram({ stage }: { stage: 'temporal' | 'six' | 'eight' | 'logfire' }) {
  const showLLM   = stage === 'six' || stage === 'eight' || stage === 'logfire'
  const showSix   = stage === 'six' || stage === 'eight' || stage === 'logfire'
  const showEight = stage === 'eight' || stage === 'logfire'
  const showFire  = stage === 'logfire'

  // Temporal uses a tall/narrow viewBox so the 3 boxes dominate the space.
  // Other stages use a wide viewBox to fit LLM + hanging boxes.
  const W = showLLM ? 1420 : 560
  const H = showLLM ? 640 : 500

  // Left stack of 3 boxes. For showLLM stages they sit inside the "Your App" container.
  // Kept deliberately compact so hanging boxes dominate visually.
  const stackX   = showLLM ? 108 : 280   // center-x of the stack (≥98 so container stays in viewBox)
  const bw       = showLLM ? 145 : 500   // box width — narrow on multi-box slides
  const bh       = showLLM ? 58  : 120   // box height
  const vGap     = showLLM ? 108 : 170   // gap from top of one box to top of next
  const startY   = showLLM ? 140 : 15    // y of top box

  const userCx   = stackX
  const agentCx  = stackX
  const tempCx   = stackX

  const userY    = startY
  const agentY   = startY + vGap
  const tempY    = startY + vGap * 2

  // LLM Provider — pushed right to give hanging boxes more room
  const llmX   = 1100
  const llmW   = 290
  const llmH   = 120
  const llmCx  = llmX + llmW / 2
  const llmBoxY = agentY + bh / 2 - llmH / 2   // top of LLM box, centered on Your Agent
  const llmCy   = agentY + bh / 2               // shared connector / hub Y

  // Midpoint of the Agent<->LLM horizontal connector
  const agentRight = stackX + bw / 2
  const midX   = (agentRight + llmX) / 2
  const midY   = llmCy

  // Hanging boxes — larger than the stack so they're easy to read
  const hangW  = 300
  const hangH  = 138

  // Compute branch x-positions so hanging boxes never overlap "Your App" or each other.
  // yourAppRight = right edge of the "Your App" container rect
  const yourAppRight = stackX + bw / 2 + 25
  const hangMargin   = 65   // gap between container edges and box edges
  const leftBranchX  = yourAppRight + hangMargin + hangW / 2
  const rightBranchX = llmX - hangMargin - hangW / 2

  const sixBranchX   = showEight ? leftBranchX : midX

  // Six-step boxes (cost/optim): centred at sixBranchX
  const vertGap = showFire ? 125 : 105
  const costX  = sixBranchX - hangW / 2
  const costY  = midY - vertGap - hangH
  const optimX = sixBranchX - hangW / 2
  const optimY = midY + vertGap

  // Eight-step boxes (guard/audit): centred at rightBranchX
  const guardX = rightBranchX - hangW / 2
  const guardY = costY
  const auditX = rightBranchX - hangW / 2
  const auditY = optimY

  // Logfire hub box (logfire stage only) — sits inline between Agent and LLM
  const logfireW = 270
  const logfireH = 100
  const logfireX = midX - logfireW / 2
  const logfireY = midY - logfireH / 2
  // Four connection points on logfire box perimeter for the diagonal spokes
  const lfTL = { x: midX - 70, y: midY - logfireH / 2 }  // top-left
  const lfTR = { x: midX + 70, y: midY - logfireH / 2 }  // top-right
  const lfBL = { x: midX - 70, y: midY + logfireH / 2 }  // bottom-left
  const lfBR = { x: midX + 70, y: midY + logfireH / 2 }  // bottom-right

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', maxHeight: '70vh', marginTop: '0.4rem' }}>
      <defs>
        <marker id="ah" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
          <polygon points="0 0, 9 3.5, 0 7" fill="var(--accent)" />
        </marker>
        <marker id="ah-green" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">
          <polygon points="0 0, 9 3.5, 0 7" fill="var(--accent-aqua)" />
        </marker>
      </defs>

      {/* "Your App" container — sized to tightly wrap the 3 stack boxes */}
      {showLLM && (
        <>
          <rect x={stackX - bw / 2 - 25} y={startY - 25} width={bw + 50} height={vGap * 2 + bh + 50} rx={14}
            fill="rgba(251,255,234,0.025)" stroke="rgba(255,255,255,0.12)" strokeWidth={1.5} />
          <text x={stackX} y={startY + vGap * 2 + bh + 42} textAnchor="middle" fontSize={14} fill="var(--color-muted)">Your App</text>
        </>
      )}

      {/* Core 3 boxes */}
      <Box label="User"
        subtitle={showLLM ? 'Submits tasks' : 'Submits tasks, expects results'}
        x={userCx - bw / 2} y={userY} w={bw} h={bh} large={!showLLM} />
      <Box label="Your Agent"
        subtitle={showLLM ? 'Tool calls & SQL' : 'Tool loop - code - SQL'}
        x={agentCx - bw / 2} y={agentY} w={bw} h={bh} large={!showLLM} />
      <Box label="Temporal" subtitle="Durable Execution"
        x={tempCx - bw / 2} y={tempY} w={bw} h={bh} large={!showLLM} />

      {/* Core vertical arrows */}
      <Line x1={userCx}  y1={userY + bh}  x2={agentCx} y2={agentY}      bidir dashed={showFire} green />
      <Line x1={agentCx} y1={agentY + bh} x2={tempCx}  y2={tempY}       bidir dashed={showFire} green />

      {/* LLM Provider — always shown; connector is direct (non-fire) or split via Logfire box */}
      {showLLM && (
        <Box label="LLM Provider" subtitle="Anthropic · OpenAI · Bedrock"
          subtitle2="Azure · Google · Vertex"
          x={llmX} y={llmBoxY} w={llmW} h={llmH} medium accent={showFire} />
      )}
      {showLLM && !showFire && (
        <Line x1={agentCx + bw / 2} y1={llmCy} x2={llmX} y2={llmCy} bidir green />
      )}

      {/* Six-step: Cost Controls and Optimization */}
      {showSix && (
        <>
          {!showFire
            ? <Line x1={sixBranchX} y1={midY - 8} x2={sixBranchX} y2={costY + hangH} green />
            : <Line x1={lfTL.x} y1={lfTL.y} x2={sixBranchX} y2={costY + hangH} dashed />}
          <Box label="Cost Controls" subtitle="Agents running for hours" subtitle2="cost 100s of times more"
            x={costX} y={costY} w={hangW} h={hangH} medium accent={showFire} />

          {!showFire
            ? <Line x1={sixBranchX} y1={midY + 8} x2={sixBranchX} y2={optimY} green />
            : <Line x1={lfBL.x} y1={lfBL.y} x2={sixBranchX} y2={optimY} dashed />}
          <Box label="Optimization" subtitle="30% faster runtime" subtitle2="literally saves hours"
            x={optimX} y={optimY} w={hangW} h={hangH} medium accent={showFire} />
        </>
      )}

      {/* Eight-step: AI Guardrails and Audit Logging */}
      {showEight && (
        <>
          {!showFire
            ? <Line x1={rightBranchX} y1={midY - 8} x2={rightBranchX} y2={guardY + hangH} green />
            : <Line x1={lfTR.x} y1={lfTR.y} x2={rightBranchX} y2={guardY + hangH} dashed />}
          <Box label="AI Guardrails" subtitle="Detect key & data exfiltration"
            subtitle2="and prompt injection"
            x={guardX} y={guardY} w={hangW} h={hangH} medium accent={showFire} />

          {!showFire
            ? <Line x1={rightBranchX} y1={midY + 8} x2={rightBranchX} y2={auditY} green />
            : <Line x1={lfBR.x} y1={lfBR.y} x2={rightBranchX} y2={auditY} dashed />}
          <Box label="Audit Logging" subtitle="Full conversations & tool calls"
            subtitle2="for debugging & compliance"
            x={auditX} y={auditY} w={hangW} h={hangH} medium accent={showFire} />
        </>
      )}

      {/* Logfire stage: inline hub box + agent↔logfire↔LLM connectors + spokes to all 4 boxes */}
      {showFire && (
        <>
          {/* Agent → Logfire and Logfire → LLM (split connector) */}
          <Line x1={agentCx + bw / 2} y1={llmCy} x2={logfireX} y2={llmCy} bidir dashed />
          <Line x1={logfireX + logfireW} y1={llmCy} x2={llmX} y2={llmCy} bidir dashed />

          {/* Logfire branded box */}
          <rect x={logfireX} y={logfireY} width={logfireW} height={logfireH} rx={10}
            fill="#0d1f22" stroke="var(--accent)" strokeWidth={2.5} />
          {/* Diamond mark: scale from viewBox 0 0 139 120 to ~52px wide */}
          <g transform={`translate(${midX - 84}, ${logfireY + logfireH / 2 - 22}) scale(${52 / 139})`}>
            <path fill="var(--accent)" d="M137.542 90.563 73.808 2.241c-2.006-2.757-6.632-2.757-8.617 0L1.456 90.563a5.318 5.318 0 0 0-.998 3.101 5.331 5.331 0 0 0 3.642 5.05l63.735 20.851h.01a5.31 5.31 0 0 0 3.293 0h.01l63.735-20.85a5.265 5.265 0 0 0 3.393-3.406 5.244 5.244 0 0 0-.749-4.746h.015Zm-68.04-76.151 25.545 35.403-23.889-7.813c-.184-.06-.38-.05-.564-.094a3.488 3.488 0 0 0-.549-.09c-.184-.025-.359-.095-.543-.095-.185 0-.355.07-.54.095-.184.02-.368.05-.548.09-.19.035-.384.035-.554.094L44.115 49.77l-.15.05L69.513 14.41h-.01ZM33.408 64.438l27.811-9.104 2.969-.967v52.838L14.324 90.887l19.084-26.449Zm41.412 42.757V54.367l30.78 10.071 19.085 26.434-49.87 16.323h.005Z" />
          </g>
          <text x={midX - 22} y={logfireY + logfireH / 2 - 6}
            fontSize={24} fontWeight={700} fill="var(--accent)">Pydantic</text>
          <text x={midX - 22} y={logfireY + logfireH / 2 + 22}
            fontSize={24} fontWeight={700} fill="var(--accent)">Logfire</text>
        </>
      )}
    </svg>
  )
}
