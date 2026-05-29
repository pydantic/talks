/**
 * Displays a prompt optimization round with the prompt text (highlighted additions)
 * and results in a clear, large format.
 */

interface Result {
  item: string
  outcome: string
  note?: string
}

interface PromptRoundProps {
  round: number
  prompt: string
  highlights?: string[]
  results: Result[]
}

export default function PromptRound({ round, prompt, highlights = [], results }: PromptRoundProps) {
  function renderPrompt(text: string) {
    if (highlights.length === 0) return <span>{text}</span>

    const parts: Array<{ text: string; highlighted: boolean }> = []
    let remaining = text

    for (const h of highlights) {
      const idx = remaining.indexOf(h)
      if (idx >= 0) {
        if (idx > 0) parts.push({ text: remaining.slice(0, idx), highlighted: false })
        parts.push({ text: h, highlighted: true })
        remaining = remaining.slice(idx + h.length)
      }
    }
    if (remaining) parts.push({ text: remaining, highlighted: false })
    if (parts.length === 0) return <span>{text}</span>

    return (
      <>
        {parts.map((p, i) =>
          p.highlighted ? (
            <span key={i} style={{ background: '#4a9e4a', color: 'white', padding: '0.1em 0.2em', borderRadius: 3 }}>
              {p.text}
            </span>
          ) : (
            <span key={i}>{p.text}</span>
          ),
        )}
      </>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '1rem' }}>
      {/* Header */}
      <h2 style={{ margin: 0, fontSize: '1.4rem', color: 'var(--color-heading)' }}>
        {round === 0 ? 'Starting Prompt:' : `Optimized Prompt (Round ${round}):`}
      </h2>

      {/* Prompt text */}
      <div
        style={{
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid rgba(255,255,255,0.1)',
          padding: '1rem 1.2rem',
          borderRadius: 8,
          fontSize: '1rem',
          lineHeight: 1.7,
          color: 'var(--color-text)',
          flex: '0 0 auto',
        }}
      >
        {renderPrompt(prompt)}
      </div>

      {/* Results */}
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginTop: '0.5rem' }}>
        {results.map((r, i) => {
          const passed = r.outcome.startsWith('Passed')
          return (
            <div
              key={i}
              style={{
                flex: '1 1 200px',
                background: passed ? 'rgba(77,255,216,0.06)' : 'rgba(255,101,80,0.06)',
                border: `1.5px solid ${passed ? 'var(--accent-aqua)' : 'var(--accent-secondary)'}`,
                borderRadius: 8,
                padding: '0.7rem 1rem',
              }}
            >
              <div style={{ fontWeight: 700, fontSize: '1.1rem', color: 'var(--color-heading)', marginBottom: '0.2rem' }}>
                {r.item}
              </div>
              <div style={{ fontSize: '1rem', color: passed ? 'var(--accent-aqua)' : 'var(--accent-secondary)', fontWeight: 600 }}>
                {r.outcome}
              </div>
              {r.note && (
                <div style={{ fontSize: '0.85rem', color: 'var(--color-muted)', marginTop: '0.15rem' }}>
                  {r.note}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
