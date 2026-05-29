/**
 * Mock UI showing the "Why this proposal?" explanation card
 * from the Logfire optimization UI (page 16 of the PDF).
 */

export default function ProposalExplanation() {
  return (
    <div
      style={{
        background: '#1a1a1a',
        border: '1px solid #333',
        borderRadius: 12,
        padding: '1.2rem 1.5rem',
        fontSize: '0.78rem',
        lineHeight: 1.6,
        maxHeight: '85%',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.9rem' }}>&#8963;</span>
          <span style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--color-heading)' }}>
            Why this proposal?
          </span>
        </div>
        <span
          style={{
            border: '1px solid #666',
            borderRadius: 4,
            padding: '0.15rem 0.5rem',
            fontSize: '0.7rem',
            color: '#ccc',
          }}
        >
          Medium Confidence
        </span>
      </div>

      {/* What's going wrong */}
      <div style={{ marginBottom: '0.8rem' }}>
        <p
          style={{
            fontSize: '0.65rem',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            color: '#888',
            marginBottom: '0.3rem',
          }}
        >
          WHAT'S GOING WRONG
        </p>
        <p style={{ color: 'var(--color-text)', margin: 0 }}>
          Missing constraint on a rare case: The model over-specifies its guess, leading to incorrect guesses, even
          when a more general, correct answer is strongly indicated by the clues. It focuses too much on 'narrowing
          down' to the most specific item, rather than guessing when a sufficiently specific (but potentially broader)
          correct answer is clear.
        </p>
      </div>

      {/* Reasoning */}
      <div style={{ marginBottom: '0.8rem' }}>
        <p
          style={{
            fontSize: '0.65rem',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            color: '#888',
            marginBottom: '0.3rem',
          }}
        >
          REASONING
        </p>
        <p style={{ color: 'var(--color-text)', margin: 0 }}>
          The current prompt's instruction to 'narrow down the possibilities efficiently' combined with 'Call
          make_guess once you are confident in your choice' can lead the model to pursue overly specific details,
          resulting in incorrect specific guesses even when a correct broader category is clear. The proposed change
          adds explicit guidance to make a broader, correct guess in such scenarios.
        </p>
      </div>

      {/* Claims */}
      <div>
        <p
          style={{
            fontSize: '0.65rem',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            color: '#888',
            marginBottom: '0.3rem',
          }}
        >
          CLAIMS AND HOW THE PROPOSED VALUE ADDRESSES THEM
        </p>
      </div>
    </div>
  )
}
