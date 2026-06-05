const stack: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '0.7rem',
};

const box: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: '0.82rem',
  lineHeight: 1.55,
  background: 'var(--surface)',
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: '10px',
  padding: '0.9rem 1.1rem',
  color: 'var(--color-text)',
  whiteSpace: 'pre',
  overflowX: 'auto',
};

const k: React.CSSProperties = { color: 'var(--accent-aqua)' };
const t: React.CSSProperties = { color: 'var(--accent)' };
const s: React.CSSProperties = { color: 'var(--accent-secondary)' };
const fn: React.CSSProperties = { color: 'var(--accent-tertiary)' };

export default function AstExample() {
  return (
    <div style={stack}>
      <div style={box}>
        greeting = <span style={s}>'Hello'</span>
        {'\n'}who = <span style={fn}>get_who</span>(greeting)
        {'\n'}<span style={s}>f'{'{'}greeting{'}'} {'{'}who{'}'}'</span>
      </div>

      <div style={box}>
        <span style={t}>Module</span> {'{'}
        {'\n'}  <span style={k}>body</span>: [
        {'\n'}    <span style={t}>Assign</span> {'{'} <span style={k}>target</span>: <span style={t}>Name</span>(<span style={s}>"greeting"</span>), <span style={k}>value</span>: <span style={t}>Constant</span>(<span style={s}>"Hello"</span>) {'}'},
        {'\n'}    <span style={t}>Assign</span> {'{'}
        {'\n'}      <span style={k}>target</span>: <span style={t}>Name</span>(<span style={s}>"who"</span>),
        {'\n'}      <span style={k}>value</span>: <span style={t}>Call</span> {'{'}
        {'\n'}        <span style={k}>func</span>: <span style={t}>Name</span>(<span style={s}>"get_who"</span>),
        {'\n'}        <span style={k}>args</span>: [<span style={t}>Name</span>(<span style={s}>"greeting"</span>)]
        {'\n'}      {'}'}
        {'\n'}    {'}'},
        {'\n'}    <span style={t}>Expr</span>(<span style={t}>FString</span> {'{'}
        {'\n'}      <span style={k}>parts</span>: [<span style={t}>Name</span>(<span style={s}>"greeting"</span>), <span style={s}>" "</span>, <span style={t}>Name</span>(<span style={s}>"who"</span>)]
        {'\n'}    {'}'})
        {'\n'}  ]
        {'\n'}{'}'}
      </div>
    </div>
  );
}
