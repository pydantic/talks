const wrap: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: '0.78rem',
  lineHeight: 1.55,
  background: 'var(--surface)',
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: '10px',
  padding: '1rem 1.2rem',
  color: 'var(--color-text)',
  whiteSpace: 'pre',
  overflowX: 'auto',
};

const kw = { color: 'var(--accent)' };
const cls = { color: 'var(--accent-aqua)' };
const str = { color: 'var(--accent-secondary)' };
const com = { color: 'var(--color-muted)', fontStyle: 'italic' as const };
const fn = { color: 'var(--accent-tertiary)' };

export default function HostLoopCode() {
  return (
    <div style={wrap}>
      <span style={kw}>class</span> <span style={cls}>MontyRun</span>:
      {'\n'}    <span style={kw}>def</span> <span style={fn}>start</span>(self) -&gt; RunProgress: ...
      {'\n'}    <span style={kw}>def</span> <span style={fn}>resume</span>(self, result) -&gt; RunProgress: ...
      {'\n'}
      {'\n'}<span style={com}># host loop</span>
      {'\n'}run = <span style={cls}>MontyRun</span>(code)
      {'\n'}progress = run.<span style={fn}>start</span>()
      {'\n'}
      {'\n'}<span style={kw}>while</span> <span style={kw}>isinstance</span>(progress, <span style={cls}>ExternalCall</span>):
      {'\n'}    <span style={com}># e.g. ExternalCall("get_who", ("Hello",))</span>
      {'\n'}    result = <span style={fn}>call_tool</span>(progress.name, progress.args)
      {'\n'}    progress = run.<span style={fn}>resume</span>(result)
      {'\n'}
      {'\n'}<span style={com}># progress == Return("Hello World")</span>
    </div>
  );
}
