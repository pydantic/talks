type Op = {
  ip: string;
  op: string;
  arg: string;
  note?: string;
};

const OPS: Op[] = [
  { ip: '0', op: 'LoadConst', arg: "'Hello'" },
  { ip: '1', op: 'StoreLocal', arg: 'greeting' },
  { ip: '2', op: 'LoadName', arg: 'get_who', note: 'host' },
  { ip: '3', op: 'LoadLocal', arg: 'greeting' },
  { ip: '4', op: 'Call', arg: '1' },
  { ip: '5', op: 'StoreLocal', arg: 'who' },
  { ip: '6', op: 'LoadLocal', arg: 'greeting' },
  { ip: '7', op: 'LoadLocal', arg: 'who' },
  { ip: '8', op: 'FormatStr', arg: '"{} {}"' },
  { ip: '9', op: 'Return', arg: 'None' },
];

const wrap: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontSize: '0.85rem',
  background: 'var(--surface)',
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: '10px',
  padding: '0.8rem 1rem',
};

const head: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '2.2rem 6.5rem 1fr auto',
  gap: '0.8rem',
  fontSize: '0.62rem',
  letterSpacing: '0.14em',
  textTransform: 'uppercase',
  color: 'var(--color-muted)',
  paddingBottom: '0.4rem',
  borderBottom: '1px solid rgba(255,255,255,0.08)',
  marginBottom: '0.3rem',
};

const row: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '2.2rem 6.5rem 1fr auto',
  gap: '0.8rem',
  padding: '0.18rem 0',
  alignItems: 'baseline',
};

const note: React.CSSProperties = {
  fontSize: '0.65rem',
  color: 'var(--accent)',
  letterSpacing: '0.05em',
};

export default function BytecodeExample() {
  return (
    <div style={wrap}>
      <div style={head}>
        <div>ip</div>
        <div>op</div>
        <div>arg</div>
        <div></div>
      </div>
      {OPS.map((o) => (
        <div key={o.ip} style={row}>
          <div style={{ color: 'var(--color-muted)' }}>{o.ip}</div>
          <div style={{ color: 'var(--accent)' }}>{o.op}</div>
          <div style={{ color: 'var(--accent-aqua)' }}>{o.arg}</div>
          <div style={note}>{o.note ?? ''}</div>
        </div>
      ))}
    </div>
  );
}
