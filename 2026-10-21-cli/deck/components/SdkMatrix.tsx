type Cell = true | false | string;

interface Row {
  label: string;
  cells: [Cell, Cell, Cell];
}

const langs = [
  { name: 'Python', pkg: 'logfire-sdk 6' },
  { name: 'JS / TS', pkg: 'logfire 0.9' },
  { name: 'Rust', pkg: 'logfire 0.12' },
];

const rows: Row[] = [
  { label: 'Built on', cells: ['opentelemetry-sdk', '@opentelemetry/sdk-node', 'tracing + opentelemetry'] },
  { label: 'Auto-instrumentation', cells: ['34 instrument_* (OTel contrib)', 'node auto-instrumentations', 'anything using tracing'] },
  { label: 'logfire.msg + msg_template', cells: [true, true, 'msg; template = span name'] },
  { label: 'logfire.json_schema', cells: [true, true, true] },
  { label: 'Pending spans', cells: [true, false, true] },
  { label: 'Prompts / variables', cells: [true, 'OpenFeature / OFREP', 'OpenFeature / OFREP'] },
];

const table: React.CSSProperties = {
  width: '100%',
  borderCollapse: 'collapse',
  fontSize: '1rem',
  marginTop: '0.4rem',
};

const th: React.CSSProperties = {
  textAlign: 'left',
  padding: '0.35rem 0.7rem',
  borderBottom: '2px solid rgba(255,255,255,0.15)',
  color: 'var(--color-heading)',
  fontWeight: 700,
};

const pkgStyle: React.CSSProperties = {
  display: 'block',
  fontFamily: 'var(--font-mono)',
  fontSize: '0.8em',
  color: 'var(--color-muted)',
  fontWeight: 400,
};

const td: React.CSSProperties = {
  padding: '0.35rem 0.7rem',
  borderBottom: '1px solid rgba(255,255,255,0.08)',
  verticalAlign: 'middle',
};

const labelStyle: React.CSSProperties = { ...td, color: 'var(--color-heading)', fontWeight: 600, whiteSpace: 'nowrap' };

function render(c: Cell) {
  if (c === true) return <span style={{ color: 'var(--accent-aqua)', fontSize: '1.3em' }}>✓</span>;
  if (c === false) return <span style={{ color: 'var(--color-muted)', fontSize: '1.3em' }}>✗</span>;
  return <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85em' }}>{c}</span>;
}

export default function SdkMatrix() {
  return (
    <table style={table}>
      <thead>
        <tr>
          <th style={th}></th>
          {langs.map((l) => (
            <th key={l.name} style={th}>
              {l.name}
              <span style={pkgStyle}>{l.pkg}</span>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.label}>
            <td style={labelStyle}>{r.label}</td>
            {r.cells.map((c, i) => (
              <td key={i} style={td}>
                {render(c)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
