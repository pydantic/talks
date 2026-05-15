const wrap: React.CSSProperties = {
  fontFamily: 'var(--font-mono)',
  background: 'var(--surface)',
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: '10px',
  padding: '1rem',
};

const HEAP_FILL = 'rgba(74, 215, 197, 0.10)';
const HEAP_STROKE = 'var(--accent-aqua)';
const HEAP_BG = 'rgba(74, 215, 197, 0.04)';
const STACK_FILL = 'rgba(229, 32, 168, 0.12)';
const STACK_STROKE = 'var(--accent)';
const STACK_BG = 'rgba(229, 32, 168, 0.04)';
const BC_BG = 'rgba(179, 136, 255, 0.04)';
const BC_STROKE_SOFT = 'rgba(179, 136, 255, 0.20)';
const BC_ACCENT = 'var(--accent-tertiary)';
const MUTED = 'var(--color-muted)';
const HEAD = 'var(--color-heading)';
const ARROW = 'var(--accent-aqua)';

function StackCell({
  x,
  y,
  w,
  h,
  type,
  value,
  hint,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  type: string;
  value: string;
  hint?: string;
}) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={6}
        fill={STACK_FILL}
        stroke={STACK_STROKE}
        strokeWidth={1.2}
      />
      <text x={x + 12} y={y + 16} fontSize={10} fill={MUTED} fontFamily="var(--font-mono)">
        {type}
      </text>
      <text x={x + 12} y={y + 36} fontSize={15} fill={HEAD} fontFamily="var(--font-mono)">
        {value}
      </text>
      {hint && (
        <text
          x={x + w - 10}
          y={y + 16}
          fontSize={9}
          fill={MUTED}
          textAnchor="end"
          fontFamily="var(--font-mono)"
        >
          {hint}
        </text>
      )}
    </g>
  );
}

function HeapEntry({
  x,
  y,
  w,
  h,
  type,
  value,
  rc,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  type: string;
  value: string;
  rc: string;
}) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={6}
        fill={HEAP_FILL}
        stroke={HEAP_STROKE}
        strokeWidth={1.2}
      />
      <text x={x + 12} y={y + 18} fontSize={10} fill={MUTED} fontFamily="var(--font-mono)">{type}</text>
      <text x={x + 56} y={y + 18} fontSize={10} fill={STACK_STROKE} fontWeight={700} fontFamily="var(--font-mono)">rc {rc}</text>
      <text x={x + 12} y={y + 38} fontSize={16} fill={HEAD} fontFamily="var(--font-mono)">{value}</text>
    </g>
  );
}

function BytecodeRow({
  x,
  y,
  op,
  arg,
  current,
  faded,
}: {
  x: number;
  y: number;
  op: string;
  arg?: string;
  current?: boolean;
  faded?: boolean;
}) {
  const opacity = faded ? 0.35 : 1;
  return (
    <g opacity={opacity}>
      <text
        x={x}
        y={y}
        fontSize={11}
        fill={current ? BC_ACCENT : 'transparent'}
        fontFamily="var(--font-mono)"
      >
        ▶
      </text>
      <text
        x={x + 14}
        y={y}
        fontSize={12}
        fill={current ? BC_ACCENT : HEAD}
        fontFamily="var(--font-mono)"
        fontWeight={current ? 700 : 400}
      >
        {op}
      </text>
      {arg && (
        <text
          x={x + 80}
          y={y}
          fontSize={12}
          fill={current ? BC_ACCENT : MUTED}
          fontFamily="var(--font-mono)"
        >
          {arg}
        </text>
      )}
    </g>
  );
}

export default function HeapVmDiagram() {
  // Three columns: BYTECODE (iterate over list[int]) | STACK | HEAP
  // Snapshot from a different snippet `(1 + 1) * 2`, IP at `Mul`:
  //   stack = [Int 2 (top, the const), Int 2 (the Add result)]
  //   heap  = [Int 2 rc=2 (interned, two stack refs!), Int 1 rc=1, ...]
  // The shared Int 2 cell + rc=2 visualises both refcounting AND interning.
  return (
    <div style={wrap}>
      <svg viewBox="0 0 640 340" style={{ width: '100%', height: 'auto', display: 'block' }}>
        <defs>
          <marker
            id="arrow-aqua-3"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="8"
            markerHeight="8"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill={ARROW} />
          </marker>
        </defs>

        {/* BYTECODE panel */}
        <rect
          x={10}
          y={10}
          width={170}
          height={320}
          rx={10}
          fill={BC_BG}
          stroke={BC_STROKE_SOFT}
          strokeDasharray="3 4"
        />
        <text x={22} y={32} fontSize={11} fill={MUTED} fontFamily="var(--font-mono)" letterSpacing={2}>
          BYTECODE
        </text>
        <text x={22} y={48} fontSize={10} fill={MUTED} fontFamily="var(--font-mono)">
          (1 + 1) * 2
        </text>
        <BytecodeRow x={22} y={94} op="LoadConst" arg="1" faded />
        <BytecodeRow x={22} y={120} op="LoadConst" arg="1" faded />
        <BytecodeRow x={22} y={146} op="Add" faded />
        <BytecodeRow x={22} y={172} op="LoadConst" arg="2" faded />
        <BytecodeRow x={22} y={198} op="Mul" current />
        <BytecodeRow x={22} y={224} op="StoreLocal" arg="x" />
        <BytecodeRow x={22} y={250} op="..." />

        {/* STACK panel */}
        <rect
          x={200}
          y={10}
          width={160}
          height={320}
          rx={10}
          fill={STACK_BG}
          stroke="rgba(229, 32, 168, 0.18)"
          strokeDasharray="3 4"
        />
        <text x={212} y={32} fontSize={11} fill={MUTED} fontFamily="var(--font-mono)" letterSpacing={2}>
          OPERAND STACK
        </text>
        <text x={212} y={48} fontSize={10} fill={MUTED} fontFamily="var(--font-mono)">
          top ↑
        </text>

        <StackCell x={212} y={62} w={138} h={48} type="Int (top)" value="2" hint="ref →" />
        <StackCell x={212} y={120} w={138} h={48} type="Int" value="2" hint="ref →" />

        {/* Empty stack slots (faded) */}
        <g opacity={0.25}>
          <rect x={212} y={178} width={138} height={36} rx={6} fill="none" stroke="rgba(229, 32, 168, 0.5)" strokeDasharray="2 3" />
          <rect x={212} y={220} width={138} height={36} rx={6} fill="none" stroke="rgba(229, 32, 168, 0.5)" strokeDasharray="2 3" />
        </g>
        <text x={212} y={300} fontSize={10} fill={MUTED} fontFamily="var(--font-mono)" letterSpacing={2}>
          SP →
        </text>

        {/* HEAP panel */}
        <rect
          x={380}
          y={10}
          width={250}
          height={320}
          rx={10}
          fill={HEAP_BG}
          stroke="rgba(74, 215, 197, 0.18)"
          strokeDasharray="3 4"
        />
        <text x={392} y={32} fontSize={11} fill={MUTED} fontFamily="var(--font-mono)" letterSpacing={2}>
          HEAP · refcounted
        </text>

        {/* Top: shared Int 2 — TWO stack refs land here, hence rc 2 */}
        <HeapEntry x={392} y={50} w={226} h={62} type="Int" value="2" rc="2" />
        <HeapEntry x={392} y={122} w={226} h={52} type="Int" value="1" rc="1" />
        <HeapEntry x={392} y={184} w={226} h={52} type="Str" value="'name'" rc="1" />
        <HeapEntry x={392} y={246} w={226} h={52} type="Code" value="module" rc="1" />

        {/* Free-cell hint at bottom */}
        <g opacity={0.4}>
          <rect x={392} y={308} width={226} height={18} rx={4} fill="none" stroke="rgba(255,255,255,0.20)" strokeDasharray="2 3" />
          <text x={505} y={320} fontSize={9} fill={MUTED} textAnchor="middle" fontFamily="var(--font-mono)">
            free pages
          </text>
        </g>

        {/* ARROWS: both stack cells point at the SAME Int 2 in the heap */}
        <path
          d="M 350 86 C 372 86, 380 80, 392 78"
          fill="none"
          stroke={ARROW}
          strokeWidth={1.6}
          markerEnd="url(#arrow-aqua-3)"
        />
        <path
          d="M 350 144 C 372 144, 380 80, 392 80"
          fill="none"
          stroke={ARROW}
          strokeWidth={1.6}
          markerEnd="url(#arrow-aqua-3)"
        />
      </svg>
    </div>
  );
}
