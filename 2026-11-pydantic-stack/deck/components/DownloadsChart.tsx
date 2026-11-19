/**
 * Pydantic Validation monthly PyPI downloads, rendered as an inline SVG line chart.
 * Single-series adaptation of the chart from the March 2026 board deck.
 */

interface DataPoint {
  month: string
  pydantic: number
}

const DATA: DataPoint[] = [
  { month: '2018/01', pydantic: 0.0 },
  { month: '2018/02', pydantic: 0.0 },
  { month: '2018/03', pydantic: 0.0 },
  { month: '2018/04', pydantic: 0.0 },
  { month: '2018/05', pydantic: 0.0 },
  { month: '2018/06', pydantic: 0.0 },
  { month: '2018/07', pydantic: 0.0 },
  { month: '2018/08', pydantic: 0.01 },
  { month: '2018/09', pydantic: 0.02 },
  { month: '2018/10', pydantic: 0.02 },
  { month: '2018/11', pydantic: 0.03 },
  { month: '2018/12', pydantic: 0.02 },
  { month: '2019/01', pydantic: 0.02 },
  { month: '2019/02', pydantic: 0.04 },
  { month: '2019/03', pydantic: 0.11 },
  { month: '2019/04', pydantic: 0.15 },
  { month: '2019/05', pydantic: 0.17 },
  { month: '2019/06', pydantic: 0.15 },
  { month: '2019/07', pydantic: 0.22 },
  { month: '2019/08', pydantic: 0.18 },
  { month: '2019/09', pydantic: 0.22 },
  { month: '2019/10', pydantic: 0.28 },
  { month: '2019/11', pydantic: 0.31 },
  { month: '2019/12', pydantic: 0.37 },
  { month: '2020/01', pydantic: 0.48 },
  { month: '2020/02', pydantic: 0.72 },
  { month: '2020/03', pydantic: 1.02 },
  { month: '2020/04', pydantic: 1.19 },
  { month: '2020/05', pydantic: 1.32 },
  { month: '2020/06', pydantic: 1.58 },
  { month: '2020/07', pydantic: 1.81 },
  { month: '2020/08', pydantic: 1.87 },
  { month: '2020/09', pydantic: 2.09 },
  { month: '2020/10', pydantic: 2.45 },
  { month: '2020/11', pydantic: 2.6 },
  { month: '2020/12', pydantic: 3.04 },
  { month: '2021/01', pydantic: 3.44 },
  { month: '2021/02', pydantic: 4.55 },
  { month: '2021/03', pydantic: 6.62 },
  { month: '2021/04', pydantic: 7.46 },
  { month: '2021/05', pydantic: 8.6 },
  { month: '2021/06', pydantic: 10.72 },
  { month: '2021/07', pydantic: 9.76 },
  { month: '2021/08', pydantic: 10.07 },
  { month: '2021/09', pydantic: 10.74 },
  { month: '2021/10', pydantic: 13.79 },
  { month: '2021/11', pydantic: 13.15 },
  { month: '2021/12', pydantic: 16.66 },
  { month: '2022/01', pydantic: 16.42 },
  { month: '2022/02', pydantic: 17.97 },
  { month: '2022/03', pydantic: 22.38 },
  { month: '2022/04', pydantic: 21.41 },
  { month: '2022/05', pydantic: 27.73 },
  { month: '2022/06', pydantic: 27.31 },
  { month: '2022/07', pydantic: 24.12 },
  { month: '2022/08', pydantic: 29.59 },
  { month: '2022/09', pydantic: 36.31 },
  { month: '2022/10', pydantic: 39.13 },
  { month: '2022/11', pydantic: 42.3 },
  { month: '2022/12', pydantic: 42.35 },
  { month: '2023/01', pydantic: 48.61 },
  { month: '2023/02', pydantic: 50.31 },
  { month: '2023/03', pydantic: 59.22 },
  { month: '2023/04', pydantic: 61.0 },
  { month: '2023/05', pydantic: 69.93 },
  { month: '2023/06', pydantic: 70.67 },
  { month: '2023/07', pydantic: 84.77 },
  { month: '2023/08', pydantic: 100.36 },
  { month: '2023/09', pydantic: 111.27 },
  { month: '2023/10', pydantic: 130.86 },
  { month: '2023/11', pydantic: 127.99 },
  { month: '2023/12', pydantic: 134.55 },
  { month: '2024/01', pydantic: 144.31 },
  { month: '2024/02', pydantic: 145.77 },
  { month: '2024/03', pydantic: 170.28 },
  { month: '2024/04', pydantic: 184.58 },
  { month: '2024/05', pydantic: 204.54 },
  { month: '2024/06', pydantic: 206.86 },
  { month: '2024/07', pydantic: 239.22 },
  { month: '2024/08', pydantic: 242.34 },
  { month: '2024/09', pydantic: 287.8 },
  { month: '2024/10', pydantic: 307.89 },
  { month: '2024/11', pydantic: 303.54 },
  { month: '2024/12', pydantic: 288.25 },
  { month: '2025/01', pydantic: 297.48 },
  { month: '2025/02', pydantic: 283.11 },
  { month: '2025/03', pydantic: 325.29 },
  { month: '2025/04', pydantic: 362.78 },
  { month: '2025/05', pydantic: 370.09 },
  { month: '2025/06', pydantic: 381.3 },
  { month: '2025/07', pydantic: 399.13 },
  { month: '2025/08', pydantic: 414.29 },
  { month: '2025/09', pydantic: 463.06 },
  { month: '2025/10', pydantic: 545.55 },
  { month: '2025/11', pydantic: 550.7 },
  { month: '2025/12', pydantic: 549.0 },
  { month: '2026/01', pydantic: 565.83 },
  { month: '2026/02', pydantic: 586.99 },
  { month: '2026/03', pydantic: 764.8 },
  { month: '2026/04', pydantic: 940.25 },
  { month: '2026/05', pydantic: 1060.25 },
  { month: '2026/06', pydantic: 1060.16 },
  { month: '2026/07', pydantic: 1113.61 },
]

const COLOR = '#E520E9'
const W = 420
const H = 260
const MT = 24
const MR = 12
const MB = 28
const ML = 44
const CW = W - ML - MR
const CH = H - MT - MB
const MAX_Y = 1200
const Y_TICKS = [0, 200, 400, 600, 800, 1000, 1200]
const END_LABEL = '1.11B'

export default function DownloadsChart() {
  const xScale = (i: number) => ML + (i / (DATA.length - 1)) * CW
  const yScale = (v: number) => MT + CH - (v / MAX_Y) * CH

  const points = DATA.map((d, i) => `${xScale(i)},${yScale(d.pydantic)}`).join(' ')

  const yearTicks: { label: string; x: number }[] = []
  DATA.forEach((d, i) => {
    if (d.month.endsWith('/01')) {
      const year = d.month.split('/')[0]!
      yearTicks.push({ label: year, x: xScale(i) })
    }
  })

  const lastIdx = DATA.length - 1
  const lastVal = DATA[lastIdx]!.pydantic

  return (
    <div style={{ width: '100%' }}>
      <div
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 13,
          fontWeight: 600,
          color: 'var(--color-heading)',
          marginBottom: 6,
          opacity: 0.85,
        }}
      >
        Pydantic Validation - downloads/month
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }}>
        {Y_TICKS.map((tick) => (
          <g key={tick}>
            <line
              x1={ML}
              x2={ML + CW}
              y1={yScale(tick)}
              y2={yScale(tick)}
              stroke="var(--color-heading)"
              strokeWidth={1}
              opacity={0.12}
            />
            <text
              x={ML - 6}
              y={yScale(tick)}
              fill="var(--color-heading)"
              fontSize={11}
              fontFamily="var(--font-mono)"
              textAnchor="end"
              dominantBaseline="middle"
              opacity={0.5}
            >
              {tick}M
            </text>
          </g>
        ))}

        {yearTicks.map((t) => (
          <text
            key={t.label}
            x={t.x}
            y={MT + CH + 18}
            fill="var(--color-heading)"
            fontSize={11}
            fontFamily="var(--font-mono)"
            textAnchor="middle"
            opacity={0.5}
          >
            {t.label}
          </text>
        ))}

        <polygon
          points={`${xScale(0)},${yScale(0)} ${points} ${xScale(lastIdx)},${yScale(0)}`}
          fill={COLOR}
          opacity={0.1}
        />

        <polyline
          points={points}
          fill="none"
          stroke={COLOR}
          strokeWidth={2.25}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        <circle cx={xScale(lastIdx)} cy={yScale(lastVal)} r={4} fill={COLOR} />

        <text
          x={xScale(lastIdx) - 4}
          y={yScale(lastVal) - 10}
          fill="var(--color-heading)"
          fontSize={13}
          fontFamily="var(--font-mono)"
          fontWeight={700}
          textAnchor="end"
        >
          {END_LABEL}
        </text>
      </svg>
    </div>
  )
}
