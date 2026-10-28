// Source: Epoch AI, downloaded 2026-10-28.
//   ECI scores:            https://epoch.ai/data/eci_scores.csv  (Epoch Capabilities Index, Claude 3.5 Sonnet = 130, GPT-5 = 150)
//   Compute and cost:      https://epoch.ai/data/notable_ai_models.csv  (training compute in FLOP, amortized hardware + energy cost in 2023 USD)
//   Trend figures:         https://epoch.ai/trends  (compute 5x/yr, cost 3.5x/yr, chip performance per dollar 1.49x/yr)

export type Lab = 'OpenAI' | 'Anthropic' | 'Google' | 'Other';

export interface Model {
  name: string;
  lab: Lab;
  /** Release date, ISO. */
  date: string;
  /** Epoch Capabilities Index. */
  eci: number;
  /** Epoch's training compute estimate, FLOP. */
  compute?: number;
  /** Epoch's amortized training cost estimate, 2023 USD. */
  cost?: number;
}

export const LAB_COLOR: Record<Lab, string> = {
  OpenAI: 'var(--accent-aqua)',
  Anthropic: 'var(--accent-secondary)',
  Google: 'var(--accent-tertiary)',
  Other: 'rgba(245, 238, 248, 0.35)',
};

/** Flagship releases from the three big labs, used for the intelligence-over-time chart. */
export const FLAGSHIPS: Model[] = [
  { name: 'GPT-4', lab: 'OpenAI', date: '2023-03-14', eci: 125.7, compute: 2.1e25, cost: 37.3e6 },
  { name: 'Claude 2', lab: 'Anthropic', date: '2023-07-11', eci: 120.2, compute: 3.87e24, cost: 4.9e6 },
  { name: 'Gemini 1.0 Pro', lab: 'Google', date: '2023-12-13', eci: 117.1 },
  { name: 'Claude 3 Opus', lab: 'Anthropic', date: '2024-02-29', eci: 126.9 },
  { name: 'GPT-4o', lab: 'OpenAI', date: '2024-05-13', eci: 129.0 },
  { name: 'Gemini 1.5 Pro', lab: 'Google', date: '2024-05-14', eci: 126.9 },
  { name: 'Claude 3.5 Sonnet', lab: 'Anthropic', date: '2024-06-20', eci: 130.0, compute: 2.7e25, cost: 25.9e6 },
  { name: 'o1-preview', lab: 'OpenAI', date: '2024-09-12', eci: 134.6 },
  { name: 'o1', lab: 'OpenAI', date: '2024-12-17', eci: 141.7 },
  { name: 'Claude 3.7 Sonnet', lab: 'Anthropic', date: '2025-02-24', eci: 141.0, compute: 3.35e25 },
  { name: 'GPT-4.5', lab: 'OpenAI', date: '2025-02-27', eci: 136.7, compute: 3.8e26, cost: 366e6 },
  { name: 'Gemini 2.5 Pro', lab: 'Google', date: '2025-03-31', eci: 144.0 },
  { name: 'o3', lab: 'OpenAI', date: '2025-04-16', eci: 146.7 },
  { name: 'Claude Opus 4', lab: 'Anthropic', date: '2025-05-22', eci: 142.5 },
  { name: 'GPT-5', lab: 'OpenAI', date: '2025-08-07', eci: 150.0, compute: 6.6e25 },
  { name: 'Claude Sonnet 4.5', lab: 'Anthropic', date: '2025-09-29', eci: 146.7 },
  { name: 'Gemini 3 Pro', lab: 'Google', date: '2025-11-18', eci: 153.1 },
  { name: 'Claude Opus 4.5', lab: 'Anthropic', date: '2025-11-24', eci: 150.1 },
  { name: 'GPT-5.2', lab: 'OpenAI', date: '2025-12-11', eci: 153.7 },
  { name: 'Claude Opus 4.6', lab: 'Anthropic', date: '2026-02-05', eci: 155.7 },
  { name: 'Gemini 3.1 Pro', lab: 'Google', date: '2026-02-19', eci: 155.3 },
  { name: 'GPT-5.5', lab: 'OpenAI', date: '2026-04-23', eci: 159.7 },
  { name: 'Claude Fable 5', lab: 'Anthropic', date: '2026-06-09', eci: 163.4 },
  { name: 'GPT-5.6 Sol', lab: 'OpenAI', date: '2026-07-09', eci: 162.8 },
  { name: 'Claude Opus 5', lab: 'Anthropic', date: '2026-07-24', eci: 163.9 },
  { name: 'GPT-6 Astra', lab: 'OpenAI', date: '2026-09-03', eci: 168.8, compute: 9.4e26 },
  { name: 'Claude Opus 5.5', lab: 'Anthropic', date: '2026-09-22', eci: 168.8 },
  { name: 'Gemini 4 Argon', lab: 'Google', date: '2026-09-30', eci: 166.5 },
];

/** Models from other labs where Epoch publishes both an ECI score and a compute estimate. Grey dots on the compute chart. */
export const OTHERS: Model[] = [
  { name: 'Llama 3.1 405B', lab: 'Other', date: '2024-07-23', eci: 128.8, compute: 3.8e25, cost: 52.9e6 },
  { name: 'DeepSeek-V3', lab: 'Other', date: '2024-12-26', eci: 132.3, compute: 3.3e24, cost: 5.4e6 },
  { name: 'DeepSeek-R1', lab: 'Other', date: '2025-01-20', eci: 138.8, compute: 3.5e24, cost: 6.8e6 },
  { name: 'Grok 3', lab: 'Other', date: '2025-04-09', eci: 138.2, compute: 3.5e26, cost: 217.8e6 },
  { name: 'Grok 4', lab: 'Other', date: '2025-07-09', eci: 146.2, compute: 5e26, cost: 387.8e6 },
];

/** Cost estimates for models without an ECI score, used only to pin the FLOP-per-dollar trend. */
export const COST_ONLY: Model[] = [
  { name: 'Gemini 1.0 Ultra', lab: 'Google', date: '2023-12-06', eci: NaN, compute: 5e25, cost: 30.7e6 },
  { name: 'Grok-2', lab: 'Other', date: '2024-08-13', eci: NaN, compute: 2.96e25, cost: 31.6e6 },
];

export function toYear(iso: string): number {
  const d = new Date(iso);
  const start = Date.UTC(d.getUTCFullYear(), 0, 1);
  const end = Date.UTC(d.getUTCFullYear() + 1, 0, 1);
  return d.getUTCFullYear() + (d.getTime() - start) / (end - start);
}

export interface Fit {
  slope: number;
  intercept: number;
  at: (x: number) => number;
}

/** Ordinary least squares, y = intercept + slope * x. */
export function linearFit(points: [number, number][]): Fit {
  const n = points.length;
  const mx = points.reduce((s, [x]) => s + x, 0) / n;
  const my = points.reduce((s, [, y]) => s + y, 0) / n;
  let sxy = 0;
  let sxx = 0;
  for (const [x, y] of points) {
    sxy += (x - mx) * (y - my);
    sxx += (x - mx) ** 2;
  }
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  return { slope, intercept, at: (x) => intercept + slope * x };
}

/** ECI as a linear function of calendar year, fitted to the flagship releases. */
export const ECI_VS_TIME = linearFit(FLAGSHIPS.map((m) => [toYear(m.date), m.eci]));

/** ECI as a linear function of log10(training compute), fitted to every model with both numbers. */
export const ECI_VS_LOG_COMPUTE = linearFit(
  [...FLAGSHIPS, ...OTHERS].filter((m) => m.compute).map((m) => [Math.log10(m.compute!), m.eci]),
);

/**
 * Epoch's measured trends for frontier language models (epoch.ai/trends, read 2026-10-28).
 * Training compute grows 5x per year; AI chip performance per dollar grows 1.49x per year; so the cost of a
 * frontier run grows about 3.4x per year, which matches Epoch's directly measured 3.5x per year.
 */
export const EPOCH = { computeGrowth: 5, flopPerDollarGrowth: 1.49, costGrowth: 3.5 };
export const COST_GROWTH = EPOCH.computeGrowth / EPOCH.flopPerDollarGrowth;

/** Frontier runs: models that were the most expensive known training run at their release. They anchor the cost line. */
export const FRONTIER_COST: Model[] = (() => {
  const withCost = [...FLAGSHIPS, ...OTHERS, ...COST_ONLY].filter((m) => m.cost).sort((a, b) => a.date.localeCompare(b.date));
  let max = 0;
  return withCost.filter((m) => {
    if (m.cost! <= max) return false;
    max = m.cost!;
    return true;
  });
})();

/**
 * Epoch's cost column is the amortized hardware and energy of the final training run only. Epoch's development-cost
 * analysis puts hardware at 47-67% of full development cost, with experiments and staff making up the rest, and
 * says full development cost is "over $100M" for GPT-4 against $37M for the run. Outside figures agree: Altman said
 * GPT-4 cost "more than $100M", Amodei put 2024-era models at about $100M. So full development cost is about 3x.
 */
export const DEV_COST_MULTIPLIER = 3;

export function devCost(m: Model): number {
  return m.cost! * DEV_COST_MULTIPLIER;
}

/** Estimates from outside Epoch's database, for calibration. Full development or cloud-rental basis, nominal USD. */
export const OTHER_ESTIMATES: { name: string; date: string; cost: number; source: string }[] = [
  { name: 'GPT-4', date: '2023-03-14', cost: 78e6, source: 'Stanford AI Index 2024, cloud rental' },
  { name: 'GPT-4', date: '2023-03-14', cost: 100e6, source: 'Sam Altman, "more than $100M"' },
  { name: 'Gemini 1.0 Ultra', date: '2023-12-06', cost: 191e6, source: 'Stanford AI Index 2024, cloud rental' },
  { name: 'Llama 3.1 405B', date: '2024-07-23', cost: 170e6, source: 'Stanford AI Index 2025' },
  { name: '2024-era models', date: '2024-06-15', cost: 100e6, source: 'Dario Amodei, mid-2024' },
  { name: 'Models in training, 2024', date: '2025-03-01', cost: 1e9, source: 'Dario Amodei, mid-2024' },
];

/** US nominal GDP, 2025, BEA. */
export const US_GDP = 30.8e12;

/**
 * log10(development cost) as a function of year, slope fixed to Epoch's trend and intercept fitted to the frontier
 * runs. "If models keep improving at the current rate" means this line continues.
 */
export const LOG_COST_VS_TIME: Fit = (() => {
  const slope = Math.log10(COST_GROWTH);
  const pts = FRONTIER_COST.map((m) => [toYear(m.date), Math.log10(devCost(m))] as [number, number]);
  const intercept = pts.reduce((s, [x, y]) => s + (y - slope * x), 0) / pts.length;
  return { slope, intercept, at: (x) => intercept + slope * x };
})();

/** Development cost (USD) of a frontier model in a given year, on the trend line. */
export function costOnTrend(year: number): number {
  return 10 ** LOG_COST_VS_TIME.at(year);
}

/** Fitted slopes, for captions and legends. */
export const IMPLIED = {
  eciPerYear: ECI_VS_TIME.slope,
  eciPerOom: ECI_VS_LOG_COMPUTE.slope,
};
