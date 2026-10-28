# Sources for the "Aside" charts

Three slides in the FastAPI Conf 2026 deck: model intelligence over time, model intelligence vs. training compute,
and what it costs to stay on the line. All data was read on 2026-10-28. The numbers used are in
`components/aside/data.ts`; the fits are computed in that file from the data arrays.

## Data

- **Epoch Capabilities Index (ECI)**: https://epoch.ai/eci, scores downloaded from https://epoch.ai/data/eci_scores.csv.
  The scale is linear by construction, with Claude 3.5 Sonnet = 130 and GPT-5 = 150. Epoch notes that only relative
  changes in slope are meaningful.
- **Training compute and training cost**: Epoch's notable models database,
  https://epoch.ai/data/notable_ai_models.csv. Columns used: "Training compute (FLOP)" and "Training compute cost
  (2023 USD)". Several compute estimates are tagged "speculative" by Epoch, including GPT-5 and Claude 3.5 Sonnet.
- **Trend figures**: https://epoch.ai/trends. Training compute for frontier language models grows 5x per year since
  2020 (90% CI 4x to 6x). AI chip performance per dollar grows 1.49x per year since 2023 (90% CI 1.36x to 1.66x).
  The cost to train frontier language models grows 3.5x per year since 2020 (90% CI 2.8x to 4.4x).
- **Epoch on the pace of capability progress**: https://epoch.ai/data-insights/ai-capabilities-progress-has-sped-up.
  A piecewise-linear fit gives about 8 ECI points per year before April 2024 and about 15 after.
- **US GDP**: $30.8T nominal for 2025, BEA, via https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_GDP.

## Chart 1: model intelligence over time

Flagship releases from OpenAI, Anthropic and Google with an ECI score, GPT-4 (March 2023) to Gemini 4 Argon
(September 2026). The line is an ordinary least-squares fit of ECI against release date; the slope comes out at
about 15 points per year, matching Epoch's post-2024 figure. Past today the line is an extrapolation.

## Chart 2: model intelligence vs. training compute

Every model where Epoch publishes both an ECI score and a training compute estimate: the flagships above plus
Llama 3.1 405B, DeepSeek-V3, DeepSeek-R1, Grok 3 and Grok 4. The line is a least-squares fit of ECI against
log10(compute), about 10 points per 10x compute.

This fit should not be used for projection. Fitting ECI against both log compute and release year, compute's
coefficient collapses to roughly zero: in this sample, time (algorithmic progress, post-training, reasoning)
explains the variation, not compute. GPT-5 beats GPT-4.5 on about a sixth of the compute. Chart 3 therefore does
not chain from this fit.

## Chart 3: what it costs to stay on the line

**Dots.** Epoch's cost column is the amortized hardware and energy cost of the final training run only, in 2023
USD: the depreciated share of chips, servers and interconnect for the hours of the run, plus electricity. It
excludes staff, experiments, failed runs and the capital cost of building the cluster. Epoch's method is described
at https://epoch.ai/publications/how-much-does-it-cost-to-train-frontier-ai-models (Cottier et al., "The rising
costs of training frontier AI models", https://arxiv.org/abs/2405.21015).

The dots on the slide multiply Epoch's figure by 3 to approximate full development cost. Justification:

- Epoch's development-cost breakdown puts hardware at 47% to 67% of the total, with R&D staff at 29% to 49% and
  energy at 2% to 6%. Epoch summarised the same study as "training alone in the tens of millions for models like
  GPT-4. Full development costs (when amortized) are over $100M for the most advanced models"
  (https://x.com/EpochAIResearch/status/1797706836935659813). Epoch's run-only figure for GPT-4 is $37M.
- Sam Altman said GPT-4 cost "more than $100 million" (MIT Technology Review, April 2023).
- Dario Amodei said in mid-2024 that current models such as GPT-4o cost about $100M, that models then in training
  were around $1B, and that $10B models could follow in 2025 to 2027
  (https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-models-that-cost-dollar1-billion-to-train-are-in-development-dollar100-billion-models-coming-soon-largest-current-models-take-only-dollar100-million-to-train-anthropic-ceo).
- The Stanford AI Index, with Epoch, gives cloud-rental-basis figures of $78M for GPT-4, $191M for Gemini 1.0 Ultra
  and $170M for Llama 3.1 405B (AI Index 2024 and 2025, https://aiindex.stanford.edu). Epoch notes cloud-rental
  estimates average about twice the amortized ones.

These outside figures are the hollow diamonds on the slide.

**Line.** The slope is Epoch's measured trend, not a fit to the dots: compute 5x per year divided by 1.49x cheaper
FLOP per year gives 3.4x per year, which matches Epoch's directly measured 3.5x. The intercept is fitted to the
runs that were the most expensive known at their release (GPT-4, Llama 3.1 405B, GPT-4.5, Grok 4). Epoch's 2024
paper projected the largest runs passing $1B by 2027; the line gives about $7B on the development-cost basis.

**Reference lines.** 0.1%, 1% and 10% of 2025 US GDP.

**What the chart does not show.** Industry-wide AI data centre investment is a different quantity: roughly 1% to
2.5% of US GDP in 2026 depending on definition (Epoch: https://epoch.ai/data-insights/ai-datacenter-share-gdp;
Apollo and Goldman Sachs estimates of $500B to $750B of hyperscaler capex in 2026). That covers every lab and
hyperscaler, inference, research and future training. One frontier run is a few percent of one lab's fleet, and
the amortized run cost is 10% to 20% of the hardware's purchase price. See
https://epoch.ai/gradient-updates/frontier-labs-dont-use-most-ai-compute for fleet sizes.
