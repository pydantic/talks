# Positions and leverage

## What the press reported the fund held

Longs (public):
- Memory: Micron, SK Hynix, Sandisk.
- Neoclouds and AI hosting: Nebius Group, CoreWeave.
- Bitcoin miners pivoting to AI/HPC hosting: IREN, Core Scientific, Applied Digital, Riot Platforms,
  CleanSpark, WhiteFiber. The fund owned about 8.2% of Core Scientific's float and 4 to 5% of Applied
  Digital, CleanSpark and WhiteFiber (SpotGamma).
- Data-centre power and equipment: Bloom Energy and other power names; Super Micro Computer (servers).

Shorts and hedges:
- Software: Adobe is the short every outlet names. The thesis was that autonomous AI agents weaken the
  traditional SaaS model.
- Puts: SpotGamma cites roughly $10.0 billion notional of puts on the VanEck Semiconductor ETF (SMH) and
  $0.6 billion on Nvidia. The Q1 2026 13F in the database shows the fund's put positions in detail.

Private: a stake in Anthropic (about $5 billion), retained after the public book was sold.

## Leverage

Gross exposure was reported at about 4x. The arithmetic quoted afterwards: a 30-percent-plus fall in the
long book is a hit of roughly 120 percent to equity before shorts and hedges are counted. The prime brokers
named are Bank of America, Goldman Sachs and JPMorgan.

## Reading the 13F filings in the database

The database has two SEC Form 13F-HR snapshots for Situational Awareness LP (CIK 2045724): the Q1 2026
filing (positions as of 31 March, filed 18 May) and the Q2 2026 filing (positions as of 30 June, filed
14 August, after the collapse). Things a 13F does and does not show:

- It lists US-listed long positions in equities and listed options, valued at quarter end. It does not show
  short positions, swaps, or foreign listings, so the SK Hynix long (Korea) and the Adobe short are absent
  and appear in `fund_holdings` only as `source = 'reported'` rows with no size.
- Options are reported at the value of the underlying shares (notional), not at the option's own value.
  A put on $2 billion of an ETF appears as a $2 billion line.
- It is a quarter-end snapshot filed up to 45 days later. Positions can change completely in between.
- Filed values are not the fund's equity. With leverage, the 13F total can be several times what the
  fund's investors actually own.

The shape of the two filings tells its own story. The Q1 book was dominated by puts on semiconductors
(SMH, Nvidia, Oracle, Broadcom, AMD, Micron, TSMC, ASML, Intel) with the longs concentrated in Bloom
Energy, Sandisk, CoreWeave, IREN and Core Scientific. By 30 June the puts were almost all gone and the
book was long about $11 billion of Sandisk and Micron alone, plus TSMC, Nebius, CoreWeave and the miners,
for a filed total of roughly $20 billion.

SpotGamma's description of the "disclosed US long book" as $1.86 billion across 26 positions with Bloom
Energy 22.8%, Sandisk 18.8%, CoreWeave 14.4%, IREN 10.4% and Core Scientific 10.1% matches the Q1 filing's
long-only weights, not its total value; treat the percentages as reliable and the dollar figure as
inconsistent with the filing.

Sources: see `sources.md` (SpotGamma; Seeking Alpha; CNBC; Blockspace; SEC EDGAR).
