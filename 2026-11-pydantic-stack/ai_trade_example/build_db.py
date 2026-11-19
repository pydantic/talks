#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb>=1.5", "yfinance>=0.2.60"]
# ///
"""Build `ai_trade.duckdb` for the AI trade / Situational Awareness example.

Data sources:
- Situational Awareness LP 13F-HR filings (CIK 2045724) from SEC EDGAR: Q1 2026 (period 2026-03-31)
  and Q2 2026 (period 2026-06-30). The Q2 filing is the last snapshot before the fund's July 2026 collapse.
- Daily prices and FX rates from Yahoo Finance via `yfinance`.
- A hand-curated list of companies (tickers, CUSIPs, themes) and dated fund events.

Every download is cached under `raw_data/` so re-running the build is offline and reproducible.
Run with `uv run ai_trade_example/build_db.py` (or `make ai-trade-db`).
"""

# yfinance is a script-only dependency (see the PEP 723 header) and ships no type stubs.
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent
RAW_DIR = ROOT / 'raw_data'
PRICES_DIR = RAW_DIR / 'prices'
FX_DIR = RAW_DIR / 'fx'
DEFAULT_DB = ROOT / 'ai_trade.duckdb'

FUND_CIK = 2045724
EDGAR_USER_AGENT = 'pydantic-talk samuel@pydantic.dev'
FILINGS_13F: list[tuple[str, str]] = [
    # (period end, accession number)
    ('2026-03-31', '0002045724-26-000008'),
    ('2026-06-30', '0000935836-26-000418'),
]

PRICE_START = '2026-01-01'
PRICE_END = '2026-09-01'  # exclusive


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    cusip: str | None
    exchange: str
    currency: str
    sector: str
    theme: str


# fmt: off
COMPANIES: list[Company] = [
    # --- memory: the epicentre of the July 2026 selloff -------------------------------------------
    Company('MU', 'Micron Technology', '595112103', 'NASDAQ', 'USD', 'Technology', 'memory'),
    Company('SNDK', 'Sandisk', '80004C200', 'NASDAQ', 'USD', 'Technology', 'memory'),
    Company('WDC', 'Western Digital', '958102105', 'NASDAQ', 'USD', 'Technology', 'memory'),
    Company('000660.KS', 'SK Hynix', None, 'KRX', 'KRW', 'Technology', 'memory'),
    Company('005930.KS', 'Samsung Electronics', None, 'KRX', 'KRW', 'Technology', 'memory'),
    # --- semiconductors ------------------------------------------------------------------------
    Company('NVDA', 'Nvidia', '67066G104', 'NASDAQ', 'USD', 'Technology', 'semis'),
    Company('AMD', 'Advanced Micro Devices', '007903107', 'NASDAQ', 'USD', 'Technology', 'semis'),
    Company('AVGO', 'Broadcom', '11135F101', 'NASDAQ', 'USD', 'Technology', 'semis'),
    Company('TSM', 'Taiwan Semiconductor (ADR)', '874039100', 'NYSE', 'USD', 'Technology', 'semis'),
    Company('INTC', 'Intel', '458140100', 'NASDAQ', 'USD', 'Technology', 'semis'),
    Company('ARM', 'Arm Holdings (ADR)', '042068106', 'NASDAQ', 'USD', 'Technology', 'semis'),
    Company('MRVL', 'Marvell Technology', '573874104', 'NASDAQ', 'USD', 'Technology', 'semis'),
    Company('ASML', 'ASML Holding (NY shares)', 'N07059210', 'NASDAQ', 'USD', 'Technology', 'semis'),
    Company('STM', 'STMicroelectronics (NY shares)', '861012102', 'NYSE', 'USD', 'Technology', 'semis'),
    Company('VSH', 'Vishay Intertechnology', '928298108', 'NYSE', 'USD', 'Technology', 'semis'),
    Company('CBRS', 'Cerebras Systems', '15675D103', 'NASDAQ', 'USD', 'Technology', 'semis'),
    # --- neoclouds -----------------------------------------------------------------------------
    Company('CRWV', 'CoreWeave', '21873S108', 'NASDAQ', 'USD', 'Technology', 'neocloud'),
    Company('NBIS', 'Nebius Group', 'N97284108', 'NASDAQ', 'USD', 'Technology', 'neocloud'),
    Company('SHAZ', 'SharonAI Holdings', '778920306', 'NASDAQ', 'USD', 'Technology', 'neocloud'),
    # --- bitcoin miners pivoting to AI/HPC hosting ---------------------------------------------
    Company('IREN', 'IREN', 'Q4982L109', 'NASDAQ', 'USD', 'Technology', 'miner_hpc'),
    Company('CORZ', 'Core Scientific', '21874A106', 'NASDAQ', 'USD', 'Technology', 'miner_hpc'),
    Company('APLD', 'Applied Digital', '038169207', 'NASDAQ', 'USD', 'Technology', 'miner_hpc'),
    Company('RIOT', 'Riot Platforms', '767292105', 'NASDAQ', 'USD', 'Technology', 'miner_hpc'),
    Company('CLSK', 'CleanSpark', '18452B209', 'NASDAQ', 'USD', 'Technology', 'miner_hpc'),
    Company('BTDR', 'Bitdeer Technologies', 'G11448100', 'NASDAQ', 'USD', 'Technology', 'miner_hpc'),
    Company('HIVE', 'HIVE Digital Technologies', '433921103', 'NASDAQ', 'USD', 'Technology', 'miner_hpc'),
    Company('WYFI', 'WhiteFiber', 'G96115103', 'NASDAQ', 'USD', 'Technology', 'miner_hpc'),
    Company('CIFR', 'Cipher Mining', '17253J106', 'NASDAQ', 'USD', 'Technology', 'miner_hpc'),
    Company('WULF', 'TeraWulf', '88080T104', 'NASDAQ', 'USD', 'Technology', 'miner_hpc'),
    # --- power and infrastructure for data centres ----------------------------------------------
    Company('BE', 'Bloom Energy', '093712107', 'NYSE', 'USD', 'Industrials', 'ai_power'),
    Company('VST', 'Vistra', '92840M102', 'NYSE', 'USD', 'Utilities', 'ai_power'),
    Company('CEG', 'Constellation Energy', '21037T109', 'NASDAQ', 'USD', 'Utilities', 'ai_power'),
    Company('GEV', 'GE Vernova', '36828A101', 'NYSE', 'USD', 'Industrials', 'ai_power'),
    Company('SEI', 'Solaris Energy Infrastructure', '83418M103', 'NYSE', 'USD', 'Energy', 'ai_power'),
    Company('TE', 'T1 Energy', '35834F104', 'NYSE', 'USD', 'Industrials', 'ai_power'),
    Company('BW', 'Babcock & Wilcox', '05614L209', 'NYSE', 'USD', 'Industrials', 'ai_power'),
    Company('PSIX', 'Power Solutions International', '73933G202', 'NASDAQ', 'USD', 'Industrials', 'ai_power'),
    Company('KEEL', 'Keel Infrastructure', '486917107', 'NASDAQ', 'USD', 'Industrials', 'ai_power'),
    Company('PUMP', 'ProPetro Holding', '74347M108', 'NYSE', 'USD', 'Energy', 'ai_power'),
    # --- servers, cooling, optics -------------------------------------------------------------
    Company('SMCI', 'Super Micro Computer', '86800U104', 'NASDAQ', 'USD', 'Technology', 'hardware'),
    Company('DELL', 'Dell Technologies', '24703L202', 'NYSE', 'USD', 'Technology', 'hardware'),
    Company('VRT', 'Vertiv Holdings', '92537N108', 'NYSE', 'USD', 'Industrials', 'hardware'),
    Company('GLW', 'Corning', '219350105', 'NYSE', 'USD', 'Technology', 'hardware'),
    # --- hyperscalers and the big AI spenders ---------------------------------------------------
    Company('META', 'Meta Platforms', '30303M102', 'NASDAQ', 'USD', 'Communication Services', 'hyperscaler'),
    Company('MSFT', 'Microsoft', '594918104', 'NASDAQ', 'USD', 'Technology', 'hyperscaler'),
    Company('GOOGL', 'Alphabet', '02079K305', 'NASDAQ', 'USD', 'Communication Services', 'hyperscaler'),
    Company('AMZN', 'Amazon', '023135106', 'NASDAQ', 'USD', 'Consumer Discretionary', 'hyperscaler'),
    Company('ORCL', 'Oracle', '68389X105', 'NYSE', 'USD', 'Technology', 'hyperscaler'),
    Company('9984.T', 'SoftBank Group', None, 'TSE', 'JPY', 'Technology', 'hyperscaler'),
    # --- software and IT services "at risk" from AI agents ------------------------------------
    Company('ADBE', 'Adobe', '00724F101', 'NASDAQ', 'USD', 'Technology', 'software_at_risk'),
    Company('CRM', 'Salesforce', '79466L302', 'NYSE', 'USD', 'Technology', 'software_at_risk'),
    Company('NOW', 'ServiceNow', '81762P102', 'NYSE', 'USD', 'Technology', 'software_at_risk'),
    Company('INTU', 'Intuit', '461202103', 'NASDAQ', 'USD', 'Technology', 'software_at_risk'),
    Company('WDAY', 'Workday', '98138H101', 'NASDAQ', 'USD', 'Technology', 'software_at_risk'),
    Company('ADSK', 'Autodesk', '052769106', 'NASDAQ', 'USD', 'Technology', 'software_at_risk'),
    Company('TEAM', 'Atlassian', '049468101', 'NASDAQ', 'USD', 'Technology', 'software_at_risk'),
    Company('INFY', 'Infosys (ADR)', '456788108', 'NYSE', 'USD', 'Technology', 'software_at_risk'),
    # --- indices and sector ETFs ----------------------------------------------------------------
    Company('SPY', 'S&P 500 ETF', '78462F103', 'NYSE', 'USD', 'ETF', 'index'),
    Company('QQQ', 'Nasdaq-100 ETF', '46090E103', 'NASDAQ', 'USD', 'ETF', 'index'),
    Company('SOXX', 'iShares Semiconductor ETF', '464287523', 'NASDAQ', 'USD', 'ETF', 'index'),
    Company('SMH', 'VanEck Semiconductor ETF', '92189F676', 'NASDAQ', 'USD', 'ETF', 'index'),
    Company('IGV', 'iShares Expanded Tech-Software ETF', '464287480', 'NYSE', 'USD', 'ETF', 'index'),
]
# fmt: on

FX_SYMBOLS: dict[str, str] = {'KRW': 'KRW=X', 'JPY': 'JPY=X'}  # Yahoo quotes USD→currency

# Positions reported in the press but not visible in a 13F (foreign listings, shorts).
REPORTED_HOLDINGS: list[tuple[str, str, str, str]] = [
    # (report_date, ticker, issuer, side)
    ('2026-06-30', '000660.KS', 'SK HYNIX INC', 'long'),
    ('2026-06-30', 'ADBE', 'ADOBE INC', 'short'),
]

FUND_EVENTS: list[tuple[str, str, float | None, str]] = [
    # (date, headline, aum_usd_bn, source_url)
    (
        '2026-07-01',
        'Situational Awareness reports a ~439% net return for H1 2026; assets peak around $45bn',
        45.0,
        'https://www.cnbc.com/2026/07/31/leopold-aschenbrenner-situational-awareness-fund-fire-sale.html',
    ),
    (
        '2026-07-08',
        'Intel down 21% in seven sessions as the semiconductor selloff spreads',
        None,
        'https://www.forbes.com/sites/petercohan/2026/07/08/intel-stock-down-21-inside-the-july-2026-semiconductor-selloff/',
    ),
    (
        '2026-07-19',
        'Investors dump Big Tech as scrutiny of AI capex grows',
        None,
        'https://www.bloomberg.com/news/articles/2026-07-19/big-tech-needs-to-justify-ai-spending-as-investors-dump-stocks',
    ),
    (
        '2026-07-28',
        'AI memory boom goes bust: Micron, SK Hynix and Sandisk down ~30% and still falling',
        None,
        'https://247wallst.com/investing/2026/07/28/ai-memory-boom-goes-bust-micron-sk-hynix-sandisk-plunge-30-and-are-still-falling/',
    ),
    (
        '2026-07-29',
        'Chip stocks shed more than $1tn; AI-beneficiary basket trails AI-disruption basket by a record 42 points',
        None,
        'https://www.cnbc.com/2026/07/29/chip-selloff-sk-hynix-samsung-softbank.html',
    ),
    (
        '2026-07-30',
        'Margin calls: Situational Awareness reported to be unwinding trades and seeking capital',
        None,
        'https://www.bloomberg.com/news/articles/2026-07-30/aschenbrenner-hedge-fund-situational-awareness-seeks-capital-after-loss-ft-says',
    ),
    (
        '2026-07-30',
        'Situational Awareness assets fall to about $10bn after liquidations',
        10.0,
        'https://www.bloomberg.com/news/articles/2026-07-30/situational-awareness-assets-fall-to-10-billion-after-losses',
    ),
    (
        '2026-07-31',
        'Fund sells its entire public equity book to Citadel at a discount',
        None,
        'https://www.cnbc.com/2026/07/31/leopold-aschenbrenner-situational-awareness-fund-fire-sale.html',
    ),
]


# ---------------------------------------------------------------------------------------------
# EDGAR 13F
# ---------------------------------------------------------------------------------------------


def _edgar_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': EDGAR_USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def ensure_13f(period: str, accession: str) -> Path:
    """Download the 13F information table XML for one filing if it isn't cached."""
    target = RAW_DIR / f'13f_{period}.xml'
    if target.exists():
        return target
    acc = accession.replace('-', '')
    base = f'https://www.sec.gov/Archives/edgar/data/{FUND_CIK}/{acc}'
    index = json.loads(_edgar_get(f'{base}/index.json'))
    names = [i['name'] for i in index['directory']['item']]
    xml_names = [n for n in names if n.endswith('.xml') and n != 'primary_doc.xml']
    if len(xml_names) != 1:
        raise RuntimeError(f'expected one information table in {base}, found {xml_names}')
    target.write_bytes(_edgar_get(f'{base}/{xml_names[0]}'))
    print(f'downloaded {target.name} <- {xml_names[0]}')
    return target


@dataclass(frozen=True)
class Holding:
    report_date: str
    ticker: str | None
    issuer: str
    title_of_class: str
    cusip: str
    side: str  # 'long' | 'put' | 'call' | 'short'
    shares: int | None
    value_usd: float | None
    source: str  # '13f' | 'reported'


def parse_13f(path: Path, period: str, cusip_to_ticker: dict[str, str]) -> list[Holding]:
    root = ET.parse(path).getroot()
    ns = root.tag[1:].split('}')[0] if root.tag.startswith('{') else ''

    def tag(name: str) -> str:
        return f'{{{ns}}}{name}' if ns else name

    def text(el: ET.Element, name: str) -> str | None:
        child = el.find(tag(name))
        return child.text.strip() if child is not None and child.text else None

    holdings: list[Holding] = []
    for it in root.iter(tag('infoTable')):
        cusip = text(it, 'cusip') or ''
        shrs = it.find(tag('shrsOrPrnAmt'))
        shares = int(text(shrs, 'sshPrnamt') or 0) if shrs is not None else None
        put_call = (text(it, 'putCall') or '').lower()
        holdings.append(
            Holding(
                report_date=period,
                ticker=cusip_to_ticker.get(cusip),
                issuer=text(it, 'nameOfIssuer') or '',
                title_of_class=text(it, 'titleOfClass') or '',
                cusip=cusip,
                side=put_call or 'long',
                shares=shares,
                value_usd=float(text(it, 'value') or 0),
                source='13f',
            )
        )
    return holdings


# ---------------------------------------------------------------------------------------------
# Yahoo Finance prices and FX
# ---------------------------------------------------------------------------------------------


def _download_history(symbol: str) -> list[tuple[date, float, int]]:
    import yfinance as yf

    hist = yf.Ticker(symbol).history(start=PRICE_START, end=PRICE_END, auto_adjust=True)
    rows: list[tuple[date, float, int]] = []
    for ts, row in hist.iterrows():
        close = float(row['Close'])
        if close != close:  # NaN
            continue
        volume = int(row['Volume']) if row['Volume'] == row['Volume'] else 0
        rows.append((ts.date(), close, volume))
    return rows


def ensure_prices(ticker: str) -> Path:
    target = PRICES_DIR / f'{ticker}.csv'
    if target.exists():
        return target
    rows = _download_history(ticker)
    if not rows:
        raise RuntimeError(f'no price data returned for {ticker}')
    with target.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'ticker', 'close', 'volume'])
        for d, close, volume in rows:
            w.writerow([d.isoformat(), ticker, f'{close:.6f}', volume])
    print(f'downloaded {ticker}: {len(rows)} rows')
    return target


def ensure_fx(currency: str, symbol: str) -> Path:
    target = FX_DIR / f'{currency}.csv'
    if target.exists():
        return target
    rows = _download_history(symbol)
    if not rows:
        raise RuntimeError(f'no FX data returned for {symbol}')
    with target.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'currency', 'usd_per_unit'])
        for d, rate, _ in rows:
            # Yahoo quotes how many units of `currency` one USD buys; we store the inverse.
            w.writerow([d.isoformat(), currency, f'{1 / rate:.10f}'])
    print(f'downloaded {currency}: {len(rows)} rows')
    return target


# ---------------------------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE companies (
    ticker   VARCHAR PRIMARY KEY,
    name     VARCHAR NOT NULL,
    cusip    VARCHAR,
    exchange VARCHAR NOT NULL,
    currency VARCHAR NOT NULL,
    sector   VARCHAR NOT NULL,
    theme    VARCHAR NOT NULL
);

CREATE TABLE prices (
    date   DATE NOT NULL,
    ticker VARCHAR NOT NULL,
    close  DOUBLE NOT NULL,
    volume BIGINT NOT NULL,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE fx_rates (
    date         DATE NOT NULL,
    currency     VARCHAR NOT NULL,
    usd_per_unit DOUBLE NOT NULL,
    PRIMARY KEY (date, currency)
);

CREATE TABLE fund_holdings (
    report_date    DATE NOT NULL,
    ticker         VARCHAR,
    issuer         VARCHAR NOT NULL,
    title_of_class VARCHAR,
    cusip          VARCHAR,
    side           VARCHAR NOT NULL,
    shares         BIGINT,
    value_usd      DOUBLE,
    source         VARCHAR NOT NULL
);

CREATE TABLE fund_events (
    date       DATE NOT NULL,
    headline   VARCHAR NOT NULL,
    aum_usd_bn DOUBLE,
    source_url VARCHAR
);

CREATE VIEW prices_usd AS
SELECT p.date, p.ticker, c.name, c.theme, c.currency,
       p.close AS close_local,
       p.close * COALESCE(fx.usd_per_unit, 1.0) AS close_usd,
       p.volume
FROM prices p
JOIN companies c ON c.ticker = p.ticker
LEFT JOIN fx_rates fx ON fx.currency = c.currency AND fx.date = p.date;
"""


def build(db_path: Path) -> None:
    RAW_DIR.mkdir(exist_ok=True)
    PRICES_DIR.mkdir(exist_ok=True)
    FX_DIR.mkdir(exist_ok=True)

    cusip_to_ticker = {c.cusip: c.ticker for c in COMPANIES if c.cusip}
    holdings: list[Holding] = []
    for period, accession in FILINGS_13F:
        holdings.extend(parse_13f(ensure_13f(period, accession), period, cusip_to_ticker))
    for report_date, ticker, issuer, side in REPORTED_HOLDINGS:
        holdings.append(Holding(report_date, ticker, issuer, '', '', side, None, None, 'reported'))

    unmapped = sorted({h.issuer for h in holdings if h.ticker is None and h.source == '13f'})
    if unmapped:
        print(f'13F issuers without a ticker mapping: {unmapped}')

    # Sequential downloads: yfinance's shared cache locks up under concurrent access.
    for company in COMPANIES:
        ensure_prices(company.ticker)
    for currency, symbol in FX_SYMBOLS.items():
        ensure_fx(currency, symbol)

    if db_path.exists():
        db_path.unlink()
    conn = duckdb.connect(str(db_path))
    conn.execute(SCHEMA)
    conn.executemany(
        'INSERT INTO companies VALUES (?, ?, ?, ?, ?, ?, ?)',
        [(c.ticker, c.name, c.cusip, c.exchange, c.currency, c.sector, c.theme) for c in COMPANIES],
    )
    conn.execute(f"INSERT INTO prices SELECT date, ticker, close, volume FROM read_csv('{PRICES_DIR}/*.csv')")
    conn.execute(f"INSERT INTO fx_rates SELECT date, currency, usd_per_unit FROM read_csv('{FX_DIR}/*.csv')")
    conn.executemany(
        'INSERT INTO fund_holdings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        [
            (h.report_date, h.ticker, h.issuer, h.title_of_class, h.cusip, h.side, h.shares, h.value_usd, h.source)
            for h in holdings
        ],
    )
    conn.executemany('INSERT INTO fund_events VALUES (?, ?, ?, ?)', FUND_EVENTS)

    for table in ['companies', 'prices', 'fx_rates', 'fund_holdings', 'fund_events']:
        (count,) = conn.execute(f'SELECT count(*) FROM {table}').fetchone() or (0,)
        print(f'{table:15} {count:>7} rows')
    conn.close()
    print(f'wrote {db_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB, help=f'output database path (default: {DEFAULT_DB})')
    args = parser.parse_args()
    build(args.db)


if __name__ == '__main__':
    main()
