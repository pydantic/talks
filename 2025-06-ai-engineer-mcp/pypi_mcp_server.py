"""MCP server to get information about python package downloads."""

import re
from typing import Any

import logfire
from google.api_core.exceptions import BadRequest
from google.cloud import bigquery
from mcp import ServerSession
from mcp.server.fastmcp import Context, FastMCP
from pydantic_ai import Agent, ModelRetry, format_as_xml
from pydantic_ai.models.mcp_sampling import MCPSamplingModel

logfire.configure()
logfire.configure(service_name='mcp-server')
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()


table_name = 'bigquery-public-data.pypi.file_downloads'
client = bigquery.Client()


pypi_agent = Agent(
    retries=2,
    system_prompt=f"""
Your job is to help users analyze downloads of python packages.

Convert the user's query into a BigQuery SQL query against the `{table_name}`
table which has the following schema:

```sql
CREATE TABLE {table_name} (
    timestamp TIMESTAMP,
    country_code STRING, -- two letter ISO country code
    url STRING,
    project STRING,
    file STRUCT<
        filename STRING,
        project STRING,
        version STRING,
        type STRING
    >,
    details STRUCT<
        installer STRUCT<
            name STRING,
            version STRING
        >,
        python STRING,
        implementation STRUCT<
            name STRING,
            version STRING
        >,
        distro STRUCT<
            name STRING,
            version STRING,
            id STRING,
            libc STRUCT<
                lib STRING,
                version STRING
            >
        >,
        system STRUCT<
            name STRING,
            release STRING
        >,
        cpu STRING,
        openssl_version STRING,
        setuptools_version STRING,
        rustc_version STRING,
        ci BOOLEAN
    >,
    tls_protocol STRING,
    tls_cipher STRING
);
```

Where possible apply a lower bound constraint to the `timestamp` column to avoid scanning to many partitions.

For example, if the user asked for an example download of the pydantic package, you could use the following query:

```sql
SELECT *
FROM `bigquery-public-data.pypi.file_downloads`
WHERE
  file.project = 'pydantic'
  AND DATE(timestamp) = current_date()
LIMIT 1
```

If the user asked for "number of downloads of pydantic broken down by month, python version, operating system,
CPU architecture, and libc version for this year and last year", you could use the following query:

```sql
SELECT
  COUNT(*) AS num_downloads,
  DATE_TRUNC(DATE(timestamp), MONTH) AS `month`,
  REGEXP_EXTRACT(details.python, r"[0-9]+\\.[0-9]+") AS python_version,
  details.system.name AS os,
  details.cpu AS cpu,
  details.distro.libc.lib AS libc
FROM `bigquery-public-data.pypi.file_downloads`
WHERE
  file.project = 'pydantic'
  AND DATE_TRUNC(DATE(timestamp), YEAR) = DATE_TRUNC(date_sub(current_date(), interval 1 YEAR), YEAR)
GROUP BY `month`, `python_version`, `os`, `cpu`, `libc`
ORDER BY `month` DESC, `num_downloads` DESC
```
""",
)


@pypi_agent.output_validator
def run_query(sql: str) -> str:
    # remove "```sql...```"
    m = re.search(r'```\w*\n(.*?)```', sql, flags=re.S)
    if m:
        sql = m.group(1).strip()

    logfire.info('running {sql}', sql=sql)
    if f'from `{table_name}`' not in sql.lower():
        raise ModelRetry(f'Query must be against the `{table_name}` table')
    try:
        query_job = client.query(sql)
        rows = query_job.result()
    except BadRequest as e:
        raise ModelRetry(f'Invalid query: {e}') from e
    data = [dict(row) for row in rows]
    return format_as_xml(data, item_tag='row', include_root_tag=False)


mcp = FastMCP('PyPI query', log_level='WARNING')


@mcp.tool()
async def pypi_downloads(question: str, ctx: Context[ServerSession, None]) -> str:
    """Analyze downloads of packages from the Python package index PyPI to answer questions about package downloads."""
    result = await pypi_agent.run(question, model=MCPSamplingModel(session=ctx.session))
    return result.output


if __name__ == '__main__':
    mcp.run()
    # from devtools import debug
    # result = pypi_agent.run_sync('How many times has pydantic been downloaded this year?', model='openai:gpt-4o')
    # debug(result.output)
