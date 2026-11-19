import 'dotenv/config'

import { openai } from '@ai-sdk/openai'
import { OpenTelemetry } from '@ai-sdk/otel'
import { codeMode } from '@pydantic/logfire-ai-sdk-code-mode'
import * as logfire from '@pydantic/logfire-node'
import { generateText, registerTelemetry, stepCountIs, tool } from 'ai'
import { z } from 'zod'

import { db } from './db.ts'

logfire.configure({ serviceName: 'ai-sdk-code-mode', console: false })
registerTelemetry(new OpenTelemetry())

const tools = {
  query: tool({
    description:
      'Run a SQL query on the SQLite database. Tables: orders(id, month "YYYY-MM", country, product, amount in USD).',
    inputSchema: z.object({ sql: z.string() }),
    outputSchema: z.array(z.record(z.string(), z.any())),
    execute: async ({ sql }) =>
      db
        .prepare(sql)
        .all()
        .map((row) => ({ ...row })),
  }),
  chart: tool({
    description: 'Show the user a bar chart.',
    inputSchema: z.object({ title: z.string(), data: z.array(z.object({ label: z.string(), value: z.number() })) }),
    outputSchema: z.string(),
    execute: async ({ title, data }) => {
      const max = Math.max(...data.map((d) => d.value))
      console.log(`\n${title}`)
      for (const { label, value } of data) {
        console.log(`${label.padStart(12)} ${'█'.repeat(Math.round((value / max) * 40))} ${value.toLocaleString()}`)
      }
      return 'chart shown'
    },
  }),
}

{
  // the model gets a single `run_code` tool, and calls `query` and `chart` from the Python it writes
  await using code = codeMode({ tools })
  const { text } = await generateText({
    model: openai('gpt-6-luna'),
    prompt: 'Which product grew fastest during 2026? Chart its revenue by month, and chart total revenue by country.',
    tools: code.tools,
    stopWhen: stepCountIs(10),
  })
  console.log(`\n${text}`)
}

await logfire.shutdown()
