import { DatabaseSync } from 'node:sqlite'

/** A year of made-up orders in an in-memory SQLite database. */
export const db = new DatabaseSync(':memory:')
db.exec('CREATE TABLE orders (id INTEGER PRIMARY KEY, month TEXT, country TEXT, product TEXT, amount REAL)')

const insert = db.prepare('INSERT INTO orders (month, country, product, amount) VALUES (?, ?, ?, ?)')
const products = ['laptop', 'phone', 'tablet', 'monitor', 'headphones']
const countries = ['US', 'UK', 'DE', 'FR', 'JP', 'BR']
for (let i = 0; i < 3000; i++) {
  const month = 1 + (i % 12)
  const product = i % products.length
  // each product grows at a different rate through the year
  const amount = (50 + ((i * 37) % 400)) * (1 + (month * product) / 20)
  insert.run(`2026-${String(month).padStart(2, '0')}`, countries[(i * 7) % countries.length]!, products[product]!, Math.round(amount))
}
