import { useState, useEffect } from "react";

type Prop = { n: string; s: string; d: string }; // name, schema-inner, description
type Tool = {
  name: string;
  description: string;
  descLong: string;
  props: Prop[];
  required: string[];
};

const TOOLS: Tool[] = [
  { name: 'get_weather', description: 'Current weather for a location',
    descLong: 'Fetch the current weather conditions, temperature, and short-term forecast for a given location.',
    props: [
      { n: 'location', s: '"type": "string"', d: 'City name, postal code, or lat,lng coordinates' },
      { n: 'units', s: '"type": "string", "enum": ["c", "f"]', d: 'Temperature units: celsius or fahrenheit' },
    ], required: ['location'] },
  { name: 'search_flights', description: 'Find flights between two airports',
    descLong: 'Search available flights between two airports on a given date and return fares and timings.',
    props: [
      { n: 'origin', s: '"type": "string"', d: 'Origin airport IATA code, for example LHR' },
      { n: 'destination', s: '"type": "string"', d: 'Destination airport IATA code, for example JFK' },
      { n: 'date', s: '"type": "string", "format": "date"', d: 'Outbound date in ISO 8601 format' },
    ], required: ['origin', 'destination', 'date'] },
  { name: 'book_hotel', description: 'Reserve a room for a date range',
    descLong: 'Reserve a hotel room for a date range and party size, returning a booking confirmation.',
    props: [
      { n: 'hotel_id', s: '"type": "string"', d: 'Unique identifier of the hotel to book' },
      { n: 'check_in', s: '"type": "string", "format": "date"', d: 'Check-in date for the stay' },
      { n: 'check_out', s: '"type": "string", "format": "date"', d: 'Check-out date for the stay' },
      { n: 'guests', s: '"type": "integer", "minimum": 1', d: 'Number of guests staying in the room' },
    ], required: ['hotel_id', 'check_in', 'check_out'] },
  { name: 'send_email', description: 'Send an email to a recipient',
    descLong: 'Compose and send an email message to one or more recipients with a subject and body.',
    props: [
      { n: 'to', s: '"type": "string", "format": "email"', d: 'Primary recipient email address' },
      { n: 'subject', s: '"type": "string"', d: 'Subject line of the email' },
      { n: 'body', s: '"type": "string"', d: 'Plain-text or HTML body content' },
    ], required: ['to', 'body'] },
  { name: 'create_calendar_event', description: 'Add an event to the calendar',
    descLong: 'Add a new event to the calendar with a title, start and end time, and optional attendees.',
    props: [
      { n: 'title', s: '"type": "string"', d: 'Title shown on the calendar event' },
      { n: 'start', s: '"type": "string", "format": "date-time"', d: 'Event start time' },
      { n: 'end', s: '"type": "string", "format": "date-time"', d: 'Event end time' },
    ], required: ['title', 'start'] },
  { name: 'search_web', description: 'Search the web for a query',
    descLong: 'Run a web search for the given query and return the most relevant ranked results.',
    props: [
      { n: 'query', s: '"type": "string"', d: 'Free-text search query' },
      { n: 'limit', s: '"type": "integer", "default": 10', d: 'Maximum number of results to return' },
    ], required: ['query'] },
  { name: 'read_file', description: 'Read the contents of a file',
    descLong: 'Read the full contents of a file from the workspace and return it as decoded text.',
    props: [
      { n: 'path', s: '"type": "string"', d: 'Absolute or workspace-relative file path' },
    ], required: ['path'] },
  { name: 'write_file', description: 'Write contents to a file',
    descLong: 'Write the given content to a file, creating it or overwriting any existing file.',
    props: [
      { n: 'path', s: '"type": "string"', d: 'Destination file path' },
      { n: 'content', s: '"type": "string"', d: 'Full content to write to the file' },
    ], required: ['path', 'content'] },
  { name: 'run_query', description: 'Run a SQL query against the database',
    descLong: 'Execute a read-only SQL query against the analytics database and return matching rows.',
    props: [
      { n: 'sql', s: '"type": "string"', d: 'SQL statement to execute' },
      { n: 'params', s: '"type": "array"', d: 'Positional query parameters' },
    ], required: ['sql'] },
  { name: 'send_slack_message', description: 'Post a message to a Slack channel',
    descLong: 'Post a message to a Slack channel or direct message as the connected workspace user.',
    props: [
      { n: 'channel', s: '"type": "string"', d: 'Channel ID or name, for example #general' },
      { n: 'text', s: '"type": "string"', d: 'Message text, supports Slack markdown' },
    ], required: ['channel', 'text'] },
  { name: 'list_contacts', description: 'List contacts matching a filter',
    descLong: 'List address-book contacts matching an optional filter, ordered alphabetically by name.',
    props: [
      { n: 'filter', s: '"type": "string"', d: 'Substring to match against name or email' },
      { n: 'limit', s: '"type": "integer", "default": 50', d: 'Maximum number of contacts to return' },
    ], required: [] },
  { name: 'get_stock_price', description: 'Get the latest price for a ticker',
    descLong: 'Get the latest traded price and daily change for a stock ticker symbol.',
    props: [
      { n: 'ticker', s: '"type": "string"', d: 'Stock ticker symbol, for example AAPL' },
      { n: 'currency', s: '"type": "string"', d: 'Currency to quote the price in' },
    ], required: ['ticker'] },
  { name: 'translate_text', description: 'Translate text to a target language',
    descLong: 'Translate a piece of text from its detected source language into a target language.',
    props: [
      { n: 'text', s: '"type": "string"', d: 'Source text to translate' },
      { n: 'target_lang', s: '"type": "string"', d: 'Target language code, for example fr' },
    ], required: ['text', 'target_lang'] },
  { name: 'summarize_document', description: 'Summarize a document',
    descLong: 'Produce a concise summary of a stored document, optionally bounded by a word count.',
    props: [
      { n: 'doc_id', s: '"type": "string"', d: 'Identifier of the document to summarize' },
      { n: 'max_words', s: '"type": "integer"', d: 'Approximate maximum length of the summary' },
    ], required: ['doc_id'] },
  { name: 'create_invoice', description: 'Create an invoice for a customer',
    descLong: 'Create and issue an invoice for a customer with a line-item total and due date.',
    props: [
      { n: 'customer_id', s: '"type": "string"', d: 'Identifier of the customer to bill' },
      { n: 'amount', s: '"type": "number"', d: 'Invoice total in the account currency' },
      { n: 'due_date', s: '"type": "string", "format": "date"', d: 'Date the invoice is due' },
    ], required: ['customer_id', 'amount'] },
  { name: 'schedule_meeting', description: 'Schedule a meeting with participants',
    descLong: 'Find a common free slot and schedule a meeting with the listed participants.',
    props: [
      { n: 'participants', s: '"type": "array"', d: 'Email addresses of the attendees' },
      { n: 'duration_mins', s: '"type": "integer"', d: 'Desired meeting length in minutes' },
    ], required: ['participants'] },
  { name: 'get_directions', description: 'Get directions between two points',
    descLong: 'Return turn-by-turn directions between two points for the chosen mode of travel.',
    props: [
      { n: 'origin', s: '"type": "string"', d: 'Starting address or coordinates' },
      { n: 'destination', s: '"type": "string"', d: 'Ending address or coordinates' },
      { n: 'mode', s: '"type": "string", "enum": ["drive", "walk", "transit"]', d: 'Mode of travel to use' },
    ], required: ['origin', 'destination'] },
  { name: 'convert_currency', description: 'Convert an amount between currencies',
    descLong: 'Convert a monetary amount from one currency to another at the latest exchange rate.',
    props: [
      { n: 'amount', s: '"type": "number"', d: 'Amount to convert' },
      { n: 'from', s: '"type": "string"', d: 'Source ISO 4217 currency code' },
      { n: 'to', s: '"type": "string"', d: 'Target ISO 4217 currency code' },
    ], required: ['amount', 'from', 'to'] },
  { name: 'fetch_news', description: 'Fetch recent news articles',
    descLong: 'Fetch recent news articles about a topic, optionally filtered to a start date.',
    props: [
      { n: 'topic', s: '"type": "string"', d: 'Topic or keyword to search news for' },
      { n: 'since', s: '"type": "string", "format": "date"', d: 'Only return articles after this date' },
    ], required: ['topic'] },
  { name: 'set_reminder', description: 'Set a reminder at a given time',
    descLong: 'Set a one-off reminder that notifies the user with a message at a specific time.',
    props: [
      { n: 'message', s: '"type": "string"', d: 'Reminder text to show the user' },
      { n: 'at', s: '"type": "string", "format": "date-time"', d: 'When the reminder should fire' },
    ], required: ['message', 'at'] },
];

function jsonNames(tools: Tool[]): string {
  return `{\n${tools.map((t) => `  "${t.name}"`).join(',\n')}\n}`;
}

function jsonFull(tools: Tool[], opts: { includeRequired?: boolean; verbose?: boolean } = {}): string {
  const { includeRequired, verbose } = opts;
  const body = tools.map((t) => {
    const props = t.props.map((p) => {
      const inner = verbose ? `{${p.s}, "description": "${p.d}"}` : `{${p.s}}`;
      return `        "${p.n}": ${inner}`;
    }).join(',\n');
    let schema =
`    "schema": {
      "type": "object",
      "properties": {
${props}
      }`;
    if (includeRequired && t.required.length) {
      schema += `,\n      "required": [${t.required.map((r) => `"${r}"`).join(', ')}]`;
    }
    schema += `\n    }`;
    const desc = verbose ? t.descLong : t.description;
    return `  "${t.name}": {\n    "description": "${desc}",\n${schema}\n  }`;
  }).join(',\n');
  return `{\n${body}\n}`;
}

// colourise raw JSON-ish text. spans use UNQUOTED attributes so later
// passes never match the quotes inside an already-inserted span.
function highlight(src: string): string {
  return src
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"([^"]+)"(\s*:)/g, '<span class=j-key>"$1"</span>$2')          // keys
    .replace(/:\s*"([^"]*)"/g, ': <span class=j-str>"$1"</span>')            // string values
    .replace(/:\s*(\d+)/g, ': <span class=j-num>$1</span>')                  // number values
    .replace(/(\[)([^\]]*?)(\])/g, (_m, a, body, c) =>
      a + body.replace(/"([^"]*)"/g, '<span class=j-str>"$1"</span>') + c)    // array string items
    .replace(/^(\s*)"([^"<]+)"(,?)$/gm, '$1<span class=j-key>"$2"</span>$3'); // bare names
}

type Frame = { text: string; fontSize: string; lineHeight: number; scroll?: boolean };

const FRAMES: Frame[] = [
  { text: jsonNames(TOOLS.slice(0, 3)), fontSize: '1.8rem', lineHeight: 1.6 },
  { text: jsonFull(TOOLS.slice(0, 3), { includeRequired: false }), fontSize: '0.58rem', lineHeight: 1.4 },
  { text: jsonFull(TOOLS, { includeRequired: true, verbose: true }), fontSize: '0.42rem', lineHeight: 1.4, scroll: true },
];

const wrap: React.CSSProperties = {
  position: 'relative',
  background: 'var(--surface)',
  border: '1px solid rgba(255,255,255,0.10)',
  borderRadius: '10px',
  padding: '0.8rem 1rem',
  height: '100%',
  boxSizing: 'border-box',
  overflow: 'hidden',
};

const preBase: React.CSSProperties = {
  margin: 0,
  // override deckx's default <pre> code-block styling (the "inner box")
  background: 'transparent',
  border: 'none',
  borderRadius: 0,
  padding: 0,
  fontFamily: 'var(--font-mono)',
  color: 'var(--color-text)',
  whiteSpace: 'pre',
};

const CSS = `
  .j-key { color: var(--accent-tertiary); }
  .j-str { color: var(--accent-aqua); }
  .j-num { color: var(--accent-secondary); }
  @keyframes toolscroll { from { transform: translateY(0); } to { transform: translateY(-50%); } }
  .tools-scroll { animation: toolscroll 15s linear infinite; }
`;

export default function ToolsJson() {
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === '.') setFrame((f) => Math.min(FRAMES.length - 1, f + 1));
      else if (e.key === ',') setFrame((f) => Math.max(0, f - 1));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const f = FRAMES[frame];
  const html = { __html: highlight(f.text) };

  return (
    <div style={wrap}>
      <style>{CSS}</style>
      {f.scroll ? (
        <div className="tools-scroll">
          <pre style={{ ...preBase, fontSize: f.fontSize, lineHeight: f.lineHeight }} dangerouslySetInnerHTML={html} />
          <pre style={{ ...preBase, fontSize: f.fontSize, lineHeight: f.lineHeight, paddingTop: '1.2rem' }} dangerouslySetInnerHTML={html} />
        </div>
      ) : (
        <pre style={{ ...preBase, fontSize: f.fontSize, lineHeight: f.lineHeight }} dangerouslySetInnerHTML={html} />
      )}
    </div>
  );
}
