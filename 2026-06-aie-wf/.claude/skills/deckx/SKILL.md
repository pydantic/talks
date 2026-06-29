---
name: deckx
description: Create a deck with deckx. Use when the user mentions "deckx", "deck" or "slides", asks to build a slide deck from MDX, asks to convert a brand palette into a deck stylesheet, or asks how to convert a deckx HTML deck into a PDF. Covers project layout, deckx.toml config, deck.mdx authoring, custom React components, the styles.css token contract, and the Chrome headless PDF command.
---

# deckx

`deckx` builds a single, self-contained HTML slide deck from one MDX file plus a CSS theme and an optional folder of React components. The HTML is print-ready and converts to PDF via Chrome headless.

## Installation

```bash
bun init -y  # if you don't already have a package.json initialized
bun add @samuelcolvin/deckx
```

The npm package is `@samuelcolvin/deckx`; the installed CLI binary is `deckx`. `npm i` / `pnpm add` work the same way. Inside `deck.mdx` you always import from `"deckx"` (a Vite alias, not the npm name).

## Project layout

```
my-deck/
├── deckx.toml          # config: title, theme, tabs, footer, paths (all optional)
├── deck.mdx            # the slides
├── styles.css          # CSS variable overrides (optional)
└── components/         # optional custom React/TSX components
    └── Hello.tsx
```

```bash
bunx deckx dev          # dev server with HMR on http://localhost:5173/
bunx deckx html         # build to ./dist/index.html
bunx deckx pdf          # build HTML, then ./dist/deck.pdf via Chrome headless
```

`html` and `pdf` accept an optional output-path positional - e.g. `bunx deckx pdf my-deck.pdf` or `bunx deckx html out/slides.html`. Use `--dir <dir>` to point at a build directory other than the current one. To convert an existing HTML file to PDF without rebuilding, use `bunx deckx html-to-pdf <input.html> <output.pdf>`.

## `deckx.toml`

All fields are optional - a deck with only `deck.mdx` works.

```toml
title = "My Deck - April 2026"        # browser tab title

# light | dark | markdown-light | markdown-dark   (default: light)
# "markdown-*" variants render source-style decorations on top:
# heading "#"/"##" prefixes, "**" strong markers, traffic-light dots,
# mono slide counter, diamond bullets.
theme = "light"

# Small footer rendered bottom-right of every slide.
footer = "Confidential - do not share"

# Path to a favicon for the browser tab. .svg / .png / .ico / .jpg.
# Inlined as a data URI so the deck stays self-contained.
favicon = "assets/favicon.svg"

# Path overrides (defaults shown).
mdx = "deck.mdx"
styles = "styles.css"
components = "components"

# Optional tab nav. When present, <Slide tab="..."> highlights the matching tab.
tabs = [
  { id = "intro", label = "Intro" },
  { id = "details", label = "Details" },
]
```

If `tabs` is omitted, the `tab` prop on `<Slide>` is ignored and slides render with a plain title topbar.

## `deck.mdx`

Standard MDX. Import `Slide` from `"deckx"`, plus any custom components from `./components/`.

```mdx
import { Slide } from "deckx";
import Hello from "./components/Hello.tsx";

<Slide layout="title" title="Investor Deck / April 2026">

### Section Label

# My Deck

## A subtitle

</Slide>

<Slide tab="intro">

# Hello world

- Bullet one
- Bullet two

<Hello name="friend" />

</Slide>

<Slide layout="statement">

# One big idea.

</Slide>
```

**Do NOT wrap slides in `<div className="deck">`** - deckx adds it for you.

### `<Slide>` props

All props are optional. A bare `<Slide>` renders a regular content slide using the deck theme.

#### `layout` - structural layout (default `'content'`)

Picks how the slide arranges its body. Adds a `.<value>-slide` class to the `.slide` element so you can target each variant from `styles.css`.

- `'content'` (default) - regular slide. Headings, paragraphs, bullets, code, and tables flow top-down inside `.slide-body`. Use for the bulk of your deck.
- `'title'` - cover / section slide. Bottom-aligns the hero; h1 is 4rem with tight letter-spacing; h2 renders in `--accent`. Pair with a leading `### Section Label` for a mono uppercase eyebrow.
- `'statement'` - centered one-liner. The body is centered both vertically and horizontally; h1 is 3.4rem; paragraphs cap at 80% width. Use for transitions between sections or "one bold idea" beats.

#### `theme` - color variant (defaults to the deck theme)

Per-slide palette override.

- Omit (or pass `'dark'`) - the slide inherits the deck-level `theme` from `deckx.toml`.
- `'light'` - forces a single slide onto the light palette (`--bg-light`, `--color-text-light`, `--color-heading-light`) regardless of the deck theme. Useful when one slide needs to break out - e.g. a screenshot of a light-themed UI on an otherwise dark deck. Adds `.light-slide` to the slide.

There is no inverse override: on a light deck, `theme="dark"` has no effect. If you need a single dark slide on a light deck, target it from CSS with a custom `id` or class.

#### Other props

- `tab` - string matching an `id` from the `tabs` array in `deckx.toml`. Replaces the plain topbar title with the tab bar, with this slide's tab highlighted. Clicking any tab in any slide jumps to the first slide whose `tab` matches. If `tabs` is not configured, the prop is silently ignored.
- `title` - plain text rendered in the topbar when `tab` is not set. Also drives `document.title`, so the browser tab updates as the active slide changes. Ignored when `tab` is set.
- `space` - `'tight'` reduces bullet/paragraph spacing (use when a slide is close to overflowing); `'wide'` increases padding and line-height (use for slides with very little text where you want generous breathing room).
- `fontSize` - `'large'` bumps body text from 1.15rem to 1.35rem, and h1/h2 proportionally. Useful for slides that need to read from the back of a room.
- `id` - sets the HTML `id` on the underlying `<section>`. Useful for targeting one slide from `styles.css` (`.slide#hero { ... }`) without adding a custom class.

### MDX gotchas

- Use `-` (hyphen-minus), never `—` (em dash).
- Keep blank lines around block elements inside a slide, but **not** between the last block and the closing `</Slide>` - MDX misparses a trailing blank.
- `### Section Label` at the top of a slide renders as a diamond + uppercase mono label.
- Slides must fit **279.4mm × 157.2mm** (16:9). If overflowing, try `space="tight"` first, then drop content.

### Images

Place images in your project (e.g. `assets/`) and import as ES modules:

```mdx
import logo from "./assets/logo.png";
<img src={logo} alt="Logo" />
```

Vite inlines them into the final HTML. The markdown `![alt](path)` syntax does **not** get inlined - always use `<img src={imported} />`.

### Code blocks

Fenced code blocks are syntax-highlighted at build time by [Shiki](https://shiki.style). Tag the fence with a language so tokens get coloured:

````mdx
```ts
export function greet(name: string): string {
  return `hello, ${name}`;
}
```
````

All Shiki bundled languages work without any per-deck configuration. Highlighting happens during the build, so the output HTML carries only inline-styled `<span>`s - no grammar files, no runtime highlighter.

Both a light and a dark theme are emitted into every code block. deckx switches between them automatically based on the slide's effective theme: dark deck themes (`dark`, `markdown-dark`) use the dark code theme, and `<Slide theme="light">` always shows the light code theme even inside a dark deck. Pick the two themes in `deckx.toml`:

```toml
code_light_theme = "github-light"   # default
code_dark_theme  = "github-dark"    # default
```

Browse the available themes at <https://shiki.style/themes>.

## Custom components

Any `.tsx` file in `components/` (or wherever `deckx.toml` `components` points) can be imported into `deck.mdx` with a relative path:

```tsx
// components/Hello.tsx
export default function Hello({ name }: { name: string }) {
  return <p>Hello, {name}!</p>;
}
```

```mdx
import Hello from "./components/Hello.tsx";
<Hello name="world" />
```

React 19 is available. Components see the same CSS variables your `styles.css` defines, so to stay on-brand, read from variables (`color: var(--accent)`) rather than hard-coding colors.

## Authoring `styles.css`

The base stylesheet handles all layout, typography, slide dimensions, the topbar, transitions, and the PDF `@page` setup. `styles.css` only needs to override CSS variables on `:root` to set brand tokens.

### Variable contract

Backgrounds:

- `--bg-deck` (default `#0d0d0d`) - background outside the slide, presenter mode only.
- `--bg-slide` (default `#1a1a1a`) - default slide background.
- `--bg-light` (default `#ffffff`) - slide bg for `light` / `markdown-light` decks and `<Slide theme="light">`.
- `--surface` (default `#2a2a2a`) - inline code background, table headers.

Text:

- `--color-text` (default white @ 85%) - body text on dark slides.
- `--color-heading` (default `#ffffff`) - h1, h2, h4, strong on dark slides.
- `--color-muted` (default `#8f888e`) - heading prefixes, slide counter, subdued UI.
- `--color-text-light` (default `#2a2230`) - body text on light slides.
- `--color-heading-light` (default `#1a1018`) - headings on light slides.

Accents:

- `--accent` (default `#4a9eff`) - primary accent: bullets, h3, links, blockquote bar.
- `--accent-secondary` (default `#ff6b6b`) - em, link hover.
- `--accent-tertiary` (default `#b388ff`) - hr gradient stop.
- `--accent-aqua` (default `#4ad7c5`) - inline code text, active tab, topbar tabs.

Fonts:

- `--font-body` (default system sans stack) - body and headings, unless `--font-heading` overrides.
- `--font-heading` (default inherits body) - headings.
- `--font-mono` (default system mono) - inline code, code blocks, tabs, counter, h3.
- `--font-terminal` (default inherits body) - body inside `.slide-body`.

### CSS class hooks

For finer control beyond the variable contract, target these classes from `styles.css`. Most decks won't need them - prefer overriding variables first.

Slide structure:

- `.deck-presenter` - outermost wrapper. Owns the viewport background and the fit-to-window scaling transform.
- `.deck` - inner slide stream (direct child of `.deck-presenter`).
- `.slide` - a single slide (`<section>`). Sized 16:9, holds topbar + content + optional footer.
- `.slide-topbar` - 48px window-chrome bar at the top of every slide.
- `.slide-content` - padded body wrapper below the topbar (this is what `--slide-padding` applies to).
- `.slide-body` - inner MDX content container, descendant of `.slide-content`.
- `.slide-footer` - bottom-right footer text, rendered when `footer` is set in `deckx.toml`.

Slide modifiers (added to `.slide` based on props):

- `.title-slide` - `layout="title"`, bottom-aligned hero.
- `.statement-slide` - `layout="statement"`, centered hero.
- `.light-slide` - `theme="light"`, forces the light palette.
- `.space-tight` / `.space-wide` - vertical spacing density.
- `.font-large` - bumps body text size.

Topbar - left (traffic lights + title/tabs):

- `.topbar-dots` - the traffic-light anchor (clicking jumps to slide 1). Only visible under `markdown-*` themes.
- `.topbar-dot` plus `.topbar-dot--red` / `.topbar-dot--yellow` / `.topbar-dot--green` - individual dots.
- `.topbar-title` - plain title text, shown when `<Slide title="...">` is set without a `tab`.

Topbar - tabs (rendered when `<Slide tab="...">` is set and `tabs` are configured in `deckx.toml`):

- `.topbar-tabs` - the tab bar container.
- `.topbar-tab-group` - per-tab wrapper containing the link plus its leading separator.
- `.topbar-tab-sep` - the `→` glyph between tabs.
- `.topbar-tab` - the tab link.
- `.topbar-tab--active` - applied to the currently-selected tab.

Topbar - right (prev/next + counter):

- `.topbar-nav` - container holding the prev/next buttons and slide counter.
- `.topbar-nav-btn` plus `.topbar-nav-prev` / `.topbar-nav-next` - the nav buttons (auto-hidden in print).
- `.topbar-nav-counter` - the `01/12` slide counter.

Deck-level theme classes (applied to both `<html>` and `.deck-presenter` based on `theme` in `deckx.toml`):

- `.theme-light` / `.theme-dark` / `.theme-markdown-light` / `.theme-markdown-dark`

MDX content inside `.slide-body` renders as plain HTML (`h1`-`h4`, `p`, `ul`, `ol`, `pre`, `code`, `table`, `blockquote`, `a`, `img`, `hr`) - target those tags directly with `.slide <tag>` selectors rather than expecting deckx to add wrapper classes.

### Mapping a brand palette

1. Pick the **most distinctive** brand color, assign to `--accent`. Pick a warm counterpoint as `--accent-secondary`.
2. Pick a slightly off-white for `--color-heading` (pure white reads sterile under projector light).
3. Pick a tinted dark for `--bg-slide` (pure black is harsh).
4. For light slides, pick a tinted light bg (cream, eggshell, lavender - not pure white) plus a near-black text color → `--bg-light` / `--color-heading-light` / `--color-text-light`.
5. For custom fonts, self-host woff2 files in `assets/` and declare them with `@font-face` in `styles.css`, then point `--font-body` / `--font-mono` at the family. Do **not** use `@import url(...)` from Google Fonts - CSS spec requires @import to come before all other statements, which Vite's CSS bundling routinely violates when concatenating the base stylesheet with yours. `@font-face` can appear anywhere. Use [Google Webfonts Helper](https://gwfh.mranftl.com/fonts) to download woff2 files; Vite inlines them into the deck, keeping it self-contained for offline / PDF export.

```css
@font-face {
  font-family: 'YourFont';
  src: url('./assets/YourFont-Regular.woff2') format('woff2');
  font-weight: 400;
  font-display: swap;
}

:root {
  --bg-slide: #...;       /* tinted dark */
  --bg-light: #...;       /* tinted light */
  --surface: #...;

  --color-heading: #...;  /* off-white */
  --color-text: rgba(...);
  --color-heading-light: #...;
  --color-text-light: #...;

  --accent: #...;            /* signature brand color */
  --accent-secondary: #...;  /* warm counterpoint */

  --font-body: 'YourFont', system-ui, sans-serif;
  --font-mono: 'YourFontMono', ui-monospace, monospace;
}
```

## Building & PDF

`bunx deckx pdf` is the easy path: it builds the HTML, prints the exact Chrome command it's about to run, then runs it. Output lands at `./dist/deck.pdf`.

If Chrome / Chromium can't be found, copy the printed command and run it yourself with the right binary path. On Linux deckx auto-detects `google-chrome`, `google-chrome-stable`, `chromium`, or `chromium-browser`.

Paper size in the printed command matches the slide dimensions (11in × 6.1875in = 16:9). If you override `--slide-width` / `--slide-height` in `styles.css`, edit the `--paper-*` flags to match before running.

To spot-check the PDF (requires `pdftoppm` from poppler):

```bash
mkdir -p ./tmp && pdftoppm -r 100 ./dist/deck.pdf ./tmp/page -png
```

One PNG per slide lands in `./tmp/`, gitignore that path.
