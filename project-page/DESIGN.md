# Instrument Noir — VectorWorld Design System

A single, named design language for the VectorWorld project page and every page
that follows it. The goal is "engineering + academic" credibility with top-tier
restraint (Apple / Linear / Vercel): near-black canvas, one instrument-cyan
accent, two type voices, a vector-graph signature, and quiet, layered motion.

> **One rule above all: restraint.** A signature is something you keep doing in
> _one_ place, not everywhere. One count-up. One quiet hero motif. One accent.
> When in doubt, make it neutral and let the content speak.

The entire system is driven by CSS custom properties in
[`src/styles/global.css`](src/styles/global.css) plus a set of `vw-*` component
classes. Change a token in one place and the whole site — and any future page —
inherits it. This file is the contract.

---

## 1. Narrative principle — form mirrors method

VectorWorld's method is _warm start → one-step frontier completion → closed-loop
rollout_. The page is laid out as a **rollout** so the structure teaches the
method:

| Page region | Method stage | Eyebrow / index |
| ----------- | ------------ | --------------- |
| Hero        | Warm start   | `STREAMING WORLD MODEL · ICML 2026 · SPOTLIGHT` |
| Method      | Frontier completion | `01 … 02 …` |
| Efficiency  | Deployment budget | `03` |
| Results / rollout | Closed loop | `04` |

Sections are separated by a **connector** (`.vw-connector`) — a faint vertical
lane with a node, echoing the vector-graph motif and visually "wiring" one stage
to the next.

---

## 2. Color — one accent, three iron rules

The brand is **instrument cyan** — it reads as LiDAR / HUD / sensor
visualization, which is on-topic and differentiated from the sea of SaaS
blue/violet.

**Iron rules for the accent (`--vw-accent`):**

1. **Highest-priority CTA only** — the arXiv button (`.vw-button-primary`).
2. **State** — active / hover / focus, scrollspy underline, selected toggles.
3. **Key delta numbers** — e.g. the `25.7% → 56.0%` arrow and figure deltas
   (`.vw-accent-text`).

Everything else is neutral ink and surface. Decorative blue/violet/emerald/
rose/amber are **retired**. The only non-accent hues that survive are
**legitimate data colors** inside the 3D viewer legend (ego / vehicle /
pedestrian / cyclist / lane-link) and semantic error red on the always-dark
viewer canvas.

### Light ↔ dark are equal, not inverted

Light mode is fully tuned (not a dark inversion). The accent **drops to
`#0891B2` on light** to hold WCAG AA on white.

| Token | Light | Dark / device-dark |
| ----- | ----- | ------------------ |
| `--vw-bg` | `#FBFBFD` | `#0A0B0D` |
| `--vw-surface-1` | `#FFFFFF` | `#121419` |
| `--vw-surface-2` | `#F4F5F7` | `#1A1D24` |
| `--vw-surface-3` | `#ECEEF1` | `#21252E` |
| `--vw-hairline` | `rgba(10,11,13,.10)` | `rgba(255,255,255,.08)` |
| `--vw-hairline-strong` | `rgba(10,11,13,.18)` | `rgba(255,255,255,.16)` |
| `--vw-text-1` | `#0A0B0D` | `#F4F6F8` |
| `--vw-text-2` | `#51565E` | `#A3AAB5` |
| `--vw-text-3` | `#8A9099` | `#6B7280` |
| `--vw-accent` | `#0891B2` | `#22D3EE` |
| `--vw-accent-strong` | `#0E7490` | `#67E8F9` |
| `--vw-accent-soft` | `rgba(8,145,178,.10)` | `rgba(34,211,238,.12)` |
| `--vw-accent-line` | `rgba(8,145,178,.32)` | `rgba(34,211,238,.30)` |
| `--vw-accent-contrast` | `#FFFFFF` | `#05141A` |
| `--vw-canvas` (viewer / media) | `#0A0B0D` | `#06080B` |

Themes are selected by `data-theme="light" | "dark" | "device"` on `<html>`.
The `dark` Tailwind variant keys on **both** `[data-theme="dark"]` and
`[data-theme="device"]` under `prefers-color-scheme: dark`, so neutral
`zinc` utilities still resolve correctly in the default `device` theme.

---

## 3. Typography — two voices, never three

| Voice | Family | Token | Used for |
| ----- | ------ | ----- | -------- |
| **Sans** (body / display) | Geist Variable → Inter → system | `--font-sans` | UI, prose, display titles (weights 300–600, never 700-heavy display) |
| **Mono** (technical) | JetBrains Mono Variable → ui-monospace | `--font-mono` | Eyebrows, section indices, code, all numbers |

Rules:

- **No serif voice.** Two voices stay clean and "engineering"; a third needs
  editorial discipline we don't want to maintain.
- Eyebrows are mono, uppercase, `letter-spacing: var(--eyebrow-tracking)`
  (`0.16em`) via `.vw-eyebrow`.
- **All metrics use `font-variant-numeric: tabular-nums`** (`.vw-num`,
  `.vw-stat-value`) so digits don't jump.
- Display size is fluid: `--display: clamp(2.75rem, 6vw, 5.25rem)`.

Fonts are **self-hosted** (`@fontsource-variable/geist`,
`@fontsource-variable/jetbrains-mono`) and `@import`-ed in `global.css`. Do not
reintroduce a network font CDN — the build environment has no access to it.

---

## 4. Foundational tokens

```text
Motion   --ease cubic-bezier(.22,1,.36,1)   --ease-emphasized cubic-bezier(.16,1,.3,1)
         --dur-fast .18s   --dur-base .4s   --dur-slow .7s          (one curve, three speeds)
Spacing  --space-1 .5rem  -2 1rem  -3 1.5rem  -4 2rem  -6 3rem  -8 4rem  -12 6rem
Radii    --r-xs .5rem  --r-btn .75rem  --r-card 1.25rem  --r-lg 1.75rem  --r-pill 9999px
```

These map onto Tailwind via `@theme inline`, so you can use utilities directly:

| Utility | Resolves to |
| ------- | ----------- |
| `bg-brand` / `text-brand` / `border-brand` | `--vw-accent` |
| `bg-brand-soft` | `--vw-accent-soft` |
| `text-brand-contrast` | `--vw-accent-contrast` |
| `text-ink-1` / `-2` / `-3` | `--vw-text-1/2/3` |
| `bg-surface-1` / `-2` / `-3` | `--vw-surface-1/2/3` |
| `border-hairline` / `border-hairline-strong` | hairline tokens |
| `rounded-xl` | `--r-card` (20px) |

Raw vars are also available for one-offs:
`var(--vw-accent)`, `var(--vw-accent-line)`, `var(--vw-accent-strong)`,
`var(--vw-canvas)`, `var(--vw-hairline-strong)`.

---

## 5. Surfaces & elevation — hairline, never black shadow

In dark mode a drop shadow is invisible and dirty. Elevation is expressed with
**1px hairline borders + a two-step surface scale**, not shadow.

- Base card: `1px solid var(--vw-hairline)` on `--vw-surface-1`.
- Nested / inset: `--vw-surface-2`.
- Hover: `translateY(-2px)` + border brightens to `--vw-accent-line`,
  transition `--dur-base var(--ease)`.
- Cards share **one radius**: `--r-card` (20px) via `.vw-card` / `rounded-xl`.

---

## 6. Motion grammar — three layers, one curve, fire once

All motion uses the **single `--ease`**. Displacement stays ≤ 20px.

| Layer | What | How |
| ----- | ---- | --- |
| **Ambient** | Hero motif, accent aurora | Slow, looping, very low contrast |
| **Reveal** | Scroll-in of content | `opacity 0→1`, `translateY 16px→0`, `--dur-slow`, **once**, optional stagger |
| **Feedback** | Buttons, copy, toggles | `--dur-fast`; `active:scale(.98)`; instant |

Reveal is wired with data attributes (script in
[`src/pages/index.astro`](src/pages/index.astro)):

- `data-reveal` — animate this element in once when it enters the viewport.
- `data-reveal-delay="120"` — per-element delay (ms).
- `data-reveal-group` — auto-stagger direct children
  (`data-reveal-stagger`, default `70`ms).

The single page count-up is opt-in and used **exactly once** (the `56.0%`
policy-success number):

- `data-countup="56.0"` `data-countup-decimals="1"`
  `data-countup-suffix="%"` (`-prefix` also supported).
- Server-renders the final value, so it is correct with JS off.

**Reduced motion**: `@media (prefers-reduced-motion: reduce)` zeroes all
animation/transition durations globally, forces `.vw-reveal` visible, and stops
the hero canvas. Count-up shows the final value immediately. Honor it — never
ship motion that can't degrade.

---

## 7. Brand signature — the vector-graph motif

The signature is a **node-edge / lane** motif drawn in the accent, used as a
through-line (not just a hero background):

- **Mark / favicon** — a single diamond node + edge in the accent.
- **`.vw-connector`** — faint lane + node between major sections.
- **`.vw-section-index`** — mono `01 / 02 / 03…` markers on section headings.
- **Hero** — a quiet lane-and-agent vector field
  ([`HeroVectorField.astro`](src/components/vectorworld/HeroVectorField.astro)),
  link opacity `~0.05–0.08`, nodes drifting slowly, parallax ≤ 8px, static
  under reduced motion. **Not** a busy particle field.
- **OG image** (`public/og-cover.png`, 1200×630) — a LiDAR sweep + diamond
  nodes in the same language so shares look on-brand.

The **3D vector-scene viewer is the centerpiece**, not a footnote: its own
stage, an accent skeleton loader (pulsing diamond node), and **segmented layer
toggles** (`aria-pressed`, accent when on) rather than plain checkboxes.

---

## 8. Component classes (`vw-*`)

Reach for these before writing new CSS. Defined in `global.css`.

| Class | Purpose |
| ----- | ------- |
| `vw-card`, `vw-card-muted`, `vw-card-interactive` | Hairline surface cards (hover lift, no dark shadow) |
| `vw-surface`, `vw-surface-muted` | Flat hairline panels |
| `vw-button-primary` | The single accent CTA (arXiv) |
| `vw-button-secondary` | Neutral pill button (accent flash on `data-copied`) |
| `vw-pill` | Small neutral pill / chip control |
| `vw-eyebrow` (`--plain`) | Mono uppercase eyebrow (node dot prefix) |
| `vw-section-index` | Mono `01/02` section number |
| `vw-section`, `vw-connector` | Section wrapper + node connector |
| `vw-lead`, `vw-card-title`, `vw-card-copy` | Type roles |
| `vw-stat-label`, `vw-stat-value`, `vw-stat-copy`, `vw-num` | Dashboard metrics (tabular nums) |
| `vw-accent-text` | Inline accent number / delta |
| `vw-anchor*` | Glass scrollspy nav + sliding indicator |
| `vw-progress` | Top reading-progress bar |
| `vw-hero-*` | Hero shell, canvas, wordmark, highlight cards |
| `vw-segmented` | Segmented control |
| `vw-media-frame(--wide/--square/--rollout/--portrait)` | Consistent media frames over `--vw-canvas` |
| `vw-reveal` | Reveal animation target (toggled by the reveal script) |
| `vw-skeleton` | Shimmer placeholder (theme surface bg) |

---

## 9. Accessibility & sustainability

- **Light/dark parity** — both fully tuned; accent steps down to `#0891B2` on
  light for AA.
- **Contrast** — text tokens target WCAG AA on their surfaces.
- **Focus** — global `:focus-visible` accent ring (2px, 2px offset).
- **Targets** — interactive controls ≥ 44px.
- **Reduced motion** — global guard (see §6).
- **Print** — `@media print` hides nav / progress / hero canvas, forces a clean
  light document with hairline card borders (reviewers print).
- **Selection** — uses `--vw-accent-soft`.

---

## 10. Reusing this system for a new page

The whole point of the token system: a new paper page inherits the brand for
free. Checklist:

1. Render under `<html data-theme="…">` and `import "../styles/global.css"`.
2. Compose from `vw-*` classes and the `brand` / `ink` / `surface` utilities —
   **do not** hardcode hex colors or shadows.
3. Use mono eyebrows + `vw-section-index`; keep the rollout narrative order.
4. Accent appears only in the three places from §2. Neutral everywhere else.
5. One count-up per page, max. Quiet ambient motion only.
6. Add `<meta>` OG/Twitter/canonical (already wired in `index.astro`, gated on
   `Astro.site`) and a 1200×630 OG image in the page's language.
7. Verify: `npm run build` (runs `astro check` + `astro build`) is green;
   `prefers-reduced-motion` degrades; light and dark both look intentional.

## 11. Assets & infra

- `public/og-cover.png` — 1200×630 Instrument-Noir share card. Regenerate from
  the `sharp`-based SVG script if the title/metrics change.
- `public/favicon.svg` — single-accent diamond-node mark.
- `@astrojs/sitemap` is enabled; it emits `sitemap-index.xml` when `site` is set
  (injected in CI). `site` / `base` are **never** hardcoded in `astro.config.ts`.
- Fonts are self-hosted; no font CDN at build time.
