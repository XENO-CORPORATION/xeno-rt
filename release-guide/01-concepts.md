# 01 — Concepts: how a XENO product lives on the platform

> Purpose: the mental model you need before you release anything — the four layers that make up a product, the two independent publishing flows (and why the split matters), how each delivery type shapes the release and the call-to-action, and the URLs a product owns.

---

## The one idea to hold onto

A product on the XENO platform is **not a single page you deploy**. It is **four independent layers joined by one string — the `<slug>`** — and those layers are updated by **two completely separate flows** that run on different cadences and touch different systems.

- **Release data** (new versions, patches, downloads) is read **live** by the website → it appears with **no platform deploy**.
- **Everything else** (catalog identity, landing content, documentation) is **compiled and prerendered at build time** → changing it **requires a platform rebuild + deploy**.

Get this split right and releasing becomes routine: shipping `v1.4.2` never touches the platform, while a marketing refresh is a deliberate, infrequent deploy. Get it wrong and you either deploy for every patch (needless) or expect a copy change to appear without a build (it never will).

---

## The four layers (all joined by `<slug>`)

| Layer | Lives in | Changes when | Needs a deploy? |
|---|---|---|---|
| **Identity** (catalog entry) | `src/lib/productCatalog.ts` — one `Product` entry | product added or re-classified | **Yes** — build + deploy |
| **Release data** | R2: `apps/<slug>/releases.json` (+ derived `version.json`) | **every release** (published by `scripts/xeno-release.mjs`) | **No** — read live |
| **Landing content** | `src/content/products/<slug>.ts` + `src/components/product/mockups/` + `public/product-assets/<slug>/` | marketing / design changes (rare) | **Yes** — build + deploy |
| **Documentation** | `src/content/docs/<slug>.ts` (registered in `src/content/docs/index.ts`) | docs authored / updated | **Yes** — build + deploy |

> The above lives in the **`xeno-platform`** repo (except release data, which lives on R2). This guide is portable into any product repo; the *paths* are canonical to `xeno-platform`.

### The slug is the join key — it must be identical everywhere

The same `<slug>` names all four layers: the catalog entry, the R2 folder `apps/<slug>/`, the content module `src/content/products/<slug>.ts`, and the docs module `src/content/docs/<slug>.ts`. It also appears in every URL the product owns. If the slug drifts between layers, the site silently falls back (lean page, empty feed, or a 404 on the download deep-link).

- The backend validates slugs against `/^[a-z0-9-]+$/` (see `src/server/routes/productDownloadRoutes.js`).
- Semver everywhere is written **without** a leading `v` (e.g. `1.4.2`, not `v1.4.2`).
- **R2-folder override:** the catalog entry may set an optional `r2` field to point at a different R2 folder; the effective folder is `p.r2 ?? p.slug`. Most products leave it unset (folder = slug); `extension` sets `r2: 'extension'`. When in doubt, confirm the entry in `src/lib/productCatalog.ts`.

---

## The two flows, and why the split matters

### Flow A — Release data (LIVE, no deploy)

The product's own repo publishes a release straight to R2. The website reads it on page load; nothing on the platform is rebuilt.

```
product repo ──► scripts/xeno-release.mjs publish ──► R2  apps/<slug>/releases.json
                                                          apps/<slug>/version.json
                                                          apps/<slug>/v<version>/<installer>
                       │
website / Hub ─────────┘  read live (no cache): new version, patch, and download link appear instantly
```

- The website fetches `https://updates.xenostudio.ai/apps/<slug>/releases.json` at page load (`fetchReleases`, `cache: 'no-cache'`), and the backend download deep-link caches it for only 30s.
- Both JSON files are uploaded with `Cache-Control: no-cache`; installers under `v<version>/` are immutable.
- **Result:** a new version, a hotfix, or a new OS installer shows up on `xenostudio.ai` and in Hub's auto-update **without any platform deploy**.

Mechanics, schemas (`releases.json` / `version.json`), and the exact publisher commands are covered in **`03-release-data.md`**. The canonical spec is `PRODUCT-PAGES-SPEC.md` and the operator how-to is `RELEASE-TO-WEBSITE.md` (both in `xeno-platform`).

### Flow B — Content & identity (COMPILED, needs build + deploy)

Catalog identity, landing content, and documentation are TypeScript modules compiled into the SPA bundle and **prerendered to static SEO HTML** at build time by `scripts/prerender-products.mjs` (which runs as the second half of `npm run build` = `vite build && node scripts/prerender-products.mjs`). Because they are baked into the build, changing any of them requires rebuilding and redeploying the frontend.

```
edit  src/lib/productCatalog.ts / src/content/products/<slug>.ts / src/content/docs/<slug>.ts
   │
   ▼  npm run build   (vite build → dist, then prerender-products.mjs → static <head> per route + sitemap.xml)
   │
   ▼  deploy to xeno-platform-001  (docker compose build frontend && up -d frontend  → xenostudio-frontend)
   │
   ▼  live on xenostudio.ai
```

The full build + on-box deploy procedure (the `git archive … | ssh xeno-platform-001` transfer into `/mnt/projects/xeno-platform`, the CRLF-normalize-text-files-only rule, `sudo docker compose build frontend && up -d frontend`, and verification) is the subject of the **deploy chapter** of this guide; the canonical source is `PRODUCT-LANDING-SPEC.md` §8–§9 in `xeno-platform`.

### Why the split exists

- **Different cadence.** Releases happen constantly (every version bump); content and docs change rarely. Coupling releases to deploys would mean a platform rebuild for every patch — slow and risky. Coupling content to a live fetch would sacrifice SEO.
- **Different criticality.** Landing/docs pages need a crawlable, route-correct `<head>` (title, canonical, Open Graph, `SoftwareApplication` / `TechArticle` JSON-LD). That can only be produced by the prerender at build time — hence the deploy. Release data is dynamic operational metadata; live-fetch is exactly right for it.
- **Practical consequence:**
  - Routine version bump / patch / new installer → **Flow A only** (no deploy).
  - Marketing copy, new mockup, new/updated docs page, or a catalog re-classification → **Flow B** (deploy).
  - **New product** → both: add the catalog identity (Flow B, deploy) + cut the first release (Flow A, live), plus optionally author landing content and docs (Flow B).

---

## Delivery types — how each shapes the release and the CTA

A product's `delivery` field in `src/lib/productCatalog.ts` is one of `desktop | web | cli | soon`, and it drives **both** how you release and which call-to-action the landing page renders. (`status` is separate — `shipping | beta | coming-soon` — and controls SEO indexing; see the prerender notes below.)

| `delivery` | How it releases | Landing CTA | Owns download/releases pages? |
|---|---|---|---|
| **desktop** | Installers uploaded to R2 under `apps/<slug>/v<version>/`; `releases.json` carries `assets.{windows,mac,linux}`; `version.json` is installer-shaped (filenames only, Hub auto-update reads it). Published via `scripts/xeno-release.mjs` with `--win/--mac/--linux`. | **Download** (deep-link) + **Releases** | Yes |
| **web** | No installer. The app launches in-browser; release data typically carries **no** `assets` (omitting `assets` = non-downloadable, which is valid). | **Open** (the catalog `launchPath`) | No |
| **cli** | Installed via `npm install -g <pkg>` — **no installer**. The releases feed is derived from **npm** (versions + publish dates) intersected with the CLI's own `RELEASE_NOTES` map, via `scripts/publish-cli-releases.mjs`. `version.json` is npm-shaped (`npm` + `install` fields, no OS keys). A CLI release can alternatively be published with `xeno-release.mjs` and no `--win/--mac/--linux`. | **Install** (the `install` command) + **Releases** | Yes |
| **soon** | Nothing to release yet — placeholder. | **Waitlist** | No |

Notes tied to `delivery` / `status` that affect what actually gets published:

- **Download deep-link is delivery-gated but version-stable.** For `desktop` (and `cli`) the landing renders `/product/<slug>/download/<os>` — a backend **302** that always resolves to the current primary installer. The link itself **never changes across versions** (`downloadLink()` in `src/lib/productCatalog.ts`); the redirect target updates as new releases land. That is what lets Flow A change downloads with no deploy.
- **Prerender is gated on both fields** (`scripts/prerender-products.mjs`): `/product/<slug>` is emitted only when `status !== 'coming-soon'` (coming-soon products are skipped); the `/download` and `/releases` static pages are emitted only when `delivery === 'desktop' || delivery === 'cli'`. A `soon` product is usually `coming-soon` and therefore not indexed.

---

## The URLs a product owns, and how they cross-link

All under `https://xenostudio.ai` unless noted. Note the site uses **singular `/product/<slug>`** for a product; **plural `/products`** is the grid index of all products.

| URL | What it is | Applies to |
|---|---|---|
| `https://xenostudio.ai/product/<slug>` | Landing page (rich `ProductLanding` if a content module exists, else the lean fallback) | all (except `coming-soon`, which isn't indexed) |
| `https://xenostudio.ai/product/<slug>/download` | Download page (chooser + latest version, read live from R2) | `desktop`, `cli` |
| `https://xenostudio.ai/product/<slug>/releases` | Full release history (renders `releases.json`) | `desktop`, `cli` |
| `https://xenostudio.ai/docs/<slug>` | Documentation home (first page of the docs module) | products with a registered docs module |
| `https://xenostudio.ai/product/<slug>/download/<os>[/<version>]` | Backend **302** deep-link to the R2 installer — stable, version-independent | `desktop` (and `cli` where assets exist) |
| `https://xenostudio.ai/product/<slug>/docs` | Legacy redirect → `/docs/<slug>` | any docs product |
| `https://updates.xenostudio.ai/apps/<slug>/releases.json` | Canonical release feed (full history) on R2 | all releasing products |
| `https://updates.xenostudio.ai/apps/<slug>/version.json` | Derived "latest stable" pointer (Hub auto-update reads this) | all releasing products |
| `https://updates.xenostudio.ai/apps/<slug>/v<version>/<installer>` | The actual installer asset (immutable) | `desktop` |

### How they cross-link

- **Grid → product.** `/products` lists every non-coming-soon product and links each card to `/product/<slug>`.
- **Landing → download / releases / launch.** The landing renders its primary CTA from `delivery` (Download / Open / Install / Waitlist, per the table above) and links to `/product/<slug>/releases` for the history.
- **Landing → docs (automatic).** The landing shows a **Documentation** secondary link **iff** a docs module is registered for that slug (`getProductDocs(slug)` returns a module) — no manual wiring. Register the docs module and the link appears; that is the whole cross-link.
- **Download page → installer.** The Download page and the `/product/<slug>/download/<os>` deep-link both resolve through `releases.json` on R2, 302-ing to `apps/<slug>/v<version>/<installer>`. The publisher (`scripts/publish-cli-releases.mjs`) even prints the human page `https://xenostudio.ai/product/<slug>/releases` and the feed `https://updates.xenostudio.ai/apps/<slug>/releases.json` on success.
- **Docs ↔ landing.** `/product/<slug>/docs` redirects to `/docs/<slug>`; the docs pages and the landing both key off the same slug.

---

## Where to go next

- **`03-release-data.md`** — the R2 layout, the `releases.json` / `version.json` schemas, and the exact `scripts/xeno-release.mjs` (desktop) and `scripts/publish-cli-releases.mjs` (CLI) commands. This is **Flow A**.
- **The landing & docs authoring chapter** — writing the `ProductContent` module, mockups, and the `ProductDocs` module. Canonical source: `PRODUCT-LANDING-SPEC.md` §0–§7 in `xeno-platform`.
- **The deploy chapter** — `npm run build` (vite + prerender) and the on-box deploy to `xeno-platform-001` / `xenostudio-frontend`. This is **Flow B**. Canonical source: `PRODUCT-LANDING-SPEC.md` §8–§9.

Canonical specs (all in `xeno-platform`, not necessarily present in a product repo): `PRODUCT-PAGES-SPEC.md` (release data + JSON contracts), `RELEASE-TO-WEBSITE.md` (release operator how-to), `PRODUCT-LANDING-SPEC.md` (content, prerender, deploy). If a detail isn't in this guide, confirm it there rather than guessing.
