# 05 — Landing Page & Docs (the deploy-required content layer)

Purpose: how to author a product's marketing **landing page** and its **documentation** on `xenostudio.ai`, why both are compiled + prerendered (so editing either needs a platform rebuild + deploy), and the exact typed-module → registry → dispatcher pattern that makes adding a product "one file + one import."

---

## Where this fits: the four layers of a product

Every XENO product is represented by exactly four things, all joined by the **`<slug>`** (the same lowercase string everywhere — catalog, R2 folder, and URLs). Two of the four layers are covered by *this* file.

| Layer | Lives in | Changes when | Deploy? |
|---|---|---|---|
| **Identity** | `src/lib/productCatalog.ts` (one entry) | product added / re-classified | build + deploy |
| **Release data** | R2 `apps/<slug>/releases.json` (+ `version.json`) | **every release** (auto, via `xeno-release`) | **NO deploy — read live** |
| **Landing content** | `src/content/products/<slug>.ts` + `src/components/product/mockups/` + `public/product-assets/<slug>/` | marketing / design changes (rare) | **build + deploy** |
| **Documentation** | `src/content/docs/<slug>.ts` (registered in `index.ts`) | docs authored / updated | **build + deploy** |

**URLs a product owns:** `/product/<slug>` (landing) · `/product/<slug>/download` · `/product/<slug>/releases` · `/docs/<slug>` (docs).

**Key consequence (PRODUCT-LANDING-SPEC.md §0):** release data is read **live** — a new version/download appears with **no platform deploy** (see `03-release-data.md`). *Landing content* **and** *documentation* are compiled into the SPA bundle and prerendered to static SEO HTML at build time, so they require a platform rebuild + deploy. This file is the deploy-required layer.

Reference implementations already in the repo: **`comms`** is the landing contract; **`agent-cli`** is the docs contract (23 pages authored). Read one of each before authoring your own — the full landing contract and design bar live in **`PRODUCT-LANDING-SPEC.md`**.

---

## Part A — Landing page

### The pattern: typed module + registry + dispatcher

**1. Typed module** — `src/content/products/_types.ts` defines the single schema every landing page conforms to: interface **`ProductContent`**. You author one `.ts` file per product that default-exports an object of this type.

**2. Registry** — `src/content/products/index.ts` imports each module into an array and exposes a lookup by slug:

```ts
import comms from './comms';
import agentCli from './agent-cli';

const MODULES: ProductContent[] = [comms, agentCli];
const BY_SLUG = new Map(MODULES.map((m) => [m.slug, m]));

export function getProductContent(slug?: string): ProductContent | undefined {
  return slug ? BY_SLUG.get(slug) : undefined;
}
export const RICH_PRODUCT_SLUGS = MODULES.map((m) => m.slug); // used by prerender for SEO
```

**3. Dispatcher** — `src/pages/ProductPage.tsx` picks the renderer at runtime. No routing change is needed to add a landing page:

```tsx
const { slug } = useParams();
const product = getProduct(slug);
if (!product) return <Navigate to="/" replace />;
const content = getProductContent(slug);
return content
  ? <ProductLanding product={product} content={content} />   // rich template
  : <LeanProductPage product={product} />;                    // lean fallback
```

Content module registered → full `ProductLanding`. Absent → the lean fallback page (same file). The route is already wired in `src/App.tsx`: `<Route path="/product/:slug" element={<ProductPage />} />`.

### Files you edit to add a LANDING

1. **`src/content/products/<slug>.ts`** — author it; default-export a `ProductContent`. `slug` **MUST** match the catalog entry in `src/lib/productCatalog.ts`. Author from the product's REAL repo (`../xeno-<app>`), not from memory.
2. **`src/content/products/index.ts`** — `import <name> from './<slug>'` and add `<name>` to the `MODULES` array.
3. **(visuals)** `src/components/product/mockups/<Product><View>.tsx` — a hand-built JSX/Tailwind mockup. Register it in `src/components/product/mockups/index.tsx` (`MOCKUPS['<key>'] = Component`), then reference it from content as `{ type: 'mockup', src: '<key>' }`.
4. **(optional raster)** `public/product-assets/<slug>/` — optimized `.webp` / `.mp4` only. Convert with `magick in.png -resize 1600x -quality 82 out.webp`. These are binary — see the CRLF caveat in the deploy section (never `sed` them).

### The `ProductContent` schema (what a landing can contain)

Only **`hero`** and **`features`** (≥1) are required. Every other section is optional, and the template **omits whatever is absent** — "no empty sections, ever." Fields (`_types.ts`):

- `slug` — must match the catalog entry.
- `hero { headline, sub, media, badges?, note? }` — required.
- `features` — required, ≥1 `FeatureSpotlight`.
- `trust?`, `highlights?`, `gallery?`, `useCases?`, `howItWorks?`, `comparison?`, `specs?`, `faq?` — all optional.
- `seo? { title?, description? }` — overrides the prerendered `<head>` for this product; falls back to `${name} — ${tagline}` / `tagline` from the catalog when absent.

Supporting types:
- `Media { type: 'image' | 'video' | 'mockup', src, alt, poster? }` — when `type: 'mockup'`, `src` is a key into the mockup registry.
- `FeatureSpotlight { eyebrow?, title, desc, bullets?, icon?, accent?, media? }` — if `media` is present the section renders as an alternating spotlight; if absent it renders in the bento grid.

### What renders automatically (do NOT rebuild)

Registering the module gives you, for free, the full `ProductLanding` template: hero, feature spotlights/bento, gallery, use-cases, how-it-works, comparison, specs, FAQ, the accent-theme system (Shift+T), the delivery-aware CTA (download / launch / install / waitlist, driven by the catalog's `delivery`), and the SEO prerender. It also auto-adds a **Documentation** link when docs exist (see below). Consult `PRODUCT-LANDING-SPEC.md` for the design bar every landing must clear.

---

## Part B — Documentation

Docs use the **same** typed-module + registry + dispatcher shape as landing content (PRODUCT-LANDING-SPEC.md §7.5), so they scale the same way.

### The pattern

**1. Typed module** — `src/content/docs/_types.ts`:
- `DocPage { slug, title, description?, body }` — `body` is a **Markdown string** (rendered by `DocMarkdown`: GFM + math + fenced code).
- `DocSection { title, pages: DocPage[] }`
- `ProductDocs { slug, productName, tagline?, sections: DocSection[], seo? { title?, description? } }` — `slug` matches the product slug in `productCatalog.ts`.
- `DocRoute { productSlug, productName, sectionTitle, pageSlug, title, description?, body }` — a flattened page+context used by prerender and search.

So a docs module is nested: `sections[].pages[]`, and each page's `body` is markdown.

**2. Registry** — `src/content/docs/index.ts`:

```ts
import agentCli from './agent-cli';

const MODULES: ProductDocs[] = [agentCli];
const BY_SLUG = new Map(MODULES.map((m) => [m.slug, m]));

export function getProductDocs(slug?: string): ProductDocs | undefined { /* BY_SLUG.get */ }
export function getDocPage(slug, pageSlug): { page, sectionTitle } | undefined { /* walks sections→pages */ }
export function firstDocPage(slug): DocPage | undefined { /* sections[0]?.pages[0] — the /docs/<slug> landing */ }
export function allDocProducts(): ProductDocs[] { return MODULES; }        // hub + prerender
export function allDocRoutes(): DocRoute[] { /* flattens product→section→page */ } // prerender + search
export const DOCUMENTED_SLUGS = MODULES.map((m) => m.slug);
```

**3. Dispatcher** — `src/pages/ProductDocs.tsx`:

```tsx
const { slug, page } = useParams();
const product = getProductDocs(slug);
if (!product) return <Navigate to="/docs" replace />;
const pageSlug = page || firstDocPage(product.slug)?.slug;
const found = pageSlug ? getDocPage(product.slug, pageSlug) : undefined;
// unknown page → redirect to first page; else:
return <DocsLayout product={product} page={found.page} sectionTitle={found.sectionTitle} />;
```

`/docs/<slug>` with no page renders the product's first page directly (no redirect flash). The same file exports `ProductDocsRedirect` so `/product/<slug>/docs` → `/docs/<slug>`. Routes are already wired in `src/App.tsx` (`/docs`, `/docs/:slug`, `/docs/:slug/:page`, `/product/:slug/docs`) — no routing change needed to add docs.

### Files you edit to add DOCS ("one file + one import")

1. **`src/content/docs/<slug>.ts`** — author it; default-export a `ProductDocs`. Each page's `body` is a markdown string. Author from the real repo (README / CLAUDE / `docs/` + command sources).
2. **`src/content/docs/index.ts`** — import it and add it to the `MODULES` array. **That's it.**

### What you get free (do NOT rebuild)

Routing, the sidebar nav, the on-page TOC (scroll-spy), ⌘K search, prev/next, the landing's *Documentation* link, and the SEO prerender all pick up the new module automatically. Also free: `DocMarkdown` (GFM + math + syntax-highlighted code with copy buttons + heading anchors), `DocsLayout` (3-pane: sidebar + content + TOC + search + mobile drawer), per-page prerender to static HTML with `TechArticle` JSON-LD, and `sitemap.xml` inclusion.

---

## The backtick gotcha (markdown-in-`.ts`-template-literal) ⚠️

Doc `body` values are markdown living inside `.ts` **template literals** (backtick-delimited strings). That means every backtick *in your markdown* collides with the string delimiter. Rules (PRODUCT-LANDING-SPEC.md §7.5):

- **Escape every backtick in a `body` as `` \` `` (single backslash + backtick).** This applies to code fences and inline code alike — a fenced block opens/closes as `` \`\`\`bash `` … `` \`\`\` `` and an inline span is `` \`<app>\` ``.
- **Writing `` \\` `` (double backslash + backtick) breaks the template literal.** If it slips in, byte-replace `\x5c\x5c\x60` → `\x5c\x60` with Python:

```python
import pathlib
p = pathlib.Path('src/content/docs/<slug>.ts')
p.write_bytes(p.read_bytes().replace(b'\x5c\x5c\x60', b'\x5c\x60'))  # \\` -> \`
```

- **Keep `<placeholders>` inside code spans.** The renderer uses `rehype-raw`, so a bare `<x>` in prose is treated as HTML. Placeholders like `` \`<model-id>\` ``, `` \`<slug>\` ``, `` \`<name>\` `` must live in code spans, or they vanish / break rendering.
- **`npm run build` is the safety net** — it compiles the module and fails loudly on a stray/unescaped backtick. Never skip the build before deploying docs.

(The same template-literal escaping applies anywhere a landing module embeds a backtick, but docs — being long markdown bodies — are where it bites.)

---

## Landing auto-links Docs (no manual wiring)

`src/pages/ProductLanding.tsx` calls `getProductDocs(product.slug)` and renders a **Documentation** secondary link **iff** a docs module is registered for that slug:

```tsx
const hasDocs = !!getProductDocs(product.slug);
// ...
{hasDocs && <Link to={`/docs/${product.slug}`} className={secondaryLink}>Documentation</Link>}
```

So: register the docs module, and the landing's Documentation link lights up automatically. Nothing else to touch.

---

## When an edit needs a deploy — vs when it doesn't

| You changed… | Layer | Needs platform build + deploy? |
|---|---|---|
| `releases.json` / `version.json` on R2 (a new version, notes, installer) | Release data | **No** — page reads it live (`fetchReleases`). See `03-release-data.md`. |
| `src/content/products/<slug>.ts`, a mockup, or a `public/product-assets/<slug>/` file | Landing content | **Yes** |
| `src/content/docs/<slug>.ts` (or its registration) | Documentation | **Yes** |
| `src/lib/productCatalog.ts` entry | Identity | **Yes** |

Landing content and docs are compiled into the bundle and prerendered by `scripts/prerender-products.mjs` at build time, which is why they need a deploy. Release data is fetched at page load, so it does not.

---

## Deploying content & docs (build + deploy)

The canonical procedure is **PRODUCT-LANDING-SPEC.md §8.2** (content flow). CI is currently manual (org Actions billing down), so the on-box flow below is the reality. Run from the `xeno-platform` repo on the content branch (currently `landing-redesign-v3` — confirm the active content branch before you start).

**1. Build clean first.** `npm run build` runs `vite build` **then** `node scripts/prerender-products.mjs`. It must be clean before you commit — this is where the backtick gotcha surfaces, and where the prerender confirms your `seo{}` head and `dist/docs/<slug>/` pages emit correctly. There is no separate lint/typecheck gate wired into `build`, so a clean build is the gate.

**2. Commit, then stream committed files to the box and rebuild the frontend** (§8.2, verbatim):

```bash
# from xeno-platform, on branch landing-redesign-v3
npm run build                      # vite + prerender — MUST be clean first
git add <changed files> && git commit -m "…"
git archive --format=tar HEAD <files> | ssh xeno-platform-001 \
  "cd /mnt/projects/xeno-platform && sudo tar xf - --overwrite \
   && find <text files> -exec sudo sed -i 's/\r$//' {} +  \  # normalize CRLF; NEVER sed binaries
   && sudo docker compose build frontend && sudo docker compose up -d frontend"
# verify: curl -sI https://xenostudio.ai/product/<slug>  → 200
```

Notes:
- `git archive` reads `HEAD`, so files must be **committed** first.
- **CRLF rule:** the repo is developed on Windows, so text sources are LF-normalized before the Docker build. Scope `<text files>` to text sources only (`.ts` `.tsx` `.css` `.html` `.md` `.mjs` `.json`) and **exclude binary assets** — running `sed` on a `.webp` / `.png` / `.mp4` / `.woff` corrupts it. Sed the text files, never the raster assets under `public/product-assets/<slug>/`.
- **Build-before-swap:** `docker compose build frontend` builds the new image; the running container is only swapped by `up -d frontend` after a successful build. If the build fails, the old `xenostudio-frontend` container keeps serving. Rollback images are tagged `:rollback`.
- If a stale/cached Docker build layer is serving old `dist`, `sudo docker compose build --no-cache frontend` is a valid operator override to force a clean rebuild. (It is not written into the §8.2 command block — treat it as an override, not a quoted spec command. It is unrelated to the R2 `Cache-Control: no-cache` header in `03-release-data.md`.)

**3. Verify.** `curl -sI https://xenostudio.ai/product/<slug>` must return **200**. Then screenshot-verify (PRODUCT-LANDING-SPEC.md §9) with headless Edge:

```bash
edge --headless --window-size=1600,1000 --virtual-time-budget=10000 \
  --screenshot=out.png "https://xenostudio.ai/product/<slug>?accent=amber"
# tall page: --window-size=1400,10000 then `magick out.png -crop WxH+X+Y +repage crop.png`
```

Check the hero, scroll, confirm the download/launch CTA, toggle Shift+T through accents, and open `/docs/<slug>`.

**Infra facts (fixed):** host `xeno-platform-001`; box path `/mnt/projects/xeno-platform`; compose service `frontend` → container `xenostudio-frontend`; the frontend is a two-stage build (`Dockerfile.frontend`: `npm ci --legacy-peer-deps` → `npm run build` → nginx serving `/usr/share/nginx/html`), published on host loopback `127.0.0.1:4040:80` behind the host nginx for `xenostudio.ai`.

---

## Responsibilities (spec §8.3)

- **Product repo** owns: its catalog entry, its releases (via `xeno-release` — see `03-release-data.md`), and design input.
- **Platform** (`xeno-platform`) owns: the content module, the mockups, the template, the prerender, and the deploy.

---

## Canonical references (PRODUCT-LANDING-SPEC.md §11)

- **Full landing contract + design bar:** `PRODUCT-LANDING-SPEC.md`
- Rich template: `src/pages/ProductLanding.tsx` · dispatcher / lean fallback: `src/pages/ProductPage.tsx`
- Landing schema: `src/content/products/_types.ts` · registry: `src/content/products/index.ts` · reference module: `src/content/products/comms.ts`
- Mockups + registry: `src/components/product/mockups/` (`index.tsx`, `CommsChat`, `CommsMobile`, …)
- Docs model + registry: `src/content/docs/_types.ts`, `src/content/docs/index.ts` · reference docs module: `src/content/docs/agent-cli.ts`
- Docs renderer/layout: `src/components/docs/` (`DocMarkdown`, `DocsLayout`, `DocsSidebar`, `TableOfContents`, `DocsSearch`, `toc`)
- Docs pages: `src/pages/DocsHome.tsx`, `src/pages/ProductDocs.tsx`
- Prerender (landing + docs): `scripts/prerender-products.mjs`
- Routes: `src/App.tsx`
- Release data (the live, no-deploy layer): see **`03-release-data.md`**
