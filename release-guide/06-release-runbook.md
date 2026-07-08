# 06 — The Release Runbook

> **Purpose:** the copy-pasteable, end-to-end checklist that takes a XENO product from "code is ready" to "live on the website" — with two clearly separated tracks (a pure version release vs. a release that also changes landing/docs), the exact commands, git conventions, and a final definition-of-done.

---

## 0. Before you start — read this

This runbook is **portable**: it is copied verbatim into every XENO product repo. Anything that varies per product appears as a placeholder — substitute the real value:

| Placeholder | Means | Example |
|---|---|---|
| `<slug>` | the product's catalog slug (also the `--app` value and the R2 folder). Extension overrides its folder via the catalog `r2` field. | `pixel`, `comms`, `agent-cli` |
| `<app>` | same as `<slug>` — the value you pass to `--app` | `hub` |
| `<version>` | semver, **no leading `v`** in the JSON feed | `0.6.4` |
| `<host>` | the deploy SSH target | `xeno-platform-001` |
| `<box-path>` | the repo checkout on the box | `/mnt/projects/xeno-platform` |

The fixed infrastructure values below are **real and identical everywhere** — do not templatize them:

- R2 bucket: `xeno-hub-releases`
- rclone remote: `r2:` → `r2:xeno-hub-releases`
- Public read domain: `https://updates.xenostudio.ai`
- Website: `https://xenostudio.ai` (product page is **singular** `/product/<slug>`)
- Deploy box: `xeno-platform-001`, path `/mnt/projects/xeno-platform`, frontend container `xenostudio-frontend`

### Where each command runs

- **Building the installer artifact** happens in the **product repo** (`../xeno-<name>`).
- **Publishing to R2** and **deploying the website** are run from a **`xeno-platform` checkout** — that is where `scripts/xeno-release.mjs`, `scripts/publish-cli-releases.mjs`, and `npm run build` live. Content deploys run on branch `landing-redesign-v3`.

### The four layers of a product (why there are two tracks)

Every product is four things joined by the slug. Only two of them need a website deploy. See **03-release-data.md** for the full schemas.

| Layer | Lives in | Deploy needed? |
|---|---|---|
| **Identity** | `src/lib/productCatalog.ts` | build + deploy |
| **Release data** | R2 `apps/<slug>/releases.json` (+ `version.json`) | **NO — read live** |
| **Landing content** | `src/content/products/<slug>.ts` + mockups + assets | build + deploy |
| **Documentation** | `src/content/docs/<slug>.ts` | build + deploy |

Release data is fetched **live** by the page on load, so a new version/download appears **with no platform deploy**. Landing content and docs are compiled + prerendered, so changing either requires a rebuild + deploy.

That split is exactly the two tracks:

- **TRACK A — a PURE release** (new version, patch, or hotfix; no site content change). Build the artifact, run the publisher, verify R2 + the live page. **No platform deploy.**
- **TRACK B — a release that ALSO changes landing/docs.** Author the content module(s), `npm run build` clean, commit, deploy on-box, verify + screenshot.

### Prerequisites (both tracks)

- **rclone** configured with an `r2:` remote pointing at `xeno-hub-releases`. Verify:
  ```bash
  rclone listremotes        # you should see: r2:
  ```
- A **`xeno-platform` checkout** with `npm ci` (or `npm ci --legacy-peer-deps`) already run, so `scripts/*.mjs` and `npm run build` work.
- **Track B only:** SSH access to `<host>` (`ssh xeno-platform-001`) and `sudo` docker rights on the box.

### THE MANDATORY RULE

> **Every release MUST be reflected on the product pages.** A release is **not complete** until `releases.json` **and** `version.json` on R2 carry the new version and the live `/product/<slug>/releases` page shows it. This is not optional — the Hub auto-updater and the website both read those two files.

---

## TRACK A — Pure release (no site content change)

Use this when you are shipping a new **version / patch / hotfix** and NOT touching landing or docs. **No platform deploy is involved** — the website reads the R2 feed live.

### A1 — Build the artifact (desktop products only)

In the **product repo**, produce the signed installer(s). CLI/npm products skip this step (they have no installer — see A2, CLI variant).

```bash
# in ../xeno-<name>
npx electron-vite build && npx electron-builder --win
# → release/<App> Setup <version>.exe   (and .dmg / .AppImage on the respective OS)
```

Note the exact installer path(s) — you pass them to the publisher as `--win` / `--mac` / `--linux`. Never overwrite an already-published `v<version>/` installer: **installers are immutable**.

### A2 — Run the publisher (from the `xeno-platform` checkout)

There are two publishers. Pick by product type.

#### Desktop / installer products → `scripts/xeno-release.mjs`

This is the **canonical desktop release publisher**. It uploads the installer(s), prepends a full entry to `releases.json` (canonical history), and regenerates `version.json` (the derived latest-stable pointer Hub auto-update reads). A release is not complete until BOTH JSON files are updated.

The first positional argument **must** be `publish`.

```bash
node scripts/xeno-release.mjs publish \
  --app <slug> --version <version> --date <YYYY-MM-DD> \
  --channel stable --type patch [--severity normal] [--title "..."] \
  (--notes "markdown" | --notes-file ../xeno-<name>/CHANGELOG-<version>.md) \
  [--win "../xeno-<name>/release/<App> Setup <version>.exe"] \
  [--mac "../xeno-<name>/release/<App>-<version>.dmg"] \
  [--linux "../xeno-<name>/release/<App>-<version>.AppImage"] \
  [--dry-run]
```

Worked desktop example (verbatim shape from `RELEASE-TO-WEBSITE.md` §3.B):

```bash
node scripts/xeno-release.mjs publish \
  --app pixel --version 0.6.4 --date 2026-06-28 --type patch \
  --notes-file ../xeno-pixel/CHANGELOG-0.6.4.md \
  --win "../xeno-pixel/release/XENO Pixel Setup 0.6.4.exe"
```

Flag behavior you can rely on (from the script):

- `--app`, `--version`, `--date`, and notes (`--notes` or `--notes-file`) are **required**; empty notes fail with `notes required`.
- `--version` strips a leading `v` (`v0.6.4` → `0.6.4`).
- `--channel` — anything not exactly `beta` becomes `stable`. Only stable entries are ever flagged `latest`.
- `--type` — one of `release | patch | hotfix`; anything else defaults to `release`.
- `--severity` — `critical` or (default) `normal`; `critical` is highlighted on the site.
- `--title` — optional short headline; when present, it is preferred over `notes` for the `version.json` `notes` field (capped at 400 chars).
- `--dry-run` — prints the `rclone` commands instead of executing them. **Always dry-run first when unsure.**

What it writes to R2:

- `apps/<slug>/v<version>/<InstallerFilename>` — the installer(s), uploaded with `rclone copyto ... --no-traverse` (no cache header → immutable).
- `apps/<slug>/releases.json` — the new entry **prepended** to the full history; `latest` recomputed so exactly one stable entry is `latest:true`.
- `apps/<slug>/version.json` — regenerated from the newest stable entry (filenames only; Hub joins `v<version>/`).

Both JSON files are uploaded with `--header-upload "Cache-Control: no-cache"`; installers are not (they cache forever).

> The script prints a closing reminder: *"trigger a product-pages prerender + frontend deploy so the static/SEO pages reflect the new version."* For a **pure version bump this is optional** — the live page fetches `releases.json` on load, and the prerendered SEO `<head>` (title/description/canonical/OG/JSON-LD) is **not** version-specific, so it does not go stale. Ignore the reminder for Track A; it is satisfied automatically the next time you run a Track B content deploy. Publishing does **not** require redeploying the website.

#### CLI / npm products → `scripts/publish-cli-releases.mjs`

CLI products (e.g. `agent-cli`, the SDK) have no installer, so the installer/auto-update machinery of `xeno-release.mjs` does not apply. This publisher builds the R2 feed from **real data**: versions + dates from the npm registry, human notes from the CLI's own `RELEASE_NOTES` map. Nothing is authored — it mirrors npm + the CLI source.

```bash
node scripts/publish-cli-releases.mjs \
  --app <slug> \
  --pkg @xeno-corporation/xeno-<name> \
  --notes ../xeno-<name>/apps/xeno-<name>/src/commands/release-notes.ts \
  [--out dist-feed] [--dry-run]
```

Worked example:

```bash
node scripts/publish-cli-releases.mjs \
  --app agent-cli \
  --pkg @xeno-corporation/xeno-agent-cli \
  --notes ../xeno-agent-cli/apps/xeno-agent-cli/src/commands/release-notes.ts
```

- `--app`, `--pkg`, `--notes` are **required**.
- The feed is the **intersection** of versions that are BOTH on npm AND carry release notes, newest-first; the npm `latest` dist-tag is flagged `latest`.
- Writes `apps/<slug>/releases.json` and an npm-shaped `apps/<slug>/version.json` (carries `version`/`date`/`npm`/`install`/`notes`, no windows/mac/linux keys). Both uploaded with `Cache-Control: no-cache`.
- `--dry-run` prints the `rclone` commands without uploading.

> **Alternative (single manual entry):** a CLI release can also be published through `xeno-release.mjs` with no installer flags, e.g.
> `node scripts/xeno-release.mjs publish --app agent-cli --version 0.4.0 --date 2026-06-28 --type release --notes "$(cat CHANGELOG-0.4.0.md)"`.
> Prefer `publish-cli-releases.mjs` when you want the whole feed auto-derived from npm.

### A3 — Verify (R2 feed + live page)

Confirm both JSON files updated and the live redirect works. See **03-release-data.md** for what each field means.

```bash
# canonical history — new version should be the first entry
curl -s https://updates.xenostudio.ai/apps/<slug>/releases.json | head

# derived latest-stable pointer — headers + body
curl -sI https://updates.xenostudio.ai/apps/<slug>/version.json

# stable download deep-link must 302 to the installer (desktop products)
curl -sI "https://xenostudio.ai/product/<slug>/download/win"
```

Then open the live pages in a browser and confirm the new version appears (the SPA fetches the feed live):

- `https://xenostudio.ai/product/<slug>/releases`
- `https://xenostudio.ai/product/<slug>` (download/launch CTA resolves to the new version)

**Track A ends here. No platform deploy.**

---

## TRACK B — Release that ALSO changes landing/docs

Use this when the release ships **new marketing/landing content or documentation** in addition to (or instead of) a version bump. Landing content and docs are compiled into the bundle and prerendered, so they require a **rebuild + on-box deploy**. See **05-landing-and-docs.md** for authoring detail.

> If this release also ships a new binary version, run **all of TRACK A first** (build → publish → verify), then do TRACK B for the content. The two are independent: R2 for versions, deploy for content.

### B1 — Author the content module(s)

Edit in the **`xeno-platform` repo** on branch `landing-redesign-v3`.

**Landing** (per 05-landing-and-docs.md):
1. `src/content/products/<slug>.ts` — default-export a `ProductContent` whose `slug` matches the catalog entry. `hero` + `features` are required; every other section is optional and omitted if absent.
2. `src/content/products/index.ts` — `import <name> from './<slug>'` and add it to the `MODULES` array.
3. (visuals) `src/components/product/mockups/<Product><View>.tsx` — hand-built JSX/Tailwind mockup; register in `src/components/product/mockups/index.tsx`; reference from content as `{ type:'mockup', src:'<key>' }`.
4. (optional raster) `public/product-assets/<slug>/` — optimized `.webp` / `.mp4` (e.g. `magick in.png -resize 1600x -quality 82 out.webp`).

**Docs** (per 05-landing-and-docs.md):
1. `src/content/docs/<slug>.ts` — default-export a `ProductDocs` (`sections[].pages[]`, each page `body` a markdown string). Registering it auto-adds a "Documentation" link on the landing page.
2. `src/content/docs/index.ts` — import it and add to the `MODULES` array. That's it — routing, nav, TOC, search, prev/next, and SEO prerender all pick it up.

> **Backtick-escaping gotcha (docs):** doc `body` values are markdown inside `.ts` template literals. Every backtick (code fences, inline code) MUST be escaped as `` \` `` — a **single** backslash + backtick. Double (`` \\` ``) breaks the template literal. Keep `<placeholders>` inside code spans (`rehype-raw` treats bare `<x>` as HTML). `npm run build` fails loudly on a stray backtick.

### B2 — Build clean (MUST pass before committing)

```bash
# in the xeno-platform checkout, on branch landing-redesign-v3
npm run build
```

`npm run build` = `vite build && node scripts/prerender-products.mjs`. It emits the SPA into `./dist`, then prerenders the SEO HTML per route and writes `sitemap.xml` + `robots.txt`. **The build must be clean** — fix any TS/build error; confirm the prerender emits the landing with the correct `<head>` from `seo`, and the doc pages under `dist/docs/<slug>/`.

### B3 — Commit (message convention + trailer)

Commit only after a clean build. Use **Conventional Commits** — `type(scope): summary` — matching the repo history (`feat`, `fix`, `docs`; scope like `product-pages`):

```bash
git add <changed files>
git commit
```

Commit message convention:

```
<type>(<scope>): <imperative summary>

<optional body — what/why>

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

- `<type>` — `feat` (new landing/docs), `fix` (mockup/content fix), `docs` (spec/reference).
- `<scope>` — typically `product-pages` (or the product `<slug>`).
- End with the `Co-Authored-By:` trailer.
- The deploy reads `HEAD` (`git archive`), so **files must be committed before deploying** — an uncommitted edit will not be transferred.

### B4 — Deploy on-box (build-before-swap)

The org's CI Actions billing is down, so the content deploy is currently **manual**. Run from the `xeno-platform` checkout on branch `landing-redesign-v3` (verbatim from PRODUCT-LANDING-SPEC §8.2):

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

Notes on this pipeline:

- **`git archive ... | ssh <host> "... sudo tar xf - --overwrite"`** streams the committed files over SSH into `<box-path>` and overwrites them in place.
- **CRLF normalization — text files ONLY.** The repo is developed on Windows, so `find <text files> -exec sudo sed -i 's/\r$//' {} +` strips CR from text sources. Scope `<text files>` to `.ts/.tsx/.css/.html/.md/.mjs/.json` and **exclude binary assets** — running `sed` on a `.webp`/`.png`/`.mp4`/`.woff` corrupts it. **NEVER sed binaries.**
- **Build-before-swap.** `docker compose build frontend` builds the new image; the running container is only swapped by `up -d frontend` **after a successful build** — if the build fails, the old container keeps serving. Rollback images are tagged `:rollback`.
- **Stale cached layer?** If a cached Docker layer serves a stale `dist`, force a clean rebuild with `sudo docker compose build --no-cache frontend` before `up -d`. (Operator override — not part of the quoted spec block.)

### B5 — Verify live + screenshot

```bash
curl -sI https://xenostudio.ai/product/<slug>          # expect: 200
```

Then a headless screenshot pass (Edge, verbatim from §9):

```bash
edge --headless --window-size=1600,1000 --virtual-time-budget=10000 \
  --screenshot=out.png "https://xenostudio.ai/product/<slug>?accent=amber"
# tall page: --window-size=1400,10000 then `magick out.png -crop WxH+X+Y +repage crop.png`
```

Visually confirm: hero renders, scroll the page, the download/launch CTA resolves, toggle `Shift+T` through all accents, and open `/docs/<slug>` if docs shipped.

---

## Git tagging conventions

- **JSON feed `version` is semver with NO leading `v`** (e.g. `0.6.4`). `xeno-release.mjs` strips a leading `v` from `--version`, so either form is accepted on the command line but the stored value is bare.
- **Git tags live in the PRODUCT repo** (where the code is built and versioned), not in `xeno-platform`. The orchestrator release workflow is *version bump → git tag → push*. The conventional tag form is `v<version>` (e.g. `v0.6.4`); in a monorepo that ships multiple packages, scope the tag as `<app>-v<version>` (e.g. `agent-cli-v0.4.0`).
- If your product repo already has an established tagging practice, follow it — **confirm the exact convention in that repo's `CLAUDE.md`/`CHANGELOG.md`** rather than assuming.
- The R2 feed and the git tag are independent artifacts: publishing to R2 does not create a tag, and tagging does not publish. Do both.

---

## Definition of done

A release is complete only when **every** applicable box is checked.

**Always (both tracks):**

- [ ] Product repo: version bumped, changes committed, **git tag pushed** (`v<version>` or `<app>-v<version>`).
- [ ] Publisher run (desktop → `scripts/xeno-release.mjs publish …`; CLI → `scripts/publish-cli-releases.mjs …`).
- [ ] `curl -s https://updates.xenostudio.ai/apps/<slug>/releases.json` shows the new version as the first entry, with exactly one stable entry flagged `latest`.
- [ ] `curl -sI https://updates.xenostudio.ai/apps/<slug>/version.json` returns 200 and points at the new latest-stable.
- [ ] Live `https://xenostudio.ai/product/<slug>/releases` shows the new release.
- [ ] **(desktop)** `curl -sI "https://xenostudio.ai/product/<slug>/download/win"` returns **302** to the new installer.

**Track A (pure release) — additionally:**

- [ ] No landing/docs files changed; **no platform deploy performed** (confirmed intentional).

**Track B (content release) — additionally:**

- [ ] Content module(s) authored and registered in the correct `index.ts` (landing and/or docs).
- [ ] `npm run build` is **clean** (vite + prerender); doc pages present under `dist/docs/<slug>/`.
- [ ] Committed with a Conventional-Commits message + `Co-Authored-By:` trailer.
- [ ] Deployed on-box (`git archive … | ssh xeno-platform-001 … sudo docker compose build frontend && … up -d frontend`); CRLF normalized on text files only.
- [ ] `curl -sI https://xenostudio.ai/product/<slug>` returns **200**.
- [ ] Headless screenshot verified (hero, scroll, CTA, accents; `/docs/<slug>` if docs shipped).

**The mandatory rule (never skip):** the release is not done until the product pages reflect it. If R2 or the live page does not show the new version, the release is incomplete — go back and finish it.

---

### See also

- **03-release-data.md** — `releases.json` / `version.json` schemas, `Release`/`ReleaseAsset` fields, how `latest` is computed, the stable-download 302 redirect.
- **05-landing-and-docs.md** — authoring a landing content module + mockups, and authoring a docs module (with the backtick-escaping rules).
