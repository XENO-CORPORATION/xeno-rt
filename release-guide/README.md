# XENO Release Guide

> The portable, self-contained playbook for cutting a release of any XENO product — installer, CLI, or website content. Read every file here **in order** and you will know EXACTLY how a new version reaches R2, the XENO Hub, and xenostudio.ai.

---

## What this folder is

This is the single source of truth for **releasing a XENO product**. It is deliberately narrow: it does not teach you how to build the app — it teaches you how, once you have a build (or a content change), to publish it so that:

- the **website** (`https://xenostudio.ai/product/<slug>`) shows the new version and download,
- the **XENO Hub** auto-update picks it up,
- and the **release history** is recorded correctly.

After reading all files below in order, an agent should be able to run a release end-to-end with no further discovery, using only verbatim commands from these files.

### This folder is PORTABLE

`release-guide/` is **copied verbatim into any XENO product repo** (`xeno-hub`, `xeno-pixel`, `xeno-motion`, `xeno-comms`, `xeno-agent-cli`, …). It is self-contained and repo-agnostic. Anything that varies per product is written as a placeholder:

| Placeholder | Meaning | Example |
|---|---|---|
| `<slug>` | Product slug — identical in the catalog, on R2, and in every URL | `pixel`, `hub`, `agent-cli` |
| `<app>` | Same as `<slug>`; the `--app` value the publishers take | `pixel` |
| `<version>` | Semver, **no leading `v`** | `0.6.4` |
| `<App>` | Human product name (installer filenames) | `XENO Pixel` |
| `<os>` | `win` \| `mac` \| `linux` (download deep-link) | `win` |

The **fixed infrastructure values are real** and never change between products — do not templatize them:

| Constant | Value |
|---|---|
| R2 bucket | `xeno-hub-releases` |
| rclone remote | `r2:` → `r2:xeno-hub-releases` |
| Public update domain | `https://updates.xenostudio.ai` |
| Per-product R2 root | `apps/<slug>/` |
| Website (singular) | `https://xenostudio.ai/product/<slug>` |
| Deploy host | `xeno-platform-001` (`ssh xeno-platform-001`) |
| Repo path on the box | `/mnt/projects/xeno-platform` |
| Frontend container | `xenostudio-frontend` (compose service `frontend`) |
| Publishers | `scripts/xeno-release.mjs`, `scripts/publish-cli-releases.mjs`, `scripts/seed-releases.mjs` (all run from the **`xeno-platform`** repo) |

> There is **no** compiled `xeno-release` binary. "xeno-release" is the Node script `node scripts/xeno-release.mjs`, run from the `xeno-platform` repo. Do not invent a standalone binary, other flags, or other filenames.

---

## Files & recommended read order

Read top to bottom. Each file assumes you have read the ones above it.

| # | File | What it covers |
|---|------|----------------|
| 1 | `01-concepts.md` | The mental model: a product = **four layers** (identity, release data, landing content, docs) joined by one `<slug>`; which layers are read **live** vs which need a platform **deploy**; the four URLs a product owns. |
| 2 | `02-cloudflare-r2.md` | The **R2 / CDN** contract **and prerequisites**: the `xeno-hub-releases` bucket, the `r2:` rclone remote (check: `rclone listremotes`), the public `https://updates.xenostudio.ai` domain, the `apps/<slug>/` layout, the `Cache-Control` model, and the access + tooling to have ready first (`rclone` `r2:` remote, `ssh xeno-platform-001`, working from the `xeno-platform` repo, Node). |
| 3 | `03-release-data.md` | The **data contract** on R2 **and both publish paths**: `releases.json` (canonical full history) + `version.json` (derived latest-stable pointer), the `Release`/`ReleaseAsset` schema, how `latest` is computed — plus how to publish a **downloadable/installer** product with `node scripts/xeno-release.mjs publish` (**§6.1**) and an **npm/CLI** product with `node scripts/publish-cli-releases.mjs` (**§6.2**), and the one-off `seed-releases.mjs` bootstrap for a brand-new feed. **Both the desktop and CLI publish paths live in this file.** |
| 4 | `04-build-and-deploy.md` | Building and shipping the **platform site**: `npm run build` (vite + prerender, **must be clean**) then the on-box deploy to `xeno-platform-001` — `git archive … \| ssh …` into `/mnt/projects/xeno-platform`, CRLF normalization on **text files only** (never on `.webp`/binaries), `sudo docker compose build frontend && … up -d frontend`, build-before-swap, `:rollback` images. |
| 5 | `05-landing-and-docs.md` | Authoring/updating **landing content + docs** (the typed module → registry → dispatcher pattern), the mockup registry, and the markdown-in-`.ts` **backtick-escaping gotcha**. These changes are compiled + prerendered, so they ship via the `04-build-and-deploy.md` flow. |
| 6 | `06-release-runbook.md` | The **end-to-end runbook**: step-by-step for each release type (installer, CLI, content), tying together the publish path and — where relevant — the build + deploy, with the verification `curl`s (the feed, the `/product/<slug>/download/<os>` 302, the landing 200) and headless-Edge screenshots. |
| 7 | `07-troubleshooting.md` | **Troubleshooting + guardrails**: common failures and fixes (feed not updating, 404 on the download deep-link, stale cached `dist`), rollback (`:rollback` images / build-before-swap), and the release rules (one `<slug>` everywhere, semver no leading `v`, always publish **both** JSON files, never overwrite an existing installer, exactly one stable entry is `latest`). |

> Throughout the guide, sibling files are referenced by the exact filenames in the table above — e.g. "see `03-release-data.md`" for the schema and publish paths, "see `04-build-and-deploy.md`" for the deploy.

---

## How to use this folder — pick your path

Almost every release is one of these. Read `01-concepts.md` first, then jump to the path you need.

- **New installer version** (desktop app, no site-content change)
  → `03-release-data.md` **§6.1** (publish with `node scripts/xeno-release.mjs publish`) → verify via `06-release-runbook.md` / `07-troubleshooting.md`.
  **No platform deploy** — the website reads `releases.json`/`version.json` **live** from R2.

- **New CLI / npm version** (no installer)
  → `03-release-data.md` **§6.2** (publish with `node scripts/publish-cli-releases.mjs`) → verify via `06-release-runbook.md` / `07-troubleshooting.md`.
  **No platform deploy** — read live from R2.

- **Marketing / landing or docs change** (no new binary)
  → author in `05-landing-and-docs.md`, then build + deploy per `04-build-and-deploy.md` → verify via `06-release-runbook.md` / `07-troubleshooting.md`.
  **Requires a platform rebuild + deploy** (content is compiled + prerendered into the bundle).

- **New product** (first release + first landing/docs)
  → all of it — including the one-off `seed-releases.mjs` bootstrap covered in `03-release-data.md`.

> Rule of thumb: **release data is read live (no deploy); landing content and docs are compiled and prerendered (deploy required).** A version bump alone never needs a platform deploy. (`xeno-release.mjs` prints a SPEC §13.6 reminder to prerender static SEO pages so the version in the crawler-visible `<head>` matches — usually optional, since the live SPA hydrates the current version.)

---

## Wire it into your project (the agent tag)

Because this folder is copied into each product repo, the repo's own instructions must **point every agent at it before any release**. Add the following heading to **both** the repo's `CLAUDE.md` and its `agents.md` (create the file if missing), near the top or in a "Releasing" section:

```md
## Releasing — BEFORE any release, read `release-guide/` in full.

This repo ships a portable `release-guide/` folder. Before cutting ANY release —
a new version (installer or CLI) OR a landing/docs content change — read every
file in `release-guide/` IN ORDER, starting with `release-guide/README.md`.
It is the single source of truth for how a release reaches R2, the Hub, and
xenostudio.ai. Do not improvise release commands; use the verbatim commands there.
```

That one pointer is all the wiring needed: the guide is self-contained, so an agent that follows the tag lands here and reads the rest in order.

### The `xeno-product-release` skill (invocable — Claude Code, Codex, XENO Agent CLI)

This folder ships the playbook as an **invocable skill** for all three agent CLIs. Say
"**release &lt;product&gt;**" (or "cut a patch", "publish the feed", "deploy the docs")
and the agent routes to the correct path (installer / CLI / content) and follows this
guide, with dry-run + confirmation gates. Two source forms cover every tool (both use
the folder name as the skill name → `xeno-product-release`):

- `release-guide/skill/SKILL.md` — the **open Agent Skills** format (Claude Code + Codex).
- `release-guide/skill/xeno-product-release.md` — the **XENO Agent CLI** format.

Install **globally** (recommended — available in every project on your machine; swap
`~/…` for a repo-relative path if you want it project-scoped and versioned instead).
Each tool discovers skills by directory:

```bash
# Claude Code   → global: ~/.claude/skills/<name>/   (project: .claude/skills/<name>/)
mkdir -p ~/.claude/skills/xeno-product-release && cp release-guide/skill/SKILL.md ~/.claude/skills/xeno-product-release/SKILL.md

# OpenAI Codex  → global: install BOTH (official docs use ~/.agents/skills; some versions use ~/.codex/skills)
mkdir -p ~/.agents/skills/xeno-product-release ~/.codex/skills/xeno-product-release
cp release-guide/skill/SKILL.md ~/.agents/skills/xeno-product-release/SKILL.md
cp release-guide/skill/SKILL.md ~/.codex/skills/xeno-product-release/SKILL.md
# in Codex, run /skills to confirm it's discovered

# XENO Agent CLI → global: run `xeno skills` to find your "User dir" (e.g. ~/.xeno-code/skills), then:
mkdir -p ~/.xeno-code/skills && cp release-guide/skill/xeno-product-release.md ~/.xeno-code/skills/xeno-product-release.md
xeno skills list   # confirm: user:xeno-product-release [enabled]
```

All are thin wrappers (progressive disclosure) that defer to the files above for
verbatim commands — keep this guide the single source of truth. Distinct from each
tool's built-in generic `release` skill (local publishability). In Claude Code and
Codex the skill is discovered implicitly from its `description`, or explicitly via
`/skills`. Spec: `../PRODUCT-RELEASE-SKILL-SPEC.md`.

---

## Quick reference (cheat sheet)

Both paths run from the **`xeno-platform`** repo. Substitute `<slug>`, `<version>`, `<App>`, `<YYYY-MM-DD>`, `<files>`.

```bash
# ── PURE RELEASE — new installer version (R2 is read live; NO platform deploy) ──
node scripts/xeno-release.mjs publish \
  --app <slug> --version <version> --date <YYYY-MM-DD> \
  --type patch --notes-file CHANGELOG.md \
  --win "release/<App> Setup <version>.exe"          # add --mac / --linux if built
curl -sI "https://xenostudio.ai/product/<slug>/download/win"    # expect 302

# ── CONTENT + DEPLOY — landing/docs edit (needs a rebuild; see 04-build-and-deploy.md) ──
npm run build                                         # vite + prerender — MUST be clean first
git add <files> && git commit -m "content(<slug>): …"
git archive --format=tar HEAD <files> | ssh xeno-platform-001 \
  "cd /mnt/projects/xeno-platform && sudo tar xf - --overwrite \
   && sudo docker compose build frontend && sudo docker compose up -d frontend"
curl -sI https://xenostudio.ai/product/<slug>         # expect 200
```

> The content-deploy line above is abbreviated. The **full** deploy (CRLF normalization on text files only — never on `.webp`/binaries) is in `04-build-and-deploy.md`; the landing/docs authoring it ships lives in `05-landing-and-docs.md`; rollback and headless-screenshot verification are covered in `06-release-runbook.md` and `07-troubleshooting.md`. For CLI/npm products, swap the first block for `node scripts/publish-cli-releases.mjs` (see `03-release-data.md` **§6.2**).
