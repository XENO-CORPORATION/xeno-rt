# 03 — Release Data: How a Release Is Written

> The core reference for the two JSON files that define a XENO product's releases (`releases.json` + `version.json`), their full schemas, and the two real scripts that publish them to R2. Copy this file into any XENO product repo; replace `<slug>` / `<app>` / `<version>` / `<pkg>` with your product's real values, and keep the fixed infrastructure values verbatim.

---

## 0. What "release data" is (and what it is not)

A product is represented by four things, joined by its **slug**. **Release data is one of them** and it is the *only* one that appears with **no platform deploy** — the website reads it live at page load.

| Layer | Lives in | Deploy? |
|---|---|---|
| Identity (catalog entry) | `src/lib/productCatalog.ts` | build + deploy |
| **Release data** | **R2 `apps/<slug>/releases.json` (+ `version.json`)** | **NO deploy — read live** |
| Landing content | `src/content/products/<slug>.ts` + mockups + assets | build + deploy |
| Documentation | `src/content/docs/<slug>.ts` | build + deploy |

Release data lives on Cloudflare R2, not in the repo. Publishing a release **uploads two JSON files (and any installers)** to R2; the live site (`fetchReleases()` in `productCatalog.ts`) and XENO Hub (auto-update) both fetch them directly.

**The two files:**

- **`releases.json`** — the **canonical, full history**. A newest-first array of every release. This is the source of truth.
- **`version.json`** — a **derived pointer** to the latest *stable* release, flattened into the legacy Hub-updater shape. Always regenerated from `releases.json`; never hand-authored.

Both live under `https://updates.xenostudio.ai/apps/<slug>/`.

The URLs a product owns for release data:

- Feed: `https://updates.xenostudio.ai/apps/<slug>/releases.json`
- Pointer: `https://updates.xenostudio.ai/apps/<slug>/version.json`
- Installers: `https://updates.xenostudio.ai/apps/<slug>/v<version>/<InstallerFilename>`
- Website releases page: `https://xenostudio.ai/product/<slug>/releases` (note: **singular** `/product/`)
- Stable download deep-link (302 → current installer): `https://xenostudio.ai/product/<slug>/download/<os>`

> **Slug vs. R2 folder.** `<slug>` is the product's catalog `slug`. The R2 folder is `p.r2 ?? p.slug` — usually identical, but a catalog entry may override `r2` (e.g. the browser extension uses `r2: 'extension'`). The publisher's `--app` flag must equal that **R2 folder**, and it must match everywhere (catalog, R2 path, download route).

---

## 1. Fixed infrastructure constants (do not change per product)

These are hard-coded into the publisher scripts and the spec. They are the same for every product — keep them verbatim.

| Thing | Exact value |
|---|---|
| R2 bucket | `xeno-hub-releases` |
| rclone remote | `r2:` → `r2:xeno-hub-releases` |
| Public read domain | `https://updates.xenostudio.ai` |
| Per-product root | `apps/<slug>/` (or `apps/<r2-override>/`) |
| Installer path | `apps/<slug>/v<version>/<InstallerFilename>` |
| Canonical feed | `apps/<slug>/releases.json` |
| Derived pointer | `apps/<slug>/version.json` |
| Website product page | `https://xenostudio.ai/product/<slug>` |

**Environment overrides** (`publish-cli-releases.mjs` only): `XENO_UPDATES_BASE` (default `https://updates.xenostudio.ai`) and `XENO_R2_REMOTE` (default `r2:xeno-hub-releases`). `xeno-release.mjs` hard-codes both constants (no env override).

**Prerequisite:** rclone configured with an `r2:` remote pointing at the `xeno-hub-releases` bucket. Verify with `rclone listremotes` → you should see `r2:`.

---

## 2. `releases.json` — the canonical schema

A JSON **array**, **newest-first**. Writers MUST emit a bare array; readers also tolerate a `{ "releases": [ ... ] }` wrapper for forward-compat, but do not write that form.

Each entry is a `Release` object (TypeScript type in `src/lib/productCatalog.ts`):

```ts
export type ReleaseType = 'release' | 'patch' | 'hotfix';
export type ReleaseChannel = 'stable' | 'beta';

export interface ReleaseAsset { label: string; file: string; size?: number; sha256?: string }

export interface Release {
  version: string;                    // semver, NO leading "v"        (required)
  date: string;                       // YYYY-MM-DD                    (required)
  latest?: boolean;                   // exactly one stable entry true (derived if absent)
  type?: ReleaseType;                 // default 'release'
  channel?: ReleaseChannel;           // default 'stable'
  severity?: 'normal' | 'critical';   // default 'normal'
  title?: string;                     // short headline               (optional)
  notes: string;                      // markdown / plain text        (REQUIRED)
  assets?: { windows?: ReleaseAsset[]; mac?: ReleaseAsset[]; linux?: ReleaseAsset[] };
}
```

### 2.1 Every field, documented

| Field | Type / allowed values | Required | Meaning |
|---|---|---|---|
| `version` | semver string, **no** leading `v` (e.g. `0.5.1`) | **yes** | The release version. The publisher strips a leading `v` if you pass one. |
| `date` | `YYYY-MM-DD` | **yes** | Publish date. No format validation beyond presence — get it right. |
| `latest` | boolean | no (derived) | Marks the entry the site/Hub treat as current. Exactly **one stable** entry is `true`; the publisher recomputes this for the whole list on every publish. Beta entries are never `latest`. |
| `type` | `release` \| `patch` \| `hotfix` | no | Defaults to `release`. Cosmetic/semantic label shown on the releases page. |
| `channel` | `stable` \| `beta` | no | Defaults to `stable`. Only `stable` entries can be `latest` and feed `version.json`. `beta` ships to opt-in (`?channel=beta`) users without disturbing the stable pointer. |
| `severity` | `normal` \| `critical` | no | Defaults to `normal`. `critical` is highlighted on the site (use for security/data-loss hotfixes). |
| `title` | string | no | Short headline (e.g. "Context-menu auto-install"). When present it is preferred over `notes` for the derived `version.json.notes`. |
| `notes` | Markdown / plain-text string | **yes** | User-facing changelog. May be short, but must be non-empty. See §7 for how to write these. |
| `assets` | `{ windows?: ReleaseAsset[]; mac?: ReleaseAsset[]; linux?: ReleaseAsset[] }` | no | ≥1 OS for downloadable (desktop) products. **Omit entirely** for npm/web products — omitted `assets` = non-downloadable, which is valid. |

> A `minOS: { windows, mac, linux }` field appears in a spec example only; it is optional and not emitted by the publishers.

### 2.2 `ReleaseAsset`

```ts
{ label: string; file: string; size?: number; sha256?: string }
```

| Field | Required | Meaning |
|---|---|---|
| `label` | **yes** | Human OS/build label. The publisher writes `Windows (x64)`, `macOS`, `Linux (AppImage)`. |
| `file` | **yes** | Path **relative to `apps/<slug>/`** — e.g. `v0.5.1/XENO-Comms Setup 0.5.1.exe`. **Never absolute.** The `v<version>/` prefix is part of it. |
| `size` | no | Size in bytes (integer). The publisher fills this from `statSync`. `0`/`""` allowed but discouraged. |
| `sha256` | no | Lowercase hex SHA-256 of the installer. The publisher streams and fills this. |

Each OS key holds an **array** of assets; the first (`[0]`) is the primary installer (what the download deep-link 302s to).

---

## 3. `version.json` — the derived "latest stable" pointer

`version.json` is **always regenerated** from the newest stable `releases.json` entry — never authored by hand. It exists for backward-compat with the XENO Hub auto-updater, which reads this file and constructs `…/apps/<slug>/v<version>/<windows|mac|linux>`. **The per-OS values are FILENAMES only** (no `v<version>/` prefix — Hub joins that itself). Do not break this shape.

### Desktop (installer) shape

```jsonc
{
  "version": "0.5.1",
  "date": "2026-06-20",
  "windows": "XENO-Comms Setup 0.5.1.exe",   // FILENAME only, not a path
  "mac": "XENO-Comms-0.5.1-arm64.dmg",
  "linux": "XENO-Comms-0.5.1.AppImage",
  "notes": "Context-menu auto-install"         // title preferred, capped at 400 chars
}
```

- `version` / `date` — copied from the latest stable release.
- `windows` / `mac` / `linux` — the filename from that release's `assets[os][0].file`, with the `v<version>/` prefix stripped. Only keys that have an asset appear.
- `notes` — `latestStable.title || latestStable.notes`, truncated to **400 characters**. (This is why a punchy `title` matters — see §7.)

If there is **no stable entry** (e.g. a beta-only history), `version.json` is **not written** at all.

### CLI / npm shape

CLI products have no installers, so `version.json` is npm-shaped — no `windows`/`mac`/`linux` keys, plus two CLI-specific fields:

```jsonc
{
  "version": "0.4.0",
  "date": "2026-06-28",
  "npm": "@xeno-corporation/xeno-agent-cli",
  "install": "npm install -g @xeno-corporation/xeno-agent-cli",
  "notes": "• Added deterministic tool-call replay\n• Fixed …"
}
```

---

## 4. How `latest` and `channel` are computed

**Rule: the newest `stable` entry is `latest`. Beta entries are never `latest`.**

This is enforced in three places, consistently:

1. **`xeno-release.mjs` (on publish).** After prepending the new entry, it walks the list newest-first and sets the **first** entry whose `(channel ?? 'stable') === 'stable'` to `latest: true`, and **every other** entry to `latest: false`. `version.json` is then regenerated from `next.find(r => (r.channel ?? 'stable') === 'stable')` — the same latest-stable entry.
2. **`publish-cli-releases.mjs`.** Sets `latest: v === reg['dist-tags'].latest` — the entry matching npm's `latest` dist-tag is flagged.
3. **Website reader (`latestRelease()` in `productCatalog.ts`).** `releases.find(r => r.latest) ?? releases[0]` — uses the flagged entry, else falls back to the first (newest) entry.

**Consequences:**

- Publishing a new stable release automatically demotes the previous stable release's `latest` to `false`. You never edit old entries by hand.
- A `beta` publish adds a `channel: 'beta'` entry with `latest: false` and **does not touch** `version.json` or the stable pointer. Beta users reach it via `?channel=beta` on the download route; everyone else keeps the stable build.
- Exactly one stable entry is ever `latest: true`.

---

## 5. Worked examples

The examples below use `<slug>` = the product's R2 folder / catalog slug and installer filenames from a hypothetical `xeno-<app>` build. Substitute your real values.

### 5.1 A release (new feature version, all three OS)

Invocation (run from the **`xeno-platform`** repo):

```bash
node scripts/xeno-release.mjs publish \
  --app <slug> --version 0.5.0 --date 2026-06-14 \
  --type release --channel stable \
  --title "Threads, presence, and agent members" \
  --notes-file ../xeno-<app>/CHANGELOG-0.5.0.md \
  --win   "../xeno-<app>/release/XENO-Comms Setup 0.5.0.exe" \
  --mac   "../xeno-<app>/release/XENO-Comms-0.5.0-arm64.dmg" \
  --linux "../xeno-<app>/release/XENO-Comms-0.5.0.AppImage"
```

Resulting `releases.json` entry (prepended, newest-first):

```json
{
  "version": "0.5.0",
  "date": "2026-06-14",
  "latest": true,
  "type": "release",
  "channel": "stable",
  "severity": "normal",
  "title": "Threads, presence, and agent members",
  "notes": "- Real-time presence and typing indicators across every device\n- Threads: reply in-line without losing the main channel\n- Agents can now join a conversation as first-class members",
  "assets": {
    "windows": [{ "label": "Windows (x64)",    "file": "v0.5.0/XENO-Comms Setup 0.5.0.exe",    "size": 98231145,  "sha256": "a1b2c3d4e5f6…" }],
    "mac":     [{ "label": "macOS",             "file": "v0.5.0/XENO-Comms-0.5.0-arm64.dmg",     "size": 91002310,  "sha256": "c3d4e5f6a1b2…" }],
    "linux":   [{ "label": "Linux (AppImage)",  "file": "v0.5.0/XENO-Comms-0.5.0.AppImage",      "size": 103882910, "sha256": "e5f6a1b2c3d4…" }]
  }
}
```

Derived `version.json`:

```json
{
  "version": "0.5.0",
  "date": "2026-06-14",
  "windows": "XENO-Comms Setup 0.5.0.exe",
  "mac": "XENO-Comms-0.5.0-arm64.dmg",
  "linux": "XENO-Comms-0.5.0.AppImage",
  "notes": "Threads, presence, and agent members"
}
```

### 5.2 A patch (small fix, same channel)

Only `--type patch` changes. Everything else is the same pipeline.

```bash
node scripts/xeno-release.mjs publish \
  --app <slug> --version 0.5.1 --date 2026-06-20 \
  --type patch --channel stable \
  --title "Context-menu auto-install" \
  --notes-file ../xeno-<app>/CHANGELOG-0.5.1.md \
  --win   "../xeno-<app>/release/XENO-Comms Setup 0.5.1.exe" \
  --mac   "../xeno-<app>/release/XENO-Comms-0.5.1-arm64.dmg" \
  --linux "../xeno-<app>/release/XENO-Comms-0.5.1.AppImage"
```

New entry (prepended; the 0.5.0 entry above is automatically flipped to `latest: false`):

```json
{
  "version": "0.5.1",
  "date": "2026-06-20",
  "latest": true,
  "type": "patch",
  "channel": "stable",
  "severity": "normal",
  "title": "Context-menu auto-install",
  "notes": "- Fix: right-click → Install now resolves the correct installer\n- Fix: presence dot no longer sticks after network drop",
  "assets": { "windows": [ … ], "mac": [ … ], "linux": [ … ] }
}
```

### 5.3 A hotfix (urgent, critical severity)

Add `--type hotfix --severity critical`. `critical` highlights the entry on the site.

```bash
node scripts/xeno-release.mjs publish \
  --app <slug> --version 0.5.2 --date 2026-06-21 \
  --type hotfix --channel stable --severity critical \
  --title "Fix data loss on quit" \
  --notes "- Critical: unsaved drafts were discarded when quitting mid-sync. Fixed.\n- Upgrade immediately." \
  --win   "../xeno-<app>/release/XENO-Comms Setup 0.5.2.exe" \
  --mac   "../xeno-<app>/release/XENO-Comms-0.5.2-arm64.dmg" \
  --linux "../xeno-<app>/release/XENO-Comms-0.5.2.AppImage"
```

New entry:

```json
{
  "version": "0.5.2",
  "date": "2026-06-21",
  "latest": true,
  "type": "hotfix",
  "channel": "stable",
  "severity": "critical",
  "title": "Fix data loss on quit",
  "notes": "- Critical: unsaved drafts were discarded when quitting mid-sync. Fixed.\n- Upgrade immediately.",
  "assets": { "windows": [ … ], "mac": [ … ], "linux": [ … ] }
}
```

After these three publishes, `releases.json` is `[0.5.2, 0.5.1, 0.5.0]` (newest-first), only `0.5.2` has `latest: true`, and `version.json` points at `0.5.2`.

### 5.4 A beta (optional, opt-in channel)

```bash
node scripts/xeno-release.mjs publish \
  --app <slug> --version 0.6.0-beta.1 --date 2026-06-25 \
  --type release --channel beta \
  --title "New composer (beta)" \
  --notes "- Preview of the redesigned composer. Feedback welcome." \
  --win "../xeno-<app>/release/XENO-Comms Setup 0.6.0-beta.1.exe"
```

The entry gets `channel: "beta"`, `latest: false`, and **`version.json` is untouched** — stable users stay on `0.5.2`. Beta testers download it via `https://xenostudio.ai/product/<slug>/download/win?channel=beta`.

---

## 6. The two publish paths

There is exactly one publisher per distribution model. **Pick by how the product ships**, not by preference.

> **There is no separate `xeno-release` binary.** `xeno-release` is the *name of a script* you run as `node scripts/xeno-release.mjs` from the `xeno-platform` repo — it is **not** a compiled/standalone executable on your `PATH`, and it is not installed globally. These two `.mjs` scripts (plus the one-off `seed-releases.mjs` in §8) are the entire publisher toolchain. When docs or commit messages say "run `xeno-release`", they mean `node scripts/xeno-release.mjs`.

### 6.1 Desktop / installer products → `scripts/xeno-release.mjs`

This is the **canonical** publisher for anything with downloadable installers (Hub, Pixel, Motion, Sound, Comms, Canvas, …). It uploads the installers, computes `size` + `sha256` for each, prepends a full `Release` entry to `releases.json`, recomputes `latest`, and regenerates `version.json`.

**Subcommand:** the first positional arg **must** be `publish`. Anything else (or nothing) just prints usage.

**Flags:**

| Flag | Required | Notes |
|---|---|---|
| `--app <slug>` | **yes** | R2 folder / product slug. |
| `--version <x.y.z>` | **yes** | A leading `v` is stripped automatically. |
| `--date <YYYY-MM-DD>` | **yes** | Presence-checked only — format it correctly. |
| `--channel stable\|beta` | no | Anything not exactly `beta` becomes `stable`. |
| `--type release\|patch\|hotfix` | no | Invalid/absent → `release`. |
| `--severity normal\|critical` | no | Anything not `critical` → `normal`. |
| `--title "..."` | no | Added to the entry only when non-empty; preferred for `version.json.notes`. |
| `--notes "..."` **or** `--notes-file FILE` | **yes** (one of) | `--notes-file` is read and `.trim()`-med. Empty → the script fails. |
| `--win <path>` / `--mac <path>` / `--linux <path>` | no | Local installer paths. Provide the OSes you built. (Note the flag is `--win`, not `--windows`.) |
| `--dry-run` | no | Prints every `rclone` command instead of executing — nothing is uploaded. |

**What it does, in order:**

1. For each provided installer: `size = statSync(path).size`, `sha256` (streamed, lowercase hex), `fname = basename(path)`, then `rclone copyto <path> r2:xeno-hub-releases/apps/<slug>/v<version>/<fname> --no-traverse`. Builds `assets[os] = [{ label, file: "v<version>/<fname>", size, sha256 }]`.
2. Builds the `Release` object (`latest: channel === 'stable'` initially; `title` and `assets` included only when present).
3. Fetches the existing `releases.json`, **dedupes by (version, channel)**, prepends the new entry, and recomputes `latest` across the whole list (first stable = `latest`).
4. Regenerates `version.json` from the latest stable entry (filenames only; `notes` = `title || notes` capped at 400 chars). Skips writing it if no stable entry exists.
5. Uploads both JSON files with `--header-upload "Cache-Control: no-cache" --no-traverse`.
6. Prints a reminder to trigger a product-pages prerender + frontend deploy (**it does not run that itself** — see §9).

**Canonical invocation** (from `RELEASE-TO-WEBSITE.md` §0):

```bash
node scripts/xeno-release.mjs publish \
  --app <slug> --version <x.y.z> --date <YYYY-MM-DD> \
  --notes "<...>" [--win <installer.exe>] [--mac <app.dmg>] [--linux <app.AppImage>]
```

> A **CLI/SDK** product can also be published through `xeno-release.mjs` with no `--win/--mac/--linux` (per `RELEASE-TO-WEBSITE.md` §3.A: `--app agent-cli --version 0.4.0 --date … --type release --notes "$(cat CHANGELOG-0.4.0.md)"`). That produces a `releases.json` entry with no `assets`. Use this when you want a single hand-written entry; use §6.2 when you want the whole feed auto-derived from npm.

### 6.2 CLI / npm products → `scripts/publish-cli-releases.mjs`

CLI products (agent-cli, sdk) have no installers, so the installer/auto-update machinery of `xeno-release.mjs` doesn't apply. This script builds the **entire feed from real data** — nothing is authored or fabricated:

- **versions + dates** come from the **npm registry** (source of truth for what's actually installable),
- **notes** come from the CLI's own **`RELEASE_NOTES`** map (the exact text the CLI shows at startup),
- the feed is the **intersection** (versions that are BOTH on npm AND carry notes), newest-first, with npm's `latest` dist-tag flagged `latest`.

**Flags:**

| Flag | Required | Notes |
|---|---|---|
| `--app <slug>` | **yes** | R2 folder / product slug. |
| `--pkg <name>` | **yes** | npm package name, e.g. `@xeno-corporation/xeno-agent-cli`. |
| `--notes <path>` | **yes** | Path to the CLI's `release-notes.ts` (where `RELEASE_NOTES` lives). |
| `--out <dir>` | no | Output dir for the built JSON. Default `$TEMP/cli-feed-<APP>`. |
| `--dry-run` | no | Prints the `rclone` commands instead of uploading. |

**Invocation:**

```bash
node scripts/publish-cli-releases.mjs \
  --app agent-cli \
  --pkg @xeno-corporation/xeno-agent-cli \
  --notes ../xeno-agent-cli/apps/xeno-agent-cli/src/commands/release-notes.ts \
  [--out dist-feed] [--dry-run]
```

**What it does:**

1. Fetches `https://registry.npmjs.org/<pkg>` (`{ cache: 'no-cache' }`) → `versions`, `time`, `dist-tags.latest`.
2. Textually extracts the `RELEASE_NOTES` object literal from the `.ts` file (regex-finds `RELEASE_NOTES … = {`, brace-matches to the close, strips trailing commas, `JSON.parse`). It does **not** execute the TS module.
3. `versions = keys(notes) ∩ (on npm with a publish date)`, sorted descending semver. If none → fails with `no versions with both an npm publish date and release notes`.
4. Emits one feed entry per version:

```json
{
  "version": "0.4.0",
  "date": "2026-06-28",
  "latest": true,
  "type": "release",
  "channel": "stable",
  "severity": "normal",
  "notes": "• Added deterministic tool-call replay\n• Fixed streaming cutoffs on slow links",
  "install": "npm install -g @xeno-corporation/xeno-agent-cli"
}
```

  - `type` / `channel` / `severity` are **always** `release` / `stable` / `normal`.
  - `latest` matches npm's `latest` dist-tag.
  - `notes` is the `RELEASE_NOTES[version]` string array joined as `• <item>` bullets.
  - `install` is a CLI-only convenience field (not part of the `Release` TS type; the site renders notes, not assets).

5. Builds a CLI-shaped `version.json` (`version`, `date`, `npm`, `install`, `notes` — no OS keys).
6. Writes both files to `<out>` and pushes each with `rclone copyto <local> r2:xeno-hub-releases/apps/<app>/<file> --header-upload "Cache-Control: no-cache" --no-traverse`.
7. Prints the page (`https://xenostudio.ai/product/<app>/releases`) and feed (`https://updates.xenostudio.ai/apps/<app>/releases.json`).

> To publish a CLI feed you must first `npm publish` the new version (so it exists on the registry) **and** add its bullets to `RELEASE_NOTES` in `release-notes.ts`. A version present in only one of the two is silently excluded from the feed.

### 6.3 Which script?

| Product ships as… | Use | Result |
|---|---|---|
| Desktop installers (`.exe`/`.dmg`/`.AppImage`) | `xeno-release.mjs` | `assets` per OS + `version.json` with OS filenames + auto-update. |
| npm package (CLI/SDK), feed auto-derived | `publish-cli-releases.mjs` | Full feed mirrored from npm + `RELEASE_NOTES`; npm-shaped `version.json`. |
| npm package, single hand-written entry | `xeno-release.mjs` (no `--win/--mac/--linux`) | One `assets`-less entry you author directly. |

---

## 7. Writing good release notes

Release notes are **user-facing**. They land on `https://xenostudio.ai/product/<slug>/releases`, in the Hub update prompt, and (truncated) in `version.json`. Write for the person deciding whether to update — not for the changelog reviewer.

**Principles:**

- **Concise and outcome-first.** Say what changed *for the user*, not how you implemented it. "Right-click → Install now resolves the correct installer" beats "refactored installer resolution in `preload.ts`".
- **One line per change.** Short bullets. Lead with the most impactful item.
- **Plain language, no internals.** No commit hashes, PR numbers, file paths, or ticket IDs.
- **Group when long.** For a big release, group under `Added` / `Changed` / `Fixed`. For a patch or hotfix, a few bullets is plenty.
- **Title is a headline.** Keep `--title` to a short, punchy phrase (roughly ≤ 60 chars). It is preferred for `version.json.notes`, which is **capped at 400 characters** — so a good title guarantees a clean Hub update prompt even when the full notes are long.
- **Mark urgency honestly.** Use `--type hotfix --severity critical` for security/data-loss fixes and say "upgrade immediately" in the notes; `critical` is highlighted on the site.

**Desktop (`xeno-release.mjs`)** — pass Markdown via `--notes "..."` or, preferably, `--notes-file CHANGELOG-<version>.md` so the notes are reviewed in the product repo. Example body:

```markdown
- Threads: reply in-line without losing the main channel
- Real-time presence and typing indicators across every device
- Fix: presence dot no longer sticks after a network drop
```

**CLI (`publish-cli-releases.mjs`)** — follow the `RELEASE_NOTES` pattern: a map from version → array of short user-facing strings, in the CLI's `release-notes.ts`. The publisher renders each string as a `•` bullet, so keep each entry a single self-contained line:

```ts
export const RELEASE_NOTES: Record<string, string[]> = {
  '0.4.0': [
    'Added deterministic tool-call replay',
    'Fixed streaming cutoffs on slow links',
    'New `--json` output for scripting',
  ],
  '0.3.2': [
    'Fixed auth token refresh loop',
  ],
};
```

Because these strings become website copy verbatim, edit them with the same care as marketing text.

---

## 8. `seed-releases.mjs` — one-off bootstrap (not per-release)

For a product that has a `version.json` on R2 but **no `releases.json` yet**, `seed-releases.mjs` backfills a single latest-stable entry so the history file exists.

```bash
node scripts/seed-releases.mjs [<slug> …]        # default slugs: hub pixel motion sound
```

It fetches each product's existing `version.json`, synthesizes one `Release` (`latest: true`, `type: 'release'`, `channel: 'stable'`, `severity: 'normal'`, `assets[os].file = "v<version>/<v[os]>"`), and uploads it with `rclone copyto <file> r2:xeno-hub-releases/apps/<slug>/releases.json --header-upload "Cache-Control: no-cache" --no-traverse`. It never touches `version.json` or installers. Run it **once** to bootstrap; use §6 for all subsequent releases.

---

## 9. After publishing: cache rules, verification, and deploy

### 9.1 Cache-Control (set by the scripts)

| Resource | `Cache-Control` | Why |
|---|---|---|
| `releases.json` | `no-cache` (via `--header-upload`) | Site/Hub must see new releases immediately. |
| `version.json` | `no-cache` (via `--header-upload`) | Same. |
| Installers `v<version>/*` | **not set by the scripts** — bucket default; spec §9 mandates `public, max-age=31536000, immutable` | Installers are content-addressed by version and never change. |
| `/product/<slug>/download/<os>` 302 | `no-store` (set on the redirect by the backend route) | The deep-link must always resolve to the *current* installer. |

**Immutability rule:** never overwrite an existing `v<version>/` installer — versions are permanent. To fix a bad build, publish a new version. `releases.json` is full history: **prepend, never replace**.

### 9.2 Verify the publish

```bash
# feed and pointer are live and no-cache:
curl -s  https://updates.xenostudio.ai/apps/<slug>/releases.json | head
curl -sI https://updates.xenostudio.ai/apps/<slug>/version.json

# the stable download deep-link 302s to the new installer:
curl -sI "https://xenostudio.ai/product/<slug>/download/win"     # expect 302
```

The download route normalizes the OS segment (`win|windows`, `mac|macos|osx`, `linux|appimage`), 30 s-caches `releases.json` per slug, and 302s to the primary asset. On bad input it returns a 404 JSON error (`BAD_SLUG`, `BAD_OS`, `NO_RELEASES`, `NO_RELEASE`, `NO_ASSET`) rather than redirecting to the wrong file.

### 9.3 Publishing does NOT deploy the website — but SEO pages need a rebuild

The live releases page and download CTAs read `releases.json` **live**, so a new version appears with **no platform deploy**. However, `xeno-release.mjs` prints a reminder (SPEC §13.6) to re-run the product-pages prerender + frontend deploy so the **static/SEO** HTML reflects the new version. That rebuild + deploy step is a separate concern — it is covered by the deploy guide in this set (operational source: `RELEASE-TO-WEBSITE.md` §6/§8 and `PRODUCT-LANDING-SPEC.md` §8). Landing content and docs changes always require that deploy; release data alone does not.

---

## 10. Rules checklist (from `RELEASE-TO-WEBSITE.md` §7)

- The **slug is identical everywhere** — catalog, `--app`, R2 path, download route.
- **Semver, no leading `v`** (the publisher strips it if you pass one).
- **Always publish BOTH** `releases.json` and `version.json` (the publisher does this for you).
- **Never overwrite** an existing `v<version>/` installer — installers are immutable.
- **`releases.json` is full history** — prepend, don't replace.
- **Exactly one stable entry has `latest: true`** — the publisher recomputes it; don't hand-edit.
- **Installers cache forever; the two JSON files are `Cache-Control: no-cache`.**

---

## 11. Cross-references

- Product identity / catalog entry (`src/lib/productCatalog.ts`) and the `Product` type — see the catalog/identity file in this set.
- Landing content and documentation authoring (compiled + prerendered → deploy required) — see the landing and docs files in this set.
- Frontend build, prerender, and on-box deploy — see the deploy guide in this set (operational source: `PRODUCT-LANDING-SPEC.md` §8/§9 and `RELEASE-TO-WEBSITE.md`).
- Canonical scripts (in `xeno-platform/scripts/`): `xeno-release.mjs`, `publish-cli-releases.mjs`, `seed-releases.mjs`.
- Reader/type source of truth: `xeno-platform/src/lib/productCatalog.ts`; download route: `xeno-platform/src/server/routes/productDownloadRoutes.js`.
