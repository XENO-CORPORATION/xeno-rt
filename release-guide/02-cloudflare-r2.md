# 02 — Cloudflare R2 / CDN

Reference for the Cloudflare R2 bucket that serves every XENO release feed and installer: the bucket, the rclone remote, the public domain, the `apps/<slug>/` path layout, the cache policy, and how to upload + verify. General for any product `<slug>`.

---

## Prerequisites

You need **rclone** installed with an **`r2:` remote** already configured against the `xeno-hub-releases` bucket. The publisher scripts and every manual command below assume this remote exists — none of them create or configure it.

Check it before you publish:

```bash
rclone listremotes        # → you should see `r2:` in the output
```

If `r2:` is missing, configure the remote first (S3-compatible endpoint for the Cloudflare account that owns the `xeno-hub-releases` bucket). Configuring rclone is out of scope for this guide — confirm the account/endpoint with the infra owner.

> You normally never run rclone by hand. The publisher scripts (`node scripts/xeno-release.mjs`, `node scripts/publish-cli-releases.mjs`) do every upload for you. The raw commands below are documented so you can verify, repair, or bootstrap a feed manually.

---

## Fixed infrastructure constants

These are hard infrastructure values — they are the same in every product repo and every script. Do **not** placeholder-swap them.

| Thing | Exact value |
|---|---|
| R2 bucket | `xeno-hub-releases` |
| rclone remote | `r2:` → remote path `r2:xeno-hub-releases` |
| Public read domain (CDN) | `https://updates.xenostudio.ai` |
| Website product page | `https://xenostudio.ai/product/<slug>` (SINGULAR `/product/`) |
| Per-product root on R2 | `apps/<slug>/` |

The public domain maps 1:1 onto the bucket: `https://updates.xenostudio.ai/apps/<slug>/...` serves the object at `r2:xeno-hub-releases/apps/<slug>/...`.

### `<slug>` vs the `r2` folder override

`<slug>` is the product's catalog slug (identical everywhere — repo, catalog entry, R2 folder, website route). A product may override **only** its R2 folder name via an `r2` field in `src/lib/productCatalog.ts` (the reader resolves `p.r2 ?? p.slug`). In practice slug and R2 folder are the same for all products except where a catalog entry sets `r2` explicitly (e.g. the browser extension uses `r2: 'extension'`). Throughout this guide `<slug>` means "the R2 folder", i.e. the resolved `r2 ?? slug`.

---

## Path layout under `apps/<slug>/`

Every product owns one folder on R2. Its shape:

```
updates.xenostudio.ai/apps/<slug>/
├── releases.json            ← CANONICAL full history (the site reads this)
├── version.json             ← DERIVED "latest stable" pointer (Hub auto-update reads this)
├── v<version>/
│   └── <InstallerFilename>  ← e.g. v0.6.4/XENO Pixel Setup 0.6.4.exe
├── v<version>/
│   └── <InstallerFilename>
└── ...
```

| Path | What it is |
|---|---|
| `apps/<slug>/releases.json` | Canonical release history — a newest-first JSON array of every release. The website fetches this live on page load. |
| `apps/<slug>/version.json` | Derived pointer to the latest **stable** release, flattened to the legacy Hub shape (filenames only). Hub's auto-updater reads this. |
| `apps/<slug>/v<version>/<InstallerFilename>` | The actual installer for one version+OS. Path is `v<X.Y.Z>/<filename>`; the version is **semver, no leading `v`** on the version string but the folder is literally `v` + version. |

Notes:
- `<version>` is semver with **no leading `v`** in the JSON `version` field; the installer **folder** is `v<version>` (e.g. version `0.6.4` → folder `v0.6.4/`).
- Inside `releases.json`, an asset's `file` is **relative to `apps/<slug>/`** — e.g. `v0.6.4/XENO Pixel Setup 0.6.4.exe`, never an absolute URL.
- Inside `version.json`, the per-OS values are **filenames only** (no `v<version>/` prefix) — Hub joins `v{version}/` itself. Do not break this shape.
- CLI/npm products (agent-cli, sdk) have **no installers**: their `apps/<slug>/` holds only `releases.json` + `version.json` (npm-shaped, carrying `npm` + `install` instead of `windows`/`mac`/`linux`).

For the full field-by-field schema of `releases.json` and `version.json`, see **03-release-data.md**. This file only covers where those objects live on R2 and how they get there.

---

## Cache policy (critical)

The two JSON pointers must be re-read the instant they change, so Hub and the website see a new release immediately. The installers never change once published, so they cache forever.

| Resource | `Cache-Control` | Why |
|---|---|---|
| `apps/<slug>/releases.json` | **`no-cache`** | Site reads it live; a stale copy hides the new release. |
| `apps/<slug>/version.json` | **`no-cache`** | Hub auto-update polls it; a stale copy hides the update. |
| `apps/<slug>/v<version>/<installer>` | not set by the upload (bucket default) | Installers are immutable per version; a version folder is written once and never overwritten. The spec mandates treating them as `public, max-age=31536000, immutable`. |

The `no-cache` header is set **at upload time** via rclone's `--header-upload` flag (see below). Installer uploads deliberately carry **no** cache header.

> A published version folder (`v<version>/`) is **immutable** — never overwrite an existing installer. Cut a new version instead. `releases.json` is full history: prepend, never replace.

---

## Uploading with rclone

### The normal path (automated)

You do not upload by hand for a normal release. The desktop publisher `node scripts/xeno-release.mjs publish ...` uploads the installer(s), prepends the new entry to `releases.json`, regenerates `version.json`, and pushes all three objects to R2 with the correct flags and cache headers. The CLI publisher `node scripts/publish-cli-releases.mjs ...` does the same for npm products (no installer). See the publisher-script reference for their invocation and behavior; this file documents only the R2 commands they run.

The exact rclone verb the scripts use is **`copyto`** (single-file copy-with-rename), with `--no-traverse` on every upload and `--header-upload "Cache-Control: no-cache"` on the two JSON files.

### The manual path (verify / repair / bootstrap)

If you ever need to push objects by hand — repairing a feed, seeding a new product, or fixing a bad upload — these are the exact commands. Substitute `<slug>`, `<version>`, and the installer path/filename.

**Installer** (immutable, no cache header). The operational how-to uses `copy` into the version folder:

```bash
rclone copy "<local-installer-path>" r2:xeno-hub-releases/apps/<slug>/v<version>/
```

The publisher script instead uses `copyto` with the explicit destination filename and `--no-traverse` (equivalent result, single object):

```bash
rclone copyto "<local-installer-path>" \
  r2:xeno-hub-releases/apps/<slug>/v<version>/<InstallerFilename> --no-traverse
```

**The two JSON pointers** (both carry `Cache-Control: no-cache`):

```bash
rclone copyto releases.json \
  r2:xeno-hub-releases/apps/<slug>/releases.json \
  --header-upload "Cache-Control: no-cache" --no-traverse

rclone copyto version.json \
  r2:xeno-hub-releases/apps/<slug>/version.json \
  --header-upload "Cache-Control: no-cache" --no-traverse
```

Flag summary:
- `copyto <local> <remote>` — copy a single file to an exact destination path (renames on copy). Scripts use `copyto` for both installers and JSON; the manual installer shortcut uses `copy <local> <dir>/` instead.
- `--header-upload "Cache-Control: no-cache"` — sets the object's `Cache-Control` header. **JSON only.** Never add this to an installer upload.
- `--no-traverse` — skips listing the destination first; used on every upload the scripts perform.

> Concrete worked example (from the operational how-to, for `pixel` v0.6.4):
> ```bash
> rclone copy "release/XENO Pixel Setup 0.6.4.exe" r2:xeno-hub-releases/apps/pixel/v0.6.4/
> rclone copyto releases.json r2:xeno-hub-releases/apps/pixel/releases.json --header-upload "Cache-Control: no-cache"
> rclone copyto version.json  r2:xeno-hub-releases/apps/pixel/version.json  --header-upload "Cache-Control: no-cache"
> ```

**A release is not complete until BOTH JSON files are updated.** Always publish `releases.json` and `version.json` together.

---

## Verifying with curl

After any publish (automated or manual), confirm the CDN is serving the new objects. The JSON pointers are `no-cache`, so these reflect the latest upload immediately.

```bash
# 1. Canonical history — newest entry should be your new version
curl -s https://updates.xenostudio.ai/apps/<slug>/releases.json | head

# 2. Derived pointer — headers + body of the latest-stable pointer
curl -sI https://updates.xenostudio.ai/apps/<slug>/version.json

# 3. Installer is reachable (downloadable products only)
curl -sI "https://updates.xenostudio.ai/apps/<slug>/v<version>/<InstallerFilename>"

# 4. Website stable deep-link must 302-redirect to the R2 installer
curl -sI "https://xenostudio.ai/product/<slug>/download/win"      # expect HTTP/… 302
```

What to look for:
- (1) The first entry in the array is your new `version`/`date`, and exactly one **stable** entry has `latest: true`.
- (2) `version.json` reports the latest stable `version`; the response headers include `Cache-Control: no-cache`.
- (4) `/product/<slug>/download/<os>` returns **302** (a redirect to the R2 installer), not 200 or 404. The `<os>` segment accepts `win`/`windows`, `mac`/`macos`/`osx`, `linux`/`appimage`. This deep-link is a stable backend redirect (`Cache-Control: no-store` on the 302) and never changes across versions.

If `releases.json`/`version.json` still show the old version after a successful upload, the cause is almost always a missing `--header-upload "Cache-Control: no-cache"` on the upload (an edge cache is holding a stale copy). Re-upload the JSON with the header.

---

## Environment overrides

Most tooling hardcodes the constants above, but two consumers accept env overrides (useful for a staging bucket/domain; leave unset for production):

| Env var | Default | Read by |
|---|---|---|
| `XENO_UPDATES_BASE` | `https://updates.xenostudio.ai` | `scripts/publish-cli-releases.mjs`, the backend download-redirect route (`src/server/routes/productDownloadRoutes.js`) |
| `XENO_R2_REMOTE` | `r2:xeno-hub-releases` | `scripts/publish-cli-releases.mjs` |

`scripts/xeno-release.mjs` and `scripts/seed-releases.mjs` **hardcode** the bucket + public domain (no env override). The website reader (`src/lib/productCatalog.ts`) hardcodes `R2_BASE = 'https://updates.xenostudio.ai'`.

---

## Related infrastructure (context, not R2)

Publishing to R2 does **not** require a website redeploy — the site fetches `releases.json` live at page load, so a new version/download appears with no platform deploy. Only *landing content* and *docs* changes require a rebuild + deploy of the `xenostudio-frontend` container on `xeno-platform-001` (repo path `/mnt/projects/xeno-platform`). That flow is separate from this file.

---

## See also

- **03-release-data.md** — full field-by-field schema of `releases.json` and `version.json`, and how `latest` is computed.
- The publisher scripts in `scripts/` — `xeno-release.mjs` (desktop/installer), `publish-cli-releases.mjs` (CLI/npm), `seed-releases.mjs` (one-off bootstrap of `releases.json` from an existing `version.json`).
- `RELEASE-TO-WEBSITE.md` in the platform repo — the operational end-to-end how-to these commands are drawn from.
