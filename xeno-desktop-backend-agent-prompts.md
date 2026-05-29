# XENO Desktop Backend Agent Prompts

These prompts are intentionally small and iterative.

Run them on the **VPS Codex agent itself**, not on the local repo.

Use the remote pattern documented in:
- [claude-to-codex.md](/X:/code/xeno-corporation/xeno-rt/claude-to-codex.md)

Architecture reference for the humans coordinating this:
- [XENO_DESKTOP_BACKEND_CONTRACT.md](/X:/bunkercloud/docs/XENO_DESKTOP_BACKEND_CONTRACT.md)

Do not ask the local repo agent to implement these directly in `X:\bunkercloud`. The target is the live VPS code.

## Prompt 1: `xeno-platform-001` Desktop Auth Contract

Run on:
- `xeno-platform-001`

Workdir:
- `/home/bunker/xeno-platform`

```text
You are Codex running directly on xeno-platform-001 in /home/bunker/xeno-platform.

Task:
Implement the desktop-auth backend contract on xenostudio.ai without breaking the canonical account/billing system.

Required contract:
1. POST /api/auth/login must return:
   - token
   - apiKey
   - user
2. POST /api/auth/register must return:
   - token
   - apiKey
   - user
3. GET /api/auth/validate must return:
   - valid
   - user
   - apiKey
4. The apiKey returned must be a real usable xeno-... key provisioned from the canonical API-key registry, not a fake placeholder.
5. Public content routes must remain public:
   - GET /api/blog?limit=N&category=X
   - GET /api/blog/categories
   - GET /api/learn?limit=N

Critical architectural rules:
- xeno-platform-001 remains the source of truth for account, billing, and canonical API-key ownership.
- Do NOT create a separate desktop-only key store.
- Do NOT expose a public internal provisioning endpoint to the desktop app.
- If a default desktop/cloud key does not exist, create or recover it through the canonical platform API-key path.
- Do NOT regress MFA, sessions, or OAuth behavior.
- Back up files before editing.

Files to inspect first:
- src/server/routes/authRoutes.js
- src/server/index.js
- any canonical API-key provisioning/runtime helpers already in the codebase

Important note:
- There are legacy auth handlers in index.js and newer canonical auth handlers in routes/authRoutes.js.
- Audit the live route mounting carefully and fix the real live path, not just one copy of the code.

After implementation:
- rebuild/restart only what is necessary
- verify live:
  - POST /api/auth/login returns token + apiKey + user
  - POST /api/auth/register returns token + apiKey + user
  - GET /api/auth/validate returns valid + user + apiKey
  - returned apiKey is in xeno-... format and is actually usable
  - public blog and learn endpoints still return 200 without auth

Final report:
- exact files changed
- exact services rebuilt/restarted
- exact curl/live verifications performed
- any remaining auth-contract gaps for the desktop app
```

## Prompt 2: `xeno-private-api-001` Desktop Proxy Compatibility

Run on:
- `xeno-private-api-001`

Primary roots:
- `/home/bunker/apps/xeno-api-proxy`
- `/home/bunker/apps/xeno-api-platform/portal`

```text
You are Codex running directly on xeno-private-api-001.

Task:
Implement the desktop proxy compatibility layer needed by the XENO desktop app, while preserving canonical platform authority.

Required contract:
1. Keep and verify:
   - POST /v1/chat/completions with Bearer apiKey
   - GET /v1/models with Bearer apiKey
2. Add or verify desktop compatibility read endpoints:
   - GET /credits/balance with Bearer web JWT
   - GET /credits/transactions?limit=N with Bearer web JWT
3. Those credit endpoints must read canonical billing/account data from the platform-backed source of truth.
4. Do NOT create any separate API-side credit authority.
5. Do NOT expose /internal/provision-key as a public desktop endpoint.

Architectural rules:
- api.xenostudio.ai is the execution surface, not the billing authority.
- JWT-authenticated credit reads here are allowed only as a compatibility layer backed by canonical platform data.
- Do NOT reintroduce local proxy-side balance logic, local usage authority, or local API-key authority.
- Back up files before editing.

Files to inspect first:
- /home/bunker/apps/xeno-api-proxy/server.js
- any portal compatibility/auth helpers if needed

After implementation:
- restart only what is necessary
- verify live:
  - GET /v1/models works with Bearer apiKey
  - POST /v1/chat/completions works with Bearer apiKey
  - GET /credits/balance works with Bearer web JWT
  - GET /credits/transactions?limit=N works with Bearer web JWT
  - credit responses reflect canonical platform data, not a separate local balance

Final report:
- exact files changed
- exact services restarted
- exact live verifications performed
- whether any desktop proxy compatibility gap still remains
```

## Prompt 3: `xeno-platform-001` OAuth MFA And Desktop Session Restore

Run on:
- `xeno-platform-001`

Workdir:
- `/home/bunker/xeno-platform`

```text
You are Codex running directly on xeno-platform-001 in /home/bunker/xeno-platform.

Task:
Audit and finish OAuth MFA parity and desktop session-restore behavior for the desktop auth contract.

Goals:
1. If MFA is enabled, OAuth sign-in must require MFA step-up before issuing the final session.
2. GET /api/auth/validate must be strong enough for desktop session restore:
   - valid
   - user
   - apiKey
3. Desktop session restore must not silently lose cloud capability just because the stored API key is missing or stale.
   If necessary, validate should repair/recover the default desktop key through the canonical key path.

Do not:
- weaken password-login MFA
- weaken session revocation
- introduce a second auth path
- bypass canonical API-key ownership

Inspect first:
- src/server/routes/authRoutes.js
- any helpers involved in canonical API-key provisioning and auth payload generation

After implementation:
- rebuild/restart only what is needed
- verify:
  - MFA overview still works
  - OAuth callback path still works
  - validate returns valid + user + apiKey
  - a fresh login and a session-restore path both end with a usable apiKey for desktop cloud calls

Final report:
- exact files changed
- exact services rebuilt/restarted
- exact live verifications performed
- whether OAuth/session-restore parity is complete
```

## Prompt 4: Legacy desktop sync retirement review

Run on:
- `xeno-platform-001`

```text
You are Codex running directly on xeno-platform-001.

Task:
Audit any remaining legacy desktop sync assumptions for xeno-hub and remove stale runtime values no longer needed by the desktop app.

Required checks:
1. No retired collaboration host is required for the desktop app to function
2. xeno-hub is published and usable
3. server_config contains correct values for:
   - LLM_BASE_URL = https://api.xenostudio.ai/v1
   - LLM_API_KEY = current internal proxy master key or current approved master key path
   - ADMIN_IDENTITY = current correct publisher/admin identity
4. Do not make any sync layer the billing authority.
5. Confirm whether the desktop app should treat my_credits / my_credit_transactions as mirror/read-only convenience state rather than source of truth.

Do not:
- redesign xeno-hub
- expose ADMIN_IDENTITY as an app concern
- move canonical billing into a sync layer

Verification:
- public reachability
- database availability
- server_config correctness
- reducer/procedure/module readiness relevant to desktop app bootstrap

Final report:
- exact changes made, if any
- exact commands and outputs summarized
- any remaining Spacetime desktop-readiness gaps
```

## Prompt 5: Final Desktop Backend Integration Sweep

Run last, after prompts 1-4.

Host choice:
- whichever host owns the remaining gap, or run separately on both

```text
You are Codex running on the live Xeno VPS environment.

Task:
Perform a final desktop-backend integration sweep and report whether the backend is ready for the XENO desktop app, excluding live payment rollout.

Approved desktop chain:
1. desktop login/register on xenostudio.ai returns token + apiKey
2. desktop stores both
3. desktop uses JWT for account/profile/billing reads
4. desktop uses apiKey for /v1/models and /v1/chat/completions
5. api.xenostudio.ai deducts credits through canonical platform-backed billing logic
6. Realtime state remains separate from billing authority

Validate:
- auth contract
- public content contract
- proxy contract
- credits/billing read contract
- spacetime readiness

Final report:
- READY or NOT READY for desktop backend integration
- exact remaining gaps, if any
- what must change in the desktop app itself to match the approved contract
```

## Recommended Run Order

1. Prompt 1 on `xeno-platform-001`
2. Prompt 2 on `xeno-private-api-001`
3. Prompt 3 on `xeno-platform-001`
4. Prompt 4 on `xeno-platform-001`
5. Prompt 5 on the host(s) that still own unresolved gaps






