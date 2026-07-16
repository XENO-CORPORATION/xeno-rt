# Xeno Backend Live Closeout Prompts

Use these prompts with the orchestration pattern in [claude-to-codex.md](/X:/code/xeno-corporation/xeno-rt/claude-to-codex.md).

These prompts are for the remaining non-payment backend closeout work on:
- `xeno-platform-001`
- `xeno-private-api-001`

They assume the canonical backend direction already in place:
- `xeno-platform-001` is the source of truth for credits, workspaces, projects, API keys, and billing/account state.
- `xeno-private-api-001` portal and API proxy must not reintroduce local auth/billing authority.
- Remaining local portal auth/user tables are rollback-only debt and must only be retired after a confirmed stability window.

## Prompt 1: Live Closeout Sweep

```text
You are working on the live Xeno estate.

Targets:
- xeno-platform-001
- xeno-private-api-001

Goal:
Finish the remaining non-payment backend/account closeout so the backend can be considered production-finished from an auth/account/workspace/project/security perspective.

Current architectural constraints you must preserve:
- xeno-platform-001 is canonical for credits, account/workspace/project state, canonical API keys, sessions, security events, and billing/account data.
- Do NOT reintroduce local portal auth as an authority on xeno-private-api-001.
- Do NOT reintroduce local proxy-side API-key validation, local debit writes, or local balance authority.
- Do NOT touch live payment provider config unless you are only validating existing non-destructive read behavior.
- Do NOT drop legacy local portal auth/user tables yet unless the explicit retirement preconditions are satisfied and you document them.

Primary task:
Audit the live deployment and finish any remaining missing runtime/UI/backend work in these areas:

1. Workspace/project account flows on both shells
- Verify both xenostudio platform shell and api portal shell expose:
  - explicit workspace selection
  - explicit project selection / project activation
  - workspace member management
  - invite create/revoke/resend
  - invite delivery history visibility
  - workspace audit history visibility
  - workspace budget visibility/update
  - pending invite inbox
  - owner transfer
- If any of those are missing or broken on live, implement them end-to-end and deploy.

2. Notification center and invite delivery operator flow
- Verify notification-center UI works on both shells against canonical notification APIs.
- Verify invite resend creates/updates delivery rows and the latest delivery state is visible in the UI or route responses.
- If provider env is missing, confirm the system degrades to a visible pending/unconfigured state rather than silent failure.

3. Workspace/project ownership on billable/product paths
- Audit the remaining live product/billable routes on xeno-platform-001.
- Confirm they accept explicit requested workspace/project context and do not silently misattribute usage to a fallback project except through the central runtime default map.
- Grep and inspect at minimum:
  - ~/xeno-platform/src/server/routes/xenoRoutes.js
  - ~/xeno-platform/src/server/routes/videoRoutes.js
  - ~/xeno-platform/src/server/routes/authRoutes.js
  - ~/xeno-platform/src/server/utils/billingRuntime.js
  - ~/xeno-platform/src/server/utils/creditTransactions.js
- Fix and deploy any route still bypassing canonical workspace/project ownership.

4. OAuth MFA parity
- Confirm password-login MFA already works.
- Verify OAuth sign-ins for configured providers now require MFA step-up when MFA is enabled.
- Test live for at least one enabled MFA account if possible without breaking the account.
- If OAuth callback still mints a session before MFA, fix it.

5. Rollback-only portal auth/user debt
- Confirm portal app routes on xeno-private-api-001 do not depend on local Prisma/bcrypt/jsonwebtoken auth logic in production paths.
- Confirm remaining local portal auth/user tables are rollback-only.
- If the required stability window has NOT passed, do not drop them; instead:
  - snapshot the current state
  - document exact retirement readiness
  - leave a clear "ready to retire after stability window" report
- If the stability window HAS passed and the live portal is confirmed healthy:
  - archive/snapshot the old local tables first
  - execute the staged retirement safely
  - verify portal auth/login/session/profile still works afterward

Required working style:
- Back up files before editing.
- Use minimal, defensible production-safe changes.
- Do not just diagnose.
- Implement, rebuild/restart, and verify.
- Do not claim completion without live verification.

Suggested verification:
- platform shell:
  - /overview/workspaces
  - /overview/projects
  - /overview/security
  - /overview/notifications
- api portal:
  - /dashboard/workspaces
  - /dashboard/projects
  - /dashboard/security
  - /dashboard/notifications
- canonical APIs:
  - /api/account/overview
  - /api/workspaces
  - /api/projects
  - /api/billing/notifications
  - /api/auth/security/overview
- live runtime:
  - create/revoke/resend invite
  - inspect delivery history
  - inspect workspace audit history
  - update workspace budget
  - update project policy
  - verify a project-scoped restriction still blocks correctly on the API path

At the end, report:
- exact files changed
- exact services rebuilt/restarted
- what was verified live
- what remains, if anything, before the backend can be called done excluding payments
```

## Prompt 2: Portal Auth Retirement Only

Use this only after the live closeout sweep reports the stability window requirement is met.

```text
You are on the live Xeno estate.

Target:
- xeno-private-api-001

Goal:
Retire the rollback-only local portal auth/user tables now that canonical platform auth is the production authority.

Guardrails:
- Snapshot/backup first.
- Do not break login, register, password change, session validation, account overview, workspaces, projects, keys, billing, or security flows.
- Do not touch platform canonical auth/billing tables on xeno-platform-001 except for validation.

Required steps:
1. Confirm preconditions
- The portal app routes no longer depend on local Prisma/bcrypt/jsonwebtoken production auth logic.
- The portal login/register/session/profile flow is bridged into canonical platform auth.
- A stability window has elapsed with no need to roll back to local tables.

2. Snapshot
- Dump/archive the local portal auth/user tables before any destructive change.
- Record the backup path in the final report.

3. Apply retirement
- Use the staged checklist and SQL template already documented in:
  - docs/XENOSTUDIO_PORTAL_AUTH_RETIREMENT.md
  - docs/sql/retire_legacy_portal_auth_template.sql
- Remove only the rollback-only local portal auth/user tables and related no-longer-used schema objects.
- Do not remove unrelated local portal application tables.

4. Rebuild/restart if needed
- Restart only the services required.

5. Verify end to end
- login
- register
- /api/auth/me
- profile update
- password change
- dashboard account/usage/keys/billing/security routes
- API key lifecycle from the portal

Final report must include:
- backup path
- exact objects retired
- exact verification commands/results
- rollback note
```

## Prompt 3: Live Stripe Wiring Validation Later

Do not run this until real Stripe secrets and price IDs are available.

```text
You are working on the live Xeno estate.

Targets:
- xeno-platform-001
- xeno-private-api-001

Goal:
Deploy and validate live Stripe-backed billing operations without regressing the canonical ledger/account model.

Use the existing commercial billing code and validate:
- provider env wiring
- webhook ingress
- top-up checkout
- subscription checkout/cancel/resume
- payment methods
- invoices
- billing portal session
- subscription allowance behavior

Preserve:
- canonical ledger authority on xeno-platform-001
- workspace-scoped billing ownership
- no local portal/API billing authority

At the end provide:
- exact envs set
- exact webhook events verified
- exact live flows tested
- any remaining billing-specific gaps
```

