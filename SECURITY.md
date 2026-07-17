# Security Policy

XENO Corporation accepts responsible security reports for xeno-rt.

## Supported Versions

| Version | Security support |
|---|---|
| 0.2.x (current beta line) | Yes |
| 0.1.x and older | No |

Security fixes target the latest supported release line. Backports are made
only when maintainers explicitly announce them.

## Report a Vulnerability

Do not open a public issue for a suspected vulnerability.

Preferred channel:

1. Open the repository's [private vulnerability reporting page](https://github.com/XENO-CORPORATION/xeno-rt/security/advisories/new).
2. Include impact, affected versions/commits, reproduction, and a suggested fix
   if available.

Alternative channel: email `security@bnkrsys.com`. Request a PGP key before
sending material that requires encrypted email.

Do not include production credentials, private model weights, personal data,
or unrelated confidential information.

## Response Targets

| Stage | Target |
|---|---|
| Initial acknowledgment | 2 business days |
| Preliminary assessment | 7 calendar days |
| Critical/high remediation plan | 30 calendar days |
| Coordinated disclosure | After a fix, or by an agreed deadline |

Targets are not warranties. Complex dependency, driver, or coordinated
ecosystem issues may require a different timeline, which will be communicated
to the reporter.

## In Scope

- Memory safety, bounds, integer overflow, or path-validation defects in GGUF
  and SafeTensors loading.
- Host or device memory corruption caused by validated runtime inputs.
- Arbitrary code execution or unintended file access through model/API input.
- Denial of service caused by a bounded request that bypasses documented limits.
- SSRF, local-file disclosure, or unsafe URL handling in server-side fetches.
- External-backend credential disclosure or loopback/remote policy bypass.
- Cross-request model, prompt, KV, cache, or generated-data disclosure.
- Supply-chain compromise in source, workflows, dependencies, or release
  artifacts.
- A bypass of a documented security boundary.

## Important Deployment Boundary

`xrt-server` does not currently implement inbound authentication, authorization,
or TLS. It binds to `127.0.0.1` by default. Exposing it directly to an untrusted
network is unsupported; use an authenticating TLS reverse proxy and network
controls.

The absence of inbound authentication is documented behavior, not an auth
bypass. A vulnerability that bypasses an operator's documented boundary,
leaks outbound credentials, or permits unintended server-side access remains in
scope.

## Generally Out of Scope

- Model output quality, bias, hallucinations, or prompt-injection behavior that
  does not cross a runtime security boundary.
- Numerical differences within documented CPU/GPU tolerance.
- Expected resource use from intentionally loading a model that exceeds the
  machine's capacity.
- Performance-only reports without a security impact.
- Unsupported model architectures or deployment platforms.
- Local privilege escalation claims that require running the server as an
  already-privileged account.

## Threat Model

xeno-rt processes model files, tokenizer/template assets, HTTP requests,
optional server-side image URLs, CUDA launch parameters, and external-backend
credentials. These inputs can be untrusted. Parsers and adapters are expected
to validate offsets, lengths, tensor geometry, metadata, paths, URLs, and
resource budgets before execution.

Operators are responsible for process identity, network isolation, TLS,
inbound authentication, filesystem permissions, model provenance, and machine
capacity.

## Disclosure and Credit

We follow coordinated disclosure. Reporters are credited when requested and
when legally possible. Do not publicly disclose unresolved details before an
agreed date.
