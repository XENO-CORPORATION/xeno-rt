# Signed, explicitly pinned artifact delivery

This contract authenticates an explicitly selected bundle manifest. It is not
an automatic update feed, a TUF implementation, Windows Authenticode or model
admission. Existing digest-only manual imports remain available for compatibility.
Public distribution must supply a separately authorized publisher identity.

## Message and trust

The signature is Ed25519 over the bytes
`xeno-rt.artifact-manifest.v1\0` followed by the 64 lowercase ASCII hex characters
of `ArtifactManifest::digest()`. The Rust manifest parser owns canonicalization;
the signer obtains the digest from `xrt-cli bundle digest --manifest <file>`.
Every file digest, size, entrypoint, license record and dependency is thereby
bound to the signature. File bytes are subsequently verified by the existing
atomic, resumable bundle installer.

Signature documents have exactly `schema_version: 1`, `algorithm: "ed25519"`,
`key_id`, `manifest_digest` and `signature`. The signature is 128 lowercase hex
characters. Trusted public-key documents have exactly `schema_version: 1`,
`algorithm: "ed25519"`, `key_id` and `public_key` (64 lowercase hex characters).
`key_id` is SHA-256 of the raw 32-byte public key. Unknown fields, malformed
lengths, unknown identities, invalid signatures and documents over 4 KiB fail.

The trusted key is supplied from an independently trusted installation or
operator configuration. Downloading a public key alongside its signature does
not establish trust. Verification also requires the caller's expected manifest
digest; an older signed artifact cannot replace a different explicitly pinned
digest. There is no timestamp, expiry, remote key rotation, revocation or
automatic rollback policy in this narrow contract.

## Prepare and inspect

Run publication tooling from the `xeno-platform` release checkout after following
`release-guide/`. This local signer neither uploads nor generates a publisher key:

```powershell
node <xeno-rt>/scripts/sign-artifact-manifest.mjs --cli <xrt-cli> --manifest <bundle.json> --out <signature.json> --expected-key-id <independently-pinned-publisher-key-id>
```

Load `XRT_MANIFEST_SIGNING_KEY` into that process from the protected operator
secret store. Its encoding is base64 PKCS#8 Ed25519. Never place the private key
in arguments, a repository file, command history, logs or a temporary file. The
signer does not pass private material to the CLI digest subprocess, checks the
expected public identity before signing, and refuses to overwrite its output.
Do not repurpose the existing workstation renderer-tool key to authorize native
executables; the release skill requires a protected-store operator decision.

For an authorized offline payload, the import form is:

```powershell
xrt-cli bundle --cache <cache> import --manifest <bundle.json> --digest <expected-digest> --signature <signature.json> --trusted-key <trusted-key.json> --directory <payload> --max-bytes <bounded-size> --confirm
```

The same paired signature/trusted-key options apply to remote bundle install.
Signature verification happens before cache creation or artifact transfer. A
successful signature never replaces license, model-quality, capability or
platform qualification. Large weights/CUDA payloads retain the runtime's
streaming/resumable artifact path instead of Hub's in-memory package pipeline.

## Qualification without publication

`scripts/qualify-artifact-signature.mjs` signs only a local test candidate using
an ephemeral in-memory TEST identity. It installs and rehashes actual CLI/server
executables, rejects a modified signature before cache creation, rejects modified
payload bytes and confirms that the previous verified candidate remains intact.
It retains only public test material. Its summary explicitly reports
`production_publisher_authorized: false`, `model_weights_bundled: false` and
`published: false`. Subsequent service tests must run the installed executable
with the SHA-256 from the inspected manifest inventory.

Drama 3's redistribution status is unknown. Prepare delivery without publishing
its weights, adding a public catalog entry or asserting a commercial license.
