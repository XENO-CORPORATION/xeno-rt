import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (path) => readFileSync(join(root, path), 'utf8');

test('hosted embedding image pins builders, runtime, model, and ORT identities', () => {
  const dockerfile = read('deploy/embedding/Dockerfile');
  const ort = JSON.parse(read('reference/runtime/onnxruntime-1.20.0-linux-x64.json'));
  const embedding = JSON.parse(read('reference/embedding/nomic-embed-text-v1.5-a15734e.json'));

  assert.match(dockerfile, /FROM rust:[^\s]+@sha256:[0-9a-f]{64} AS builder/);
  assert.match(dockerfile, /ENV RUSTUP_TOOLCHAIN=1\.85\.1/);
  assert.match(dockerfile, /test "\$\(rustc --version\)" = 'rustc 1\.85\.1 \(4eb161250 2025-03-15\)'/);
  assert.ok(dockerfile.indexOf('rustc --version') < dockerfile.indexOf('cargo build --release --locked'));
  assert.match(dockerfile, /FROM debian:[^\s]+@sha256:[0-9a-f]{64}/);
  assert.match(dockerfile, new RegExp(ort.archive.sha256));
  assert.match(dockerfile, new RegExp(String(ort.archive.size_bytes)));
  assert.equal(embedding.artifacts[0].sha256, 'b4342336debaea79de872370664b0aaeb67dea4605513d00ee236ea871a81f27');
  assert.match(dockerfile, /nomic-embed-text-v1\.5-a15734e\.json/);
  assert.match(dockerfile, /2dc870de10066111e27bc6c25375d27f455e1de8a277b9bc5623f473ac9d2121/);
  assert.match(dockerfile, /USER 10001:10001/);
});

test('hosted embedding service has no public port and drops ambient privilege', () => {
  const compose = read('deploy/embedding/docker-compose.yml');
  assert.doesNotMatch(compose, /^\s+ports:/m);
  assert.match(compose, /name: xeno-platform_xenostudio-network/);
  assert.match(compose, /read_only: true/);
  assert.match(compose, /cap_drop:\s*\r?\n\s+- ALL/);
  assert.match(compose, /no-new-privileges:true/);
  assert.match(compose, /\/etc\/xeno\/xrt-embedding\.env/);
});

test('remote deploy proves an isolated candidate before changing latest', () => {
  const remote = read('scripts/remote-deploy-embedding.sh');
  const candidateGate = remote.indexOf('target-host candidate readiness passed');
  const productionTag = remote.indexOf('docker tag "$IMAGE:$SHA" "$IMAGE:latest"');
  assert.ok(candidateGate > 0);
  assert.ok(productionTag > candidateGate);
  assert.match(remote, /SECRET_MODE=.*stat -c/);
  assert.match(remote, /mode 600 or 640/);
  assert.match(remote, /attempting image rollback/);
});

test('operator deploy defaults to dry-run and requires an explicit execute flag', () => {
  const deploy = read('scripts/deploy-embedding-runtime.mjs');
  assert.match(deploy, /execute: false/);
  assert.match(deploy, /if \(!opts\.execute\) process\.exit\(0\)/);
  assert.match(deploy, /git', \['archive'/);
});
