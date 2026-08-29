#!/usr/bin/env node
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const argv = process.argv.slice(2);
const opts = {
  execute: false,
  buildOnly: false,
  rollback: false,
  host: 'xeno-platform-001',
  root: '/mnt/projects/xeno-rt-embedding',
};
for (let index = 0; index < argv.length; index += 1) {
  const arg = argv[index];
  if (arg === '--execute') opts.execute = true;
  else if (arg === '--build-only') opts.buildOnly = true;
  else if (arg === '--rollback') opts.rollback = true;
  else if (arg === '--host') opts.host = argv[++index];
  else if (arg === '--root') opts.root = argv[++index];
  else throw new Error(`unknown argument: ${arg}`);
}

const repo = new URL('..', import.meta.url).pathname.replace(/^\/(?:([A-Za-z]:))/, '$1');
const git = (args) => execFileSync('git', args, { cwd: repo, encoding: 'utf8' }).trim();
const run = (command, args) => {
  const result = spawnSync(command, args, { cwd: repo, stdio: 'inherit', encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`${command} exited ${result.status}`);
};

const sha = git(['rev-parse', 'HEAD']);
const shortSha = sha.slice(0, 12);
const branch = git(['rev-parse', '--abbrev-ref', 'HEAD']);
const dirty = git(['status', '--porcelain', '--', 'Cargo.toml', 'Cargo.lock', 'crates', 'deploy', 'reference', 'scripts']);
if (dirty) throw new Error(`deployment inputs are dirty:\n${dirty}`);

console.log(`xeno-rt embedding deploy branch=${branch} sha=${sha}`);
console.log(`host=${opts.host} root=${opts.root} mode=${opts.rollback ? 'rollback' : opts.buildOnly ? 'build-only' : 'swap'}`);
console.log(opts.execute ? 'EXECUTE' : 'DRY-RUN');

if (opts.rollback) {
  const remote = `docker image inspect xeno-rt-embedding:rollback >/dev/null && docker tag xeno-rt-embedding:rollback xeno-rt-embedding:latest && cd ${opts.root}/current && docker compose --project-name xeno-rt-embedding --file deploy/embedding/docker-compose.yml up -d --force-recreate --no-deps xrt-embedding`;
  console.log(`ssh ${opts.host} sudo bash -lc <rollback command>`);
  if (!opts.execute) process.exit(0);
  run('ssh', [opts.host, `sudo bash -lc ${shellQuote(remote)}`]);
  process.exit(0);
}

console.log(`1. archive exact committed source ${shortSha}`);
console.log(`2. copy archive plus remote deploy helper to ${opts.host}`);
console.log('3. build a digest-pinned Linux image and install the checksum-locked model bundle');
console.log('4. run an isolated target-host candidate and require authenticated readiness');
console.log(opts.buildOnly ? '5. retain the candidate image without swapping production' : '5. swap the protected-network service with automatic image rollback');
if (!opts.execute) process.exit(0);

const stage = mkdtempSync(join(tmpdir(), 'xrt-embedding-deploy-'));
const archive = join(stage, `xrt-embedding-${shortSha}.tar`);
try {
  run('git', ['archive', '--format=tar', '-o', archive, 'HEAD']);
  run('ssh', [opts.host, 'mkdir -p /tmp/xrt-embedding-deploy']);
  run('scp', ['-q', archive, `${opts.host}:/tmp/xrt-embedding-deploy/source.tar`]);
  run('scp', ['-q', join(repo, 'scripts', 'remote-deploy-embedding.sh'), `${opts.host}:/tmp/xrt-embedding-deploy/remote-deploy-embedding.sh`]);
  const mode = opts.buildOnly ? 'build-only' : 'swap';
  run('ssh', [opts.host, `sudo bash /tmp/xrt-embedding-deploy/remote-deploy-embedding.sh --sha ${sha} --tar /tmp/xrt-embedding-deploy/source.tar --mode ${mode} --root ${opts.root}`]);
} finally {
  rmSync(stage, { recursive: true, force: true });
}

function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

