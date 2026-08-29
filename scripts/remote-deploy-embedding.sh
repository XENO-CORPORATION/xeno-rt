#!/usr/bin/env bash
set -euo pipefail

SHA=""
TAR=""
MODE="swap"
ROOT="/mnt/projects/xeno-rt-embedding"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --sha) SHA="$2"; shift 2 ;;
    --tar) TAR="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    *) echo "remote-deploy-embedding: unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$SHA" ] || { echo "remote-deploy-embedding: --sha is required" >&2; exit 2; }
[ -f "$TAR" ] || { echo "remote-deploy-embedding: tar not found: $TAR" >&2; exit 2; }
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "remote-deploy-embedding: --sha must be 40 lowercase hex characters" >&2; exit 2; }
[[ "$ROOT" == /mnt/projects/* ]] || { echo "remote-deploy-embedding: --root must stay below /mnt/projects" >&2; exit 2; }
case "$MODE" in swap|build-only) ;; *) echo "remote-deploy-embedding: invalid --mode" >&2; exit 2 ;; esac

SECRET_FILE="/etc/xeno/xrt-embedding.env"
[ -f "$SECRET_FILE" ] || { echo "remote-deploy-embedding: missing $SECRET_FILE" >&2; exit 1; }
grep -Eq '^XRT_EMBEDDING_API_KEY=.{32,}$' "$SECRET_FILE" || {
  echo "remote-deploy-embedding: XRT_EMBEDDING_API_KEY is missing or too short" >&2
  exit 1
}
SECRET_MODE="$(stat -c '%a' "$SECRET_FILE")"
case "$SECRET_MODE" in 600|640) ;; *) echo "remote-deploy-embedding: $SECRET_FILE must be mode 600 or 640" >&2; exit 1 ;; esac

NETWORK="xeno-platform_xenostudio-network"
docker network inspect "$NETWORK" >/dev/null

RELEASE_DIR="$ROOT/releases/$SHA"
SOURCE_DIR="$RELEASE_DIR/source"
rm -rf -- "$RELEASE_DIR"
mkdir -p "$SOURCE_DIR" "$ROOT/.deploy"
tar xf "$TAR" -C "$SOURCE_DIR"
find "$SOURCE_DIR" -type f \( -name '*.rs' -o -name '*.toml' -o -name '*.json' -o -name '*.md' -o -name '*.sh' -o -name 'Dockerfile' \) -exec sed -i 's/\r$//' {} +

LOG="$ROOT/.deploy/deploy.log"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

IMAGE="xeno-rt-embedding"
if docker image inspect "$IMAGE:latest" >/dev/null 2>&1; then
  docker tag "$IMAGE:latest" "$IMAGE:rollback"
  log "snapshotted current image as $IMAGE:rollback"
fi

log "building target-host image sha=$SHA"
docker build \
  --file "$SOURCE_DIR/deploy/embedding/Dockerfile" \
  --tag "$IMAGE:$SHA" \
  "$SOURCE_DIR"

CANDIDATE="xrt-embedding-candidate-${SHA:0:12}"
docker rm -f "$CANDIDATE" >/dev/null 2>&1 || true
docker run -d --rm \
  --name "$CANDIDATE" \
  --network "$NETWORK" \
  --env-file "$SECRET_FILE" \
  --read-only \
  --tmpfs /tmp:size=64m,mode=1777 \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  "$IMAGE:$SHA" >/dev/null

candidate_ready=0
for _ in $(seq 1 60); do
  if docker exec "$CANDIDATE" sh -ec 'curl -fsS -H "Authorization: Bearer $XRT_EMBEDDING_API_KEY" http://127.0.0.1:3099/v1/runtime/status >/dev/null'; then
    candidate_ready=1
    break
  fi
  sleep 2
done
if [ "$candidate_ready" -ne 1 ]; then
  docker logs "$CANDIDATE" --tail 100 >&2 || true
  docker rm -f "$CANDIDATE" >/dev/null 2>&1 || true
  echo "remote-deploy-embedding: target-host candidate did not become ready" >&2
  exit 1
fi
docker rm -f "$CANDIDATE" >/dev/null
log "target-host candidate readiness passed sha=$SHA"

if [ "$MODE" = "build-only" ]; then
  log "build-only complete; running service unchanged sha=$SHA"
  exit 0
fi

docker tag "$IMAGE:$SHA" "$IMAGE:latest"
cd "$SOURCE_DIR"
docker compose --project-name xeno-rt-embedding --file deploy/embedding/docker-compose.yml up -d --force-recreate --no-deps xrt-embedding

if docker inspect --format '{{.State.Health.Status}}' xeno-rt-embedding 2>/dev/null | grep -qx healthy; then
  ln -sfn "$SOURCE_DIR" "$ROOT/current"
  log "swap healthy sha=$SHA"
  exit 0
fi
for _ in $(seq 1 60); do
  if docker inspect --format '{{.State.Health.Status}}' xeno-rt-embedding 2>/dev/null | grep -qx healthy; then
    ln -sfn "$SOURCE_DIR" "$ROOT/current"
    log "swap healthy sha=$SHA"
    exit 0
  fi
  sleep 2
done

log "swap failed health gate; attempting image rollback"
if docker image inspect "$IMAGE:rollback" >/dev/null 2>&1; then
  docker tag "$IMAGE:rollback" "$IMAGE:latest"
  docker compose --project-name xeno-rt-embedding --file deploy/embedding/docker-compose.yml up -d --force-recreate --no-deps xrt-embedding
fi
exit 1
