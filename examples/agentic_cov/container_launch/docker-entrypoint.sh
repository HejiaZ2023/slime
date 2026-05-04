#!/usr/bin/env bash
# Container entrypoint baked into docker/Dockerfile.llm4cov.
#
# The slime+llm4cov image always needs SSH (the rollout's
# llm4cov.eda_client.remote_* reach the EDA host via an SSH alias), so
# this script REQUIRES the caller to bind-mount a host SSH directory at
# /run/host-ssh. It is copied into $HOME/.ssh on start and the file
# modes OpenSSH expects are applied (700 dir, 600 private keys, 644
# *.pub / known_hosts). The copy — rather than a direct mount — lets
# the container write its own ControlPath sockets without polluting the
# host directory.
#
# Optionally registers the host docker group inside the container when
# DOCKER_GID is provided, so a bind-mounted docker socket is usable when
# running as `-u 0:$DOCKER_GID`.
#
# This script must stay free of secrets — it is the only file in
# examples/agentic_cov/container_launch/ that is checked in.

set -euo pipefail

SSH_SRC=/run/host-ssh
SSH_DST="${HOME:-/root}/.ssh"

if [ ! -d "$SSH_SRC" ]; then
    echo "[entrypoint] ERROR: required SSH directory $SSH_SRC not mounted" >&2
    echo "[entrypoint] expected: docker run ... -v <host-ssh-dir>:$SSH_SRC:ro ..." >&2
    exit 1
fi

mkdir -p "$SSH_DST"
cp -r "$SSH_SRC"/. "$SSH_DST"/

chmod 700 "$SSH_DST"
find "$SSH_DST" -type d -exec chmod 700 {} \;
find "$SSH_DST" -type f -name "*.pub" -exec chmod 644 {} \;
find "$SSH_DST" -type f -name "known_hosts" -exec chmod 644 {} \;
find "$SSH_DST" -type f ! -name "*.pub" ! -name "known_hosts" -exec chmod 600 {} \;

if [ ! -f "$SSH_DST/config" ]; then
    echo "[entrypoint] ERROR: missing SSH config at $SSH_DST/config" >&2
    exit 1
fi

if [ -n "${DOCKER_GID:-}" ] \
        && ! getent group docker >/dev/null \
        && ! getent group "$DOCKER_GID" >/dev/null; then
    groupadd -g "$DOCKER_GID" docker
fi

exec "$@"
