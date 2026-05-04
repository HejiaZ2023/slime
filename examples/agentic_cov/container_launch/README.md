# container_launch

Local-only launcher for the `slime-llm4cov` image. Holds the per-host SSH
keys, git identity, and `docker run` flags needed to start a training
container that can reach the EDA worker host via SSH.

Only `docker-entrypoint.sh`, `.gitignore`, and this README are tracked.
Everything else (`Makefile`, `.gitconfig`, `.ssh/`) is git-ignored — it
contains private keys and per-server paths that must not leave the host.

## Layout

```
container_launch/
├── docker-entrypoint.sh    # tracked — baked into docker/Dockerfile.llm4cov
├── .gitignore              # tracked — keeps the rest local
├── README.md               # tracked
├── Makefile                # local — `make build|run|attach|stop|rm|rerun`
├── .gitconfig              # local — bind-mounted to /root/.gitconfig
└── .ssh/                   # local — bind-mounted to /run/host-ssh,
    ├── config              #         copied to /root/.ssh by entrypoint
    ├── id_ed25519
    ├── id_ed25519.pub
    └── ...
```

## Deploy to a new server

1. `scp -r examples/agentic_cov/container_launch <server>:/path/`.
2. On the server, `docker pull <your-registry>/slime-llm4cov:<tag>`.
3. Edit `Makefile`: set `IMAGE`, `HF_DIR`, `USER_DIR`, `DATA_DIR`, and the
   `WANDB_API_KEY` / `HF_TOKEN` env vars to match the host.
4. `make run` — starts the container detached, with `.ssh/` populated and
   `git` configured. `make attach` for an interactive shell.

## How the SSH bootstrap works

`Makefile` bind-mounts `./.ssh` read-only at `/run/host-ssh`. The image's
`ENTRYPOINT` (this directory's `docker-entrypoint.sh`) copies it into
`/root/.ssh` on container start and fixes the file modes OpenSSH requires
(`700` dir, `600` private keys, `644` `*.pub` / `known_hosts`). The copy —
rather than a direct mount — lets the container write its own
`ControlPath` sockets without polluting the host directory.

The entrypoint hard-fails if `/run/host-ssh` is not mounted or if the
copied tree lacks a `config` file. The slime+llm4cov image always needs
SSH (the rollout's `llm4cov.eda_client.remote_*` calls go through an SSH
alias), so silently starting without it would just defer the failure to
the first rollout step. To run the image without SSH (e.g. for a one-off
`python -c` smoke test) override the entrypoint:
`docker run --entrypoint='' ... slime-llm4cov:<tag> bash`.
