import hashlib
import json
import logging
import os
import time
from pathlib import Path

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_resume_ready_marker(args, rollout_id: int) -> None:
    """Mark a native checkpoint resumable only after the rollout state is durable."""
    if not args.save:
        return
    save_dir = Path(args.save)
    native_dir = save_dir / f"iter_{rollout_id:07d}"
    native_marker = native_dir / ".complete.json"
    if not native_marker.is_file():
        raise RuntimeError(f"native checkpoint completion marker is missing: {native_marker}")

    rollout_state = None
    if args.rollout_global_dataset:
        rollout_state = save_dir / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
        if not rollout_state.is_file():
            raise RuntimeError(f"rollout dataset state is missing: {rollout_state}")

    ready = {
        "version": 1,
        "kind": "slime_resume_bundle",
        "iteration": int(rollout_id),
        "created_at_unix": time.time(),
        "native_checkpoint": {
            "path": native_dir.name,
            "complete_marker_sha256": _file_sha256(native_marker),
        },
        "rollout_state": (
            {
                "path": rollout_state.relative_to(save_dir).as_posix(),
                "bytes": rollout_state.stat().st_size,
                "sha256": _file_sha256(rollout_state),
            }
            if rollout_state is not None
            else None
        ),
    }
    marker = save_dir / f"resume_ready_step_{rollout_id}.json"
    staging = marker.with_suffix(".json.tmp")
    staging.write_text(json.dumps(ready, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, marker)
    logger.info("Published complete resume bundle marker: %s", marker)


def train(args):
    configure_logger()
    t_start = time.time()

    logger.info("=== slime train() start ===")
    logger.info("  hf_checkpoint : %s", args.hf_checkpoint)
    logger.info("  load (resume) : %s", args.load)
    logger.info("  save          : %s", args.save)
    logger.info("  ref_load      : %s", args.ref_load)
    logger.info("  save_hf       : %s", args.save_hf)
    logger.info("  num_rollout=%d  save_interval=%d  eval_interval=%s",
                args.num_rollout, args.save_interval, args.eval_interval)
    logger.info("  lr=%s  rollout_batch_size=%d  n_samples_per_prompt=%d  global_batch_size=%d",
                args.lr, args.rollout_batch_size,
                args.n_samples_per_prompt, args.global_batch_size)

    # allocate the GPUs
    t_pg = time.time()
    pgs = create_placement_groups(args)
    logger.info("placement groups ready (%.1fs)", time.time() - t_pg)

    init_tracking(args)
    if args.use_wandb and getattr(args, "wandb_run_id", None):
        logger.info("W&B run_id=%s  project=%s  group=%s",
                    args.wandb_run_id, args.wandb_project,
                    getattr(args, "wandb_group", None))

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    logger.info("starting rollout manager (sglang engine init + dataset load)...")
    t_rollout = time.time()
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    logger.info("rollout manager ready (%.1fs)", time.time() - t_rollout)

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    logger.info("sglang metrics endpoint: %s", router_addr)
    update_tracking_open_metrics(args, router_addr)

    # create the actor and critic models
    logger.info("initializing Megatron actor model  load=%s  hf_checkpoint=%s",
                args.load, args.hf_checkpoint)
    t_megatron = time.time()
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    logger.info("Megatron actor ready  start_rollout_id=%d  (%.1fs)",
                args.start_rollout_id, time.time() - t_megatron)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    if not args.critic_train_only:
        logger.info("pushing initial model weights to sglang engines...")
        t_weights = time.time()
        actor_model.update_weights()
        logger.info("initial weights pushed (%.1fs)", time.time() - t_weights)

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    logger.info("=== startup complete in %.1fs, entering train loop at rollout_id=%d ===",
                time.time() - t_start, args.start_rollout_id)

    def offload_train(rollout_id):
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            if args.critic_train_only:
                critic_model.clear_memory()
            else:
                actor_model.clear_memory()

    def save(rollout_id):
        actor_saved = False
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps and not args.critic_train_only):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
            actor_saved = True
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))
        if actor_saved:
            publish_resume_ready_marker(args, rollout_id)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        save_num_rollout = None if getattr(args, "no_save_final_rollout", False) else args.num_rollout
        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, save_num_rollout):
            save(rollout_id)

        offload_train(rollout_id)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        if not args.critic_train_only:
            actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
