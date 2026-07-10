"""Entry point for multi-round agentic llm4cov GRPO training.

Usage (in place of ``python train.py`` in slime's docs):

    python -m examples.agentic_cov.train \
        --rollout-function-path examples.agentic_cov.rollout.generate_rollout \
        --eval-function-path    examples.agentic_cov.rollout.eval_rollout \
        --data-source-path      examples.agentic_cov.data_source.LlmCovDataSource \
        --num-agentic-rounds 2 \
        --eval-num-agentic-rounds 1 \
        --llm4cov-dataset-name       Senlimulin/CodeV_R1_5918_dataset \
        --llm4cov-dataset-split      train \
        --llm4cov-eval-dataset-name  Senlimulin/2026UCSDIntern_SlimeRL_training_dataset \
        --llm4cov-eval-dataset-split validation \
        --eda-server            paladin_centos \
        --eda-repo-dir          /workspace/llm4cov_eda \
        --advantage-estimator grpo \
        --calculate-per-token-loss \
        ...  # + the usual slime/megatron/sglang flags
        # (omit --use-kl-loss to disable KL; it is store_true, default False)
"""

from __future__ import annotations

import logging

from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger

from train import train  # slime/train.py

logger = logging.getLogger(__name__)


def add_agentic_args(parser):
    parser.add_argument(
        "--num-agentic-rounds",
        type=int,
        default=2,
        help="Number of sequential rollout rounds per prompt during training (K).",
    )
    parser.add_argument(
        "--eval-num-agentic-rounds",
        type=int,
        default=1,
        help="React follow-up rounds during eval (same as --react-rounds in batch_query_eval); total rounds = 1 + value",
    )
    parser.add_argument(
        "--llm4cov-dataset-name",
        type=str,
        default="Senlimulin/CodeV_R1_5918_dataset",
        help="HF dataset name for training prompts (llm4cov.datasets.load.load_dataset_by_name).",
    )
    parser.add_argument(
        "--llm4cov-dataset-split",
        type=str,
        default="train",
        help="Split name for the training dataset.",
    )
    parser.add_argument(
        "--llm4cov-dataset-step-offset",
        type=int,
        default=0,
        help=(
            "Skip this many rollout steps in the training dataset order before "
            "starting. With the same rollout seed, training step n then consumes "
            "the prompts that step n+offset would have consumed."
        ),
    )
    parser.add_argument(
        "--llm4cov-eval-dataset-name",
        type=str,
        default="Senlimulin/2026UCSDIntern_SlimeRL_training_dataset",
        help="HF dataset name for evaluation prompts. Loaded independently from training.",
    )
    parser.add_argument(
        "--llm4cov-eval-dataset-split",
        type=str,
        default="validation",
        help="Split name for the eval dataset.",
    )
    parser.add_argument(
        "--eda-server",
        type=str,
        required=True,
        help="SSH host for the remote EDA coverage worker (e.g. paladin_centos).",
    )
    parser.add_argument(
        "--eda-repo-dir",
        type=str,
        required=True,
        help="Remote path to the llm4cov_eda checkout on the EDA server.",
    )
    parser.add_argument(
        "--eda-stage-timeout",
        type=int,
        default=30,
        help=(
            "Per-stage coverage-job timeout in seconds, passed to "
            "run_remote_cov_job_pipeline's `timeout` (which it uses "
            "per-stage; the SSH wall-clock budget is timeout*3). Mirrors "
            "DEFAULT_EDA_SINGLE_STAGE_TIMEOUT_S=30 in llm4cov_oss's "
            "scripts/batch_query_eval.py."
        ),
    )
    parser.add_argument(
        "--eda-log-feedback-train",
        action="store_true",
        default=False,
        help=(
            "During training rollouts: fetch full EDA coverage detail report "
            "(block/expression/FSM uncovered entries) and embed verbatim in the "
            "next-round tool-feedback message. Slows down each training step."
        ),
    )
    parser.add_argument(
        "--eda-log-feedback-eval",
        action="store_true",
        default=False,
        help=(
            "During eval rollouts: fetch full EDA coverage detail report "
            "(block/expression/FSM uncovered entries) and embed verbatim in the "
            "next-round tool-feedback message. Mirrors batch_query_eval.py behaviour."
        ),
    )
    parser.add_argument(
        "--use-uncovered-log",
        action="store_true",
        default=False,
        help=(
            "When EDA log feedback is on (--eda-log-feedback-train/-eval), build the "
            "tool-feedback from the structured cov_info['uncovered'] (compact per-bin "
            "uncovered list) instead of the raw truncated IMC detail text. No effect "
            "without --eda-log-feedback-*. Off by default."
        ),
    )
    parser.add_argument(
        "--use-uncovered-reward",
        action="store_true",
        default=False,
        help=(
            "Add a group-level diversity bonus to the reward: reward = coverage_score "
            "+ div_lam * diversity, where diversity is the rarity-weighted novel-coverage "
            "share over the group's uncovered bin_ids (sum over group == 1). Requires "
            "--eda-log-feedback-train (uncovered data); no-op without it. Off by default."
        ),
    )
    parser.add_argument(
        "--div-lam",
        dest="div_lam",
        type=float,
        default=None,
        help=(
            "Diversity reward weight lambda (reward = coverage_score + div_lam*diversity). "
            "Required when --use-uncovered-reward is on."
        ),
    )
    parser.add_argument(
        "--use-opd-relay",
        action="store_true",
        default=False,
        help=(
            "Enable OPD relay training. Student rollouts are generated locally, "
            "student/teacher EDA is delegated to the paladin relay, and a gated "
            "student top-k KL is used instead of RL loss for that prompt-round "
            "when the best teacher beats the best student."
        ),
    )
    parser.add_argument(
        "--opd-teachers",
        type=str,
        default="stage0:1,stage1:1",
        help="Comma-separated teacher rollout spec, e.g. 'stage0:1,stage1:1'.",
    )
    parser.add_argument(
        "--opd-lambda",
        type=float,
        default=1.0,
        help="Weight for the OPD KL loss on gated student-token distributions.",
    )
    parser.add_argument(
        "--opd-topk",
        type=int,
        default=16,
        help="Student top-k tokens per generated position used for OPD KL; <=1 uses sampled-token fallback.",
    )
    parser.add_argument(
        "--opd-student-topk-mode",
        choices=("decode", "posthoc"),
        default="posthoc",
        help="Recover student OPD top-k ids with an exact posthoc prefill or during decode.",
    )
    parser.add_argument(
        "--opd-gate-eps",
        type=float,
        default=0.0,
        help="Require best_teacher_reward > best_student_reward + eps to use OPD.",
    )
    parser.add_argument(
        "--opd-timeout",
        type=float,
        default=1800.0,
        help="Wall-clock timeout in seconds for each OPD relay job.",
    )
    parser.add_argument(
        "--opd-poll",
        type=float,
        default=1.0,
        help="Polling interval in seconds while waiting for OPD relay results.",
    )
    parser.add_argument(
        "--opd-namespace",
        type=str,
        default="opd",
        help="Relay xfer namespace under incoming/ and results/.",
    )
    parser.add_argument(
        "--opd-server",
        type=str,
        default="local",
        help="OPD relay server hint. Use local on paladin, or a host for SFTP mode.",
    )
    parser.add_argument(
        "--opd-transport",
        type=str,
        default="",
        choices=["", "local", "sftp", "http"],
        help="OPD relay transport. Empty auto-selects local for paladin/local, else sftp.",
    )
    parser.add_argument(
        "--opd-http-url",
        type=str,
        default="",
        help="HTTP OPD worker base URL for direct Tailscale student-to-teacher transfer.",
    )
    parser.add_argument(
        "--opd-xfer-dir",
        type=str,
        default="/mnt/raid0_ssd/eda/xfer",
        help="Local xfer root when --opd-transport local is used.",
    )
    parser.add_argument("--opd-sftp-host", type=str, default="", help="SFTP relay host.")
    parser.add_argument("--opd-sftp-port", type=int, default=2222, help="SFTP relay port.")
    parser.add_argument("--opd-sftp-user", type=str, default="gpujobs", help="SFTP relay user.")
    parser.add_argument(
        "--opd-sftp-key",
        type=str,
        default="~/.ssh/brev_eda_sftp",
        help="Private key for SFTP relay transport.",
    )
    parser.add_argument(
        "--opd-score-student-rollouts",
        action="store_true",
        default=False,
        help="Ask teachers to score student responses; enabled automatically by the OPD relay path.",
    )
    parser.add_argument(
        "--rollout-log-dir",
        type=str,
        default=None,
        help=(
            "Directory for per-rollout split log files. "
            "Eval output goes to eval_step_<rollout_id>.log; "
            "train output goes to train_step_<start>-<end>.log (grouped by save-interval). "
            "When set, rollout.py logger stops propagating to the main log so "
            "TRAIN_*/EVAL_* lines appear only in the split files. "
            "Defaults to None (everything in main log)."
        ),
    )
    return parser


if __name__ == "__main__":
    args = parse_args(add_custom_arguments=add_agentic_args)
    configure_logger()
    # Default rollout log function if not explicitly overridden
    if not args.custom_rollout_log_function_path:
        args.custom_rollout_log_function_path = (
            "examples.agentic_cov.rollout.log_train_samples"
        )
    # Default split log dir: <save>_rollout_logs (always enabled)
    if not args.rollout_log_dir and args.save:
        args.rollout_log_dir = args.save.rstrip("/") + "_rollout_logs"

    if getattr(args, "use_opd_relay", False):
        if not getattr(args, "opd_teachers", ""):
            raise ValueError("--use-opd-relay requires --opd-teachers")

    logger.info("=== llm4cov slime RL training config ===")

    # ---------- checkpoint paths ----------
    logger.info("  hf_checkpoint : %s", args.hf_checkpoint)
    logger.info("  load (resume) : %s", args.load)
    logger.info("  save          : %s", args.save)
    logger.info("  ref_load      : %s", args.ref_load)
    logger.info("  save_hf       : %s", args.save_hf)

    # ---------- datasets ----------
    logger.info("  train dataset : %s  split=%s  step_offset=%d",
                args.llm4cov_dataset_name, args.llm4cov_dataset_split,
                getattr(args, "llm4cov_dataset_step_offset", 0))
    logger.info("  eval  dataset : %s  split=%s",
                args.llm4cov_eval_dataset_name, args.llm4cov_eval_dataset_split)

    # ---------- agentic / EDA ----------
    logger.info("  num_agentic_rounds=%d  eval_num_agentic_rounds=%d  eda_log_feedback_train=%s  eda_log_feedback_eval=%s",
                args.num_agentic_rounds, args.eval_num_agentic_rounds,
                getattr(args, "eda_log_feedback_train", False),
                getattr(args, "eda_log_feedback_eval", False))
    logger.info("  EDA server=%s  repo=%s  stage_timeout=%ds",
                args.eda_server, args.eda_repo_dir,
                getattr(args, "eda_stage_timeout", 30))

    logger.info("  OPD relay enabled=%s  teachers=%s  lambda=%s  gate_eps=%s",
                getattr(args, "use_opd_relay", False),
                getattr(args, "opd_teachers", ""),
                getattr(args, "opd_lambda", None),
                getattr(args, "opd_gate_eps", None))
    logger.info("  OPD relay transport=%s  server=%s  namespace=%s  timeout=%ss",
                getattr(args, "opd_transport", "") or "auto",
                getattr(args, "opd_server", ""),
                getattr(args, "opd_namespace", "opd"),
                getattr(args, "opd_timeout", None))

    # ---------- training hyperparams ----------
    logger.info("  num_rollout=%d  save_interval=%d  eval_interval=%s",
                args.num_rollout, args.save_interval, args.eval_interval)
    logger.info("  lr=%s  rollout_batch_size=%d  n_samples_per_prompt=%d  global_batch_size=%d",
                args.lr, args.rollout_batch_size,
                args.n_samples_per_prompt, args.global_batch_size)
    logger.info("  offload_rollout=%s  offload_train=%s",
                getattr(args, "offload_rollout", False),
                getattr(args, "offload_train", False))

    # ---------- tracking ----------
    logger.info("  wandb_project=%s  wandb_group=%s",
                args.wandb_project, getattr(args, "wandb_group", None))

    logger.info("==========================================")

    train(args)
