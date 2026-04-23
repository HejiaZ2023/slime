"""Entry point for multi-round agentic llm4cov GRPO training.

Usage (in place of ``python train.py`` in slime's docs):

    python -m examples.agentic_cov.train \
        --rollout-function-path examples.agentic_cov.rollout.generate_rollout \
        --eval-function-path    examples.agentic_cov.rollout.eval_rollout \
        --data-source-path      examples.agentic_cov.data_source.LlmCovDataSource \
        --num-agentic-rounds 2 \
        --eval-num-agentic-rounds 1 \
        --llm4cov-dataset-name       zhuyaoyu/CodeV-R1-dataset \
        --llm4cov-dataset-split      train \
        --llm4cov-eval-dataset-name  hez2024/cvdp_ecov_eval \
        --llm4cov-eval-dataset-split eval \
        --eda-server            paladin_centos \
        --eda-repo-dir          /workspace/llm4cov_eda \
        --advantage-estimator grpo \
        --calculate-per-token-loss \
        --use-kl-loss=false \
        ...  # + the usual slime/megatron/sglang flags
"""

from __future__ import annotations

from slime.utils.arguments import parse_args

from train import train  # slime/train.py


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
        help="Number of sequential rollout rounds per prompt during evaluation.",
    )
    parser.add_argument(
        "--llm4cov-dataset-name",
        type=str,
        default="zhuyaoyu/CodeV-R1-dataset",
        help="HF dataset name for training prompts (llm4cov.datasets.load.load_dataset_by_name).",
    )
    parser.add_argument(
        "--llm4cov-dataset-split",
        type=str,
        default="train",
        help="Split name for the training dataset.",
    )
    parser.add_argument(
        "--llm4cov-eval-dataset-name",
        type=str,
        default="hez2024/cvdp_ecov_eval",
        help="HF dataset name for evaluation prompts. Loaded independently from training.",
    )
    parser.add_argument(
        "--llm4cov-eval-dataset-split",
        type=str,
        default="eval",
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
        "--eda-job-timeout",
        type=int,
        default=600,
        help="Per-sample coverage-job timeout in seconds.",
    )
    return parser


if __name__ == "__main__":
    args = parse_args(add_custom_arguments=add_agentic_args)
    train(args)
