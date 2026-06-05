#!/usr/bin/env python3
"""
recompute_pass_targets.py — 按指定 valid 集的 target 重算"训练时 eval"的 is_pass_targets pass rate。

背景：训练时 eval（rollout.py）记录在 log 里的 is_pass_targets 在部分 run（尤其 step49→499）是
**有 bug 的**——即使 coverage 明显低于 target，也会记成 True。本脚本从 log 取每个样本的真实
coverage，按 valid 集的 target_percentage 重新、正确地判定 is_pass_targets。

输入训练时 eval 的 log（run-path 下的 eval_step_49.log / eval_step_499.log 等），其中每个样本一轮
记录为一行：
    rollout.py:339 - EVAL_EDA dataset_id=5276 round=1/4 idx=0 status=success reward=1.68 \
        coverage=0.6817 is_pass_xrun=True is_pass_targets=True filename=...
该 eval 为 markov-react：每 task 跑多轮、best(最高覆盖率)那轮驱动后续，最终该 task 的覆盖率取其
所有轮中的最高 coverage（即 Best@1，与 log 末尾 summary 的 overall_coverage Best@1 一致）。

判定复刻 llm4cov/datasets/eval.py：is_pass_targets = 每个 target 满足 coverage*100 >= target_percentage
（target.metric 为 "Overall Average" 时直接用该 task 的 overall_coverage）。覆盖率取自 log 真实值、
与 target 无关——换 valid 集只改判定阈值。log 中缺失的 task（xrun 全失败/未生成）记为不通过。

用法:
  python recompute_pass_targets.py <eval_step_*.log> \
      [--valid1 val_codev_rl_test_with_targets.parquet] \
      [--valid2 val_codev_rl_test_r1cov.parquet]

路径自适应容器挂载：给定路径不存在时自动尝试 /mnt/raid0_ssd <-> /data 前缀互换。
"""
import argparse
import re
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("需要 pandas：在 llm4cov 容器内跑，例如\n"
             "  docker exec -i llm4cov_verl python3 recompute_pass_targets.py <log>")

DEFAULT_V1 = "/mnt/raid0_ssd/sheng/valid_dataset/val_codev_rl_test_with_targets.parquet"
DEFAULT_V2 = "/mnt/raid0_ssd/sheng/valid_dataset/val_codev_rl_test_r1cov.parquet"

# host /mnt/raid0_ssd  <->  container /data
_SWAPS = [("/mnt/raid0_ssd", "/data"), ("/data", "/mnt/raid0_ssd")]

EVAL_RE = re.compile(
    r"EVAL_EDA (?:dataset_id|ctx)=(\d+) round=\d+/\d+ idx=\d+ status=(\w+) "
    r"reward=[\d.]+ coverage=([\d.]+) is_pass_xrun=\w+ is_pass_targets=(\w+)"
)


def resolve(p):
    p = Path(p)
    if p.exists():
        return p
    s = str(p)
    for frm, to in _SWAPS:
        if s.startswith(frm):
            cand = Path(to + s[len(frm):])
            if cand.exists():
                return cand
    return p


def load_targets(parquet_path):
    """{problem_id(str): [(metric, target_percentage_float), ...]}"""
    df = pd.read_parquet(resolve(parquet_path))
    out = {}
    for _, row in df.iterrows():
        out[str(row["problem_id"])] = [
            (t["metric"], float(t["target_percentage"])) for t in row["targets"]
        ]
    return out


def parse_eval_log(path):
    """返回 {dataset_id(str): (best_coverage, log_is_pass_targets_str)}，取最高 coverage 那一轮。"""
    p = resolve(path)
    if not p.exists():
        sys.exit(f"找不到 eval log: {path}")
    best = {}
    n_rows = 0
    for line in p.read_text(errors="ignore").splitlines():
        m = EVAL_RE.search(line)
        if not m:
            continue
        did, status, cov, log_ptg = m.group(1), m.group(2), float(m.group(3)), m.group(4)
        n_rows += 1
        if status != "success":
            continue
        if did not in best or cov > best[did][0]:
            best[did] = (cov, log_ptg)
    if n_rows == 0:
        sys.exit(f"{p} 中没有 EVAL_EDA 行 —— 不是训练时 eval 的 log？")
    return best, n_rows


def recompute(best, targets):
    """按 targets 重算。返回 (recomputed_Pass@1, mean_overall_coverage, n_task, n_missing)。"""
    passes, covs, missing = [], [], 0
    for pid, tgs in targets.items():
        rec = best.get(pid)
        if rec is None:
            missing += 1
            passes.append(False)
            covs.append(0.0)
            continue
        cov = rec[0]
        passes.append(all(cov * 100 >= tgt for (_m, tgt) in tgs))
        covs.append(cov)
    n = len(passes) or 1
    return sum(passes) / n, sum(covs) / n, len(passes), missing


def log_original_pass(best, task_ids):
    """log 里记录的原始 is_pass_targets pass rate（best 轮），缺失记不过。供对比（可能有误）。"""
    vals = [(best[t][1] == "True") if t in best else False for t in task_ids]
    return (sum(vals) / len(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("eval_log", help="训练时 eval 的 log（含 EVAL_EDA 行），如 eval_step_49.log")
    ap.add_argument("--valid1", default=DEFAULT_V1, help=f"第一个 valid 集 (默认 {DEFAULT_V1})")
    ap.add_argument("--valid2", default=DEFAULT_V2, help=f"第二个 valid 集 (默认 {DEFAULT_V2})")
    args = ap.parse_args()

    best, n_rows = parse_eval_log(args.eval_log)
    print(f"eval log     : {resolve(args.eval_log)}")
    print(f"EVAL_EDA 行  : {n_rows}   涉及 task(dataset_id): {len(best)}")

    tg1 = load_targets(args.valid1)
    orig = log_original_pass(best, list(tg1.keys()))
    print(f"log 原始 is_pass_targets : Pass@1 = {orig * 100:.1f} %  (训练时记录值，部分 run/step49-499 有误，仅供对比)")

    for tag, vp in [("valid1", args.valid1), ("valid2", args.valid2)]:
        tg = load_targets(vp)
        p1, mcov, n, missing = recompute(best, tg)
        warn = f"   ⚠ {missing} 个 task 在 log 中无成功覆盖率（记为不过）" if missing else ""
        print(f"\n[{tag}] {resolve(vp)}")
        print(f"  valid tasks      : {n}{warn}")
        print(f"  overall_coverage : Best@1 = {mcov:.4f}  ({mcov * 100:.1f} %)")
        print(f"  is_pass_targets  : Pass@1 = {p1 * 100:.1f} %   (按该 valid 集 target 重算)")


if __name__ == "__main__":
    main()
