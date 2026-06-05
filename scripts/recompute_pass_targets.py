#!/usr/bin/env python3
"""
recompute_pass_targets.py — 按指定 valid 集的 target 重算一次 eval 的 is_pass_targets pass rate。

输入一个 eval（batch_query_eval --debug 跑出的）：可以给 eval log（其中含一行
"Debug output written to .../eval_debug.json"），也可以直接给该 eval_debug.json。
脚本对两个 valid 集分别用其 targets 重新判定 is_pass_targets，输出 Pass@1 / Pass@5。

判定逻辑复刻 llm4cov/datasets/eval.py：
  is_pass_targets = has_coverage 且 对每个 target  actual_cov*100 >= target_percentage
  （target.metric == "Overall Average" 时 actual_cov 取 sample 的 overall_coverage，
    其他 metric 从 eval_result.misc 取）
覆盖率(overall_coverage)取自 debug json 中每个 sample 的真实结果，与 target 无关 —— 换 valid
集只改变判定阈值、不改变覆盖率。

用法:
  python recompute_pass_targets.py <eval_log 或 eval_debug.json> \
      [--valid1 val_codev_rl_test_with_targets.parquet] \
      [--valid2 val_codev_rl_test_r1cov.parquet]

路径自适应容器挂载：给定路径不存在时自动尝试 /mnt/raid0_ssd <-> /data 前缀互换，
因此 host 和 llm4cov_verl 容器内都能直接跑。
"""
import argparse
import json
import re
import sys
from collections import defaultdict
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


def resolve(p):
    """返回实际存在的路径，自动尝试容器挂载前缀互换。"""
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


def find_debug_json(path):
    p = resolve(path)
    if not p.exists():
        sys.exit(f"找不到输入文件: {path}")
    if p.suffix == ".json":
        return p
    text = p.read_text(errors="ignore")
    m = re.findall(r"Debug output written to (\S+)", text)
    if not m:
        sys.exit(f"未在 {p} 中找到 'Debug output written to ...' —— 该 eval 需用 --debug 跑")
    return resolve(m[-1])


def load_targets(parquet_path):
    """{problem_id(str): [(metric, target_percentage_float), ...]}"""
    df = pd.read_parquet(resolve(parquet_path))
    out = {}
    for _, row in df.iterrows():
        out[str(row["problem_id"])] = [
            (t["metric"], float(t["target_percentage"])) for t in row["targets"]
        ]
    return out


def sample_pass(eval_result, targets_for_pid):
    """复刻 eval.py 的判定。"""
    if not eval_result.get("has_coverage", False):
        return False
    cov = eval_result.get("overall_coverage", 0.0)
    misc = eval_result.get("misc", {}) or {}
    for metric, tgt in targets_for_pid:
        actual = cov if metric == "Overall Average" else misc.get(metric)
        if not isinstance(actual, float) or actual * 100 < tgt:
            return False
    return True


def compute(samples, targets):
    """返回 (Pass@1, Pass@k, n_task, missing_pids)。
    Pass@1 = 每 task 内样本平均通过率，再对 task 求平均；
    Pass@k = 每 task 任一样本通过即算过，再对 task 求平均（k = 每 task 样本数）。"""
    per_task = defaultdict(list)
    missing = set()
    for s in samples:
        pid = str(s["context_id"])
        if pid not in targets:
            missing.add(pid)
            continue
        per_task[pid].append(sample_pass(s["eval_result"], targets[pid]))
    n = len(per_task)
    if n == 0:
        return 0.0, 0.0, 0, missing
    p1 = sum(sum(v) / len(v) for v in per_task.values()) / n
    pk = sum(1.0 if any(v) else 0.0 for v in per_task.values()) / n
    return p1, pk, n, missing


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("eval_log", help="eval log（含 'Debug output written to ...'）或直接给 eval_debug.json")
    ap.add_argument("--valid1", default=DEFAULT_V1, help=f"第一个 valid 集 (默认 {DEFAULT_V1})")
    ap.add_argument("--valid2", default=DEFAULT_V2, help=f"第二个 valid 集 (默认 {DEFAULT_V2})")
    args = ap.parse_args()

    dj = find_debug_json(args.eval_log)
    data = json.load(open(dj))
    samples = data.get("samples")
    if not samples:
        sys.exit(f"{dj} 中没有 'samples' 字段（不是 batch_query_eval --debug 的输出？）")
    print(f"debug json : {dj}")
    print(f"samples    : {len(samples)}")

    # sanity: debug json 自带的原始 is_pass_targets（用第一个 valid 集即 with_targets 算的），
    # 用 --valid1 重算应与之吻合
    orig = (data.get("per_task") or {}).get("is_pass_targets")
    if orig:
        o1 = sum(v["@1"] for v in orig.values()) / len(orig)
        ok = sum(v.get("@5", v["@1"]) for v in orig.values()) / len(orig)
        print(f"debug json 原 is_pass_targets : Pass@1={o1 * 100:.1f}%  Pass@5={ok * 100:.1f}%  (应≈valid1)")

    for tag, vp in [("valid1", args.valid1), ("valid2", args.valid2)]:
        tg = load_targets(vp)
        p1, pk, n, missing = compute(samples, tg)
        print(f"\n[{tag}] {resolve(vp)}")
        warn = f"   ⚠ {len(missing)} 个 problem_id 不在该集（已跳过）" if missing else ""
        print(f"  matched tasks   : {n}{warn}")
        print(f"  is_pass_targets : Pass@1 = {p1 * 100:.1f} %   Pass@5 = {pk * 100:.1f} %")


if __name__ == "__main__":
    main()
