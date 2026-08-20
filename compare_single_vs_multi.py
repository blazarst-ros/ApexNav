#!/usr/bin/env python3
"""
对比 lite-apexnav 单智能体基座（组A）、lite-multiagent 禁用 agent（组B）、
lite-multiagent 双 agent（组C）的评测结果。

用法:
    python compare_single_vs_multi.py \
        --base /home/uuufo/ApexNav-lite-single/videos/test_base_hm3dv2_first20/continue.txt \
        --single /home/uuufo/ApexNav-lite/videos/test_single_hm3dv2_first20/continue.txt \
        [--multi /home/uuufo/ApexNav-lite/videos/test_multiagent_hm3dv2_first20/continue.txt]

输出:
    逐条 episode 对比表 + 汇总指标（Success/SPL/SoftSPL/DistToGoal）+ 失败类型分布
"""
import argparse
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path


def parse_continue(path: str) -> list:
    """解析 continue.txt，返回逐条记录列表。

    每条记录包含: scene_id, episode_id, result, label, 以及指标表中的数值。
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n(?=Scene ID: )", text)
    records = []
    for block in blocks:
        block = block.strip()
        if not block or not block.startswith("Scene ID:"):
            continue
        rec = {}
        m = re.search(r"Scene ID: (\S+)", block)
        rec["scene_id"] = m.group(1) if m else ""
        m = re.search(r"Episode ID: (\S+)", block)
        rec["episode_id"] = m.group(1) if m else ""
        m = re.search(r"success or not: (.+)", block)
        rec["result"] = m.group(1).strip() if m else "?"
        m = re.search(r"target to find is (.+)", block)
        rec["label"] = m.group(1).strip() if m else "?"
        # 指标表数值: |      Average Success      |  70.00% |
        for metric, key in [
            (r"Average Success\s*\|\s*([\d.]+)%", "success_pct"),
            (r"Average SPL\s*\|\s*([\d.]+)%", "spl_pct"),
            (r"Average Soft SPL\s*\|\s*([\d.]+)%", "soft_spl_pct"),
            (r"Average Distance to Goal\s*\|\s*([\d.]+)", "dtg"),
        ]:
            m = re.search(metric, block)
            rec[key] = float(m.group(1)) if m else None
        records.append(rec)
    return records


def summarize(records: list) -> dict:
    n = len(records)
    results = Counter(r["result"] for r in records)
    success = sum(1 for r in records if r["result"] == "success")
    # 用最后一条的累计平均（continue.txt 每条都是累计值）
    last = records[-1] if records else {}
    return {
        "n": n,
        "success": success,
        "success_rate": success / n * 100 if n else 0.0,
        "spl_pct": last.get("spl_pct"),
        "soft_spl_pct": last.get("soft_spl_pct"),
        "dtg": last.get("dtg"),
        "results": dict(results),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="组A 基座 continue.txt")
    ap.add_argument("--single", required=True, help="组B 双机禁用agent continue.txt")
    ap.add_argument("--multi", default=None, help="组C 双agent continue.txt（可选）")
    args = ap.parse_args()

    groups = [
        ("组A 基座", parse_continue(args.base), args.base),
        ("组B 双机禁用agent", parse_continue(args.single), args.single),
    ]
    if args.multi:
        groups.append(("组C 双agent", parse_continue(args.multi), args.multi))

    for name, recs, path in groups:
        print(f"{name}: {len(recs)} 条  {path}")
    print()

    sums = {name: summarize(recs) for name, recs, _ in groups}

    # 逐条对齐对比（以组A为基准）
    n = max(len(recs) for _, recs, _ in groups)
    print("=" * 110)
    header = f"{'epi':>4} {'组A结果':<26}"
    for name, _, _ in groups[1:]:
        header += f" {name:<26}"
    print(header)
    print("=" * 110)
    n_match = 0
    for i in range(n):
        row = f"{i:>4} "
        ref = groups[0][1][i]["result"] if i < len(groups[0][1]) else "(缺)"
        row += f"{ref:<26} "
        all_same = True
        for name, recs, _ in groups[1:]:
            r = recs[i]["result"] if i < len(recs) else "(缺)"
            row += f"{r:<26} "
            if r != ref:
                all_same = False
        if all_same:
            n_match += 1
        print(row)

    print()
    print("=" * 110)
    print("汇总对比")
    print("=" * 110)
    width = 22
    header = f"{'指标':<{width}}"
    for name, _, _ in groups:
        header += f" {name:>16}"
    print(header)
    print(f"{'条数':<{width}}" + "".join(f" {sums[n]['n']:>16}" for n, _, _ in groups))
    print(f"{'Success 数':<{width}}" + "".join(f" {sums[n]['success']:>16}" for n, _, _ in groups))
    print(f"{'Success 率':<{width}}" + "".join(f" {sums[n]['success_rate']:>15.2f}%" for n, _, _ in groups))
    for key, name in [("spl_pct", "SPL"), ("soft_spl_pct", "Soft SPL"), ("dtg", "Dist to Goal")]:
        vals = [sums[n].get(key) for n, _, _ in groups]
        row = f"{name:<{width}}"
        for v in vals:
            row += f" {str(v) if v is not None else '-':>16}"
        print(row)
    print(f"{'逐条结果一致数(相对组A)':<{width}} {n_match:>16} / {n:>16}")

    print()
    print("失败类型分布")
    all_keys = sorted(set().union(*[set(s["results"]) for s in sums.values()]))
    header = f"{'类型':<30}"
    for name, _, _ in groups:
        header += f" {name:>16}"
    print(header)
    for k in all_keys:
        row = f"{k:<30}"
        for name, _, _ in groups:
            row += f" {sums[name]['results'].get(k, 0):>16}"
        print(row)


if __name__ == "__main__":
    main()
