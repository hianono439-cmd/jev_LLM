"""Jev 결과와 LLM 결과가 같은 조건에서 나왔는지 확인한다 — 같은 mention · 같은 후보 순서 · 같은 선택지.

    python3 check_parity.py out/jev_kall_….jsonl out/llm_kall_….jsonl
"""
import json
import sys


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("repeat", 0) == 0:
                out[r["mention_id"]] = r
    return out


def main(a_path, b_path):
    a, b = load(a_path), load(b_path)
    problems = []
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a:
        problems.append(f"{a_path} 에만 있는 mention {len(only_a)}개 (예: {only_a[:3]})")
    if only_b:
        problems.append(f"{b_path} 에만 있는 mention {len(only_b)}개 (예: {only_b[:3]})")
    configs = {r.get("config") for r in a.values()} | {r.get("config") for r in b.values()}
    if len(configs) > 1:
        problems.append(f"config 가 다르다: {sorted(map(str, configs))}")
    bad_cands = bad_opts = bad_gold = 0
    for mid in sorted(set(a) & set(b)):
        ra, rb = a[mid], b[mid]
        if ra.get("gold_id") != rb.get("gold_id") or ra.get("split") != rb.get("split"):
            bad_gold += 1
        if ra.get("cand_ids") != rb.get("cand_ids"):
            bad_cands += 1
            if bad_cands <= 3:
                ca, cb = ra.get("cand_ids") or [], rb.get("cand_ids") or []
                problems.append(f"{mid} 후보가 다르다: {len(ca)}개 {ca[:4]}… vs {len(cb)}개 {cb[:4]}…")
        if not ra.get("error") and not rb.get("error") and set(ra.get("probabilities", {})) != set(rb.get("probabilities", {})):
            bad_opts += 1
    for n, what in ((bad_gold, "정답·분할이 다른 mention"), (bad_cands, "후보 순서가 다른 mention"), (bad_opts, "선택지 집합이 다른 mention")):
        if n:
            problems.append(f"{what} {n}개")
    if problems:
        print("불일치 — 비교 전에 맞출 것")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print(f"일치 — mention {len(a)}개, 후보 순서 · 선택지 · 정답 · 분할 모두 같다")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
