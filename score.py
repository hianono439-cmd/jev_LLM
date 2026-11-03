"""정규화 결과 채점 — Jev · LLM 공통. 결과 파일(jsonl)을 여러 개 주면 나란히 비교한다.

결과 한 줄에 필요한 칸 (LLM 쪽도 이 이름 그대로 쓰면 이 스크립트로 채점된다):
  mention_id gold_id entity_type variant missing_client split   ← mentions.csv 에서 그대로
  system("jev"|"llm") config pred_id probabilities{id: p} confidence
  있으면 더 잰다: pred_id_rev(순서 뒤집은 답) noul_any(후보 중 같은 게 있을 확률) gold_in_cands
                  repeat(반복 회차) latency_ms input_tokens output_tokens error

    python3 score.py out/jev_kall_1103-1500.jsonl out/llm_kall_1103-1500.jsonl
    python3 score.py out/llm_....jsonl --price-in 0.15 --price-out 0.6   # LLM 단가(100만 토큰당 $)
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

JEV_PRICE_IN = 0.042  # 입력 100만 토큰당 $, 출력 무료
TARGET = 0.95
BINS = [(0, .5), (.5, .7), (.7, .8), (.8, .9), (.9, .95), (.95, 1.01)]


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ok = [r for r in rows if not r.get("error")]
    return rows, ok


def acc(rs, key="pred_id"):
    return sum(r[key] == r["gold_id"] for r in rs) / len(rs) if rs else float("nan")


def pct(x):
    return "—" if x != x else f"{x * 100:.1f}%"


def threshold_for(rs, target=TARGET):
    """확신도 내림차순으로 자동 처리할 때 정답률 target 을 지키는 가장 낮은 문턱. 없으면 None."""
    best = None
    srt = sorted(rs, key=lambda r: -r["confidence"])
    hit = 0
    for i, r in enumerate(srt, 1):
        hit += r["pred_id"] == r["gold_id"]
        if hit / i >= target:
            best = r["confidence"]
    return best


def ece(rs, n_bins=10):
    """p_max(선택한 답의 확률) 기준 기대 보정 오차."""
    b = defaultdict(list)
    for r in rs:
        p = max(r["probabilities"].values())
        b[min(int(p * n_bins), n_bins - 1)].append((p, r["pred_id"] == r["gold_id"]))
    n = len(rs)
    return sum(len(v) / n * abs(statistics.mean(p for p, _ in v) - statistics.mean(c for _, c in v)) for v in b.values())


def report(path, args):
    rows, ok = load(path)
    first = [r for r in ok if r.get("repeat", 0) == 0]
    name = Path(path).stem
    sysname = first[0]["system"] if first else "?"
    L = [f"## {name}", "", f"- system `{sysname}` · config `{first[0].get('config', '') if first else ''}` · "
         f"model `{first[0].get('model', '') if first else ''}` · {len(first)}건 (오류 {len(rows) - len(ok)})"]
    if not first:
        return "\n".join(L), {}
    if first[0].get("model") == "mock":
        L.append("- mock: 가짜 확률이다. 숫자에 의미 없음. 파이프라인 확인용")

    S = {"정답률": acc(first)}
    L += ["", "| 구분 | 건수 | 정답률 |", "|---|---|---|", f"| 전체 | {len(first)} | **{pct(S['정답률'])}** |"]
    for t in ("project", "client", "skill"):
        g = [r for r in first if r["entity_type"] == t]
        if g:
            L.append(f"| {t} | {len(g)} | {pct(acc(g))} |")
    for mc, lab in (("0", "project · 고객사 있음"), ("1", "project · **고객사 결측**")):
        g = [r for r in first if r["entity_type"] == "project" and str(r["missing_client"]) == mc]
        if g:
            L.append(f"| {lab} | {len(g)} | {pct(acc(g))} |")
            S[lab] = acc(g)

    byv = defaultdict(list)
    for r in first:
        byv[r["variant"]].append(r)
    L += ["", "변형별 — " + " · ".join(f"{v} {pct(acc(g))}({len(g)})" for v, g in sorted(byv.items(), key=lambda x: acc(x[1])))]

    if any("gold_in_cands" in r for r in first):
        S["후보 재현율"] = sum(bool(r.get("gold_in_cands")) for r in first) / len(first)
        L.append(f"- **후보 재현율**(정답이 후보 안에 있음) {pct(S['후보 재현율'])} — 이게 정답률의 천장")

    gold_none = [r for r in first if r["gold_id"] == "NONE"]
    pred_none = [r for r in first if r["pred_id"] == "NONE"]
    if gold_none:
        rec = sum(r["pred_id"] == "NONE" for r in gold_none) / len(gold_none)
        prec = sum(r["gold_id"] == "NONE" for r in pred_none) / len(pred_none) if pred_none else float("nan")
        L.append(f"- 「없음」 선택지 — 새 실체를 잡은 비율 {pct(rec)} ({len(gold_none)}건 중) · 「없음」 이라 한 것 중 맞은 비율 {pct(prec)}")
        if any("noul_any" in r for r in first):
            gate = lambda r: "NONE" if r["noul_any"] < 0.5 else r["pred_id"]
            for r in first:
                r["pred_gate"] = gate(r)
            S["정답률(Noul 관문)"] = acc(first, "pred_gate")
            rec2 = sum(r["pred_gate"] == "NONE" for r in gold_none) / len(gold_none)
            L.append(f"- Noul 관문(「같은 게 있나」 < 0.5 면 NONE) — 정답률 {pct(S['정답률(Noul 관문)'])} · 새 실체 잡은 비율 {pct(rec2)}")
        wrong_new = sum(r["pred_id"] == "NONE" and r["gold_id"] != "NONE" for r in first)
        L.append(f"- 있는 실체를 「없음」 으로 판정 {wrong_new}건 (그대로 새 실체로 넣으면 중복 실체가 된다, 11/2 회의 A-2)")

    L += ["", "**확신도 구간별 실제 정답률**", "", "| 확신도 | 건수 | 정답률 |", "|---|---|---|"]
    for lo, hi in BINS:
        g = [r for r in first if lo <= r["confidence"] < hi]
        L.append(f"| {lo:.2f}~{min(hi, 1):.2f} | {len(g)} | {pct(acc(g))} |")
    S["ECE"] = ece(first)
    L.append(f"\n- ECE(p_max 기준, 10구간) **{S['ECE']:.3f}**")

    dev = [r for r in first if r["split"] == "dev"]
    test = [r for r in first if r["split"] == "test"]
    t = threshold_for(dev) if dev else None
    if t is None:
        L.append(f"- 자동 반영 문턱 — dev 에서 정답률 {TARGET:.0%} 를 지키는 문턱이 없다")
    else:
        auto = [r for r in test if r["confidence"] >= t]
        S["자동 처리율(test)"] = len(auto) / len(test) if test else float("nan")
        L.append(f"- **자동 반영 문턱**(dev 에서 정답률 {TARGET:.0%} 유지) 확신도 ≥ {t:.3f} → test 에서 "
                 f"**{pct(S['자동 처리율(test)'])} 자동 처리**, 그 정답률 {pct(acc(auto))}")

    if any("pred_id_rev" in r for r in first):
        same = [r for r in first if r["pred_id"] == r.get("pred_id_rev")]
        diff = [r for r in first if r["pred_id"] != r.get("pred_id_rev")]
        S["순서 일관성"] = len(same) / len(first)
        first_pos = sum(r["pred_id"] == r["cand_ids"][0] for r in first if r.get("cand_ids")) / len(first)
        L.append(f"- **후보 순서를 뒤집어도 같은 답** {pct(S['순서 일관성'])} — 같을 때 정답률 {pct(acc(same))} · 다를 때 {pct(acc(diff))}"
                 f" · 첫 자리 후보를 고른 비율 {pct(first_pos)}")

    reps = defaultdict(set)
    for r in ok:
        reps[r["mention_id"]].add(r["pred_id"])
    if any(r.get("repeat", 0) > 0 for r in ok):
        S["반복 일관성"] = sum(len(v) == 1 for v in reps.values()) / len(reps)
        L.append(f"- **같은 입력 반복 시 같은 답** {pct(S['반복 일관성'])} — 100% 면 A-3 「중간 → 같은 판정 반복」 은 효과가 없다")

    tin = sum(r.get("input_tokens", 0) for r in first)
    tout = sum(r.get("output_tokens", 0) for r in first)
    if sysname == "jev":
        cost = tin / 1e6 * JEV_PRICE_IN
    else:
        cost = (tin * args.price_in + tout * args.price_out) / 1e6 if args.price_in is not None else float("nan")
    lat = [r["latency_ms"] for r in first if "latency_ms" in r]
    S["비용/1000건"] = cost / len(first) * 1000 if cost == cost else float("nan")
    L.append(f"- 토큰 입력 {tin:,} · 출력 {tout:,} · 비용 ${cost:.4f} (1,000건당 ${S['비용/1000건']:.4f})"
             + (f" · 지연 중앙 {statistics.median(lat):.0f}ms · p95 {sorted(lat)[int(len(lat) * .95) - 1]:.0f}ms" if lat else ""))
    return "\n".join(L), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--price-in", type=float, default=None, help="LLM 입력 100만 토큰당 $")
    ap.add_argument("--price-out", type=float, default=0.0, help="LLM 출력 100만 토큰당 $")
    args = ap.parse_args()
    parts, summary = [], {}
    for f in args.files:
        text, S = report(f, args)
        parts.append(text)
        summary[Path(f).stem] = S
    if len(args.files) > 1:
        keys = sorted({k for S in summary.values() for k in S}, key=lambda k: list(next(iter(summary.values()))).index(k)
                      if k in next(iter(summary.values())) else 99)
        head = "| 지표 | " + " | ".join(summary) + " |"
        rows = [head, "|---|" + "---|" * len(summary)]
        for k in keys:
            vals = []
            for S in summary.values():
                v = S.get(k, float("nan"))
                vals.append(f"{v:.3f}" if k in ("ECE",) else (f"${v:.4f}" if "비용" in k else pct(v)))
            rows.append(f"| {k} | " + " | ".join(vals) + " |")
        parts.insert(0, "## 나란히 비교\n\n" + "\n".join(rows))
    out = "\n\n".join(parts)
    print(out)
    dest = Path(args.files[0]).with_suffix(".report.md") if len(args.files) == 1 else Path(args.files[0]).parent / "compare.report.md"
    dest.write_text(out + "\n", encoding="utf-8")
    print(f"\n→ {dest}")


if __name__ == "__main__":
    main()
