"""실행 로그 · mention 별 한 줄 · 실행 후 요약 — Jev(jev_normalize.py) · LLM(llm_normalize.py) 공용.

같은 함수로 찍으니 두 시스템의 터미널 출력과 out/*.log 형식이 같다. 채점의 기준은 score.py 이고, 이건 눈으로 보는 용도.
"""
import logging
import statistics
from collections import Counter, defaultdict

log = logging.getLogger("normalize")


def setup_log(path=None, quiet=False):
    """터미널 + (path 가 있으면) 로그 파일. quiet 면 mention 별 줄(DEBUG)은 파일에만."""
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    term = logging.StreamHandler()
    term.setLevel(logging.INFO if quiet else logging.DEBUG)
    term.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(term)
    if path:
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        log.addHandler(fh)


def log_row(row, i, n):
    """mention 하나 = 한 줄. 표기 → 답 (정답 ✓/✗) 확신도 · 역순 답 · 지연 · 토큰 · 호출 수."""
    head = f"[{i:>{len(str(n))}}/{n}] {row['mention_id']} {row['entity_type']:7s} {row['surface']!r}"
    if "error" in row:
        log.warning(f"{head} → 오류: {row['error']}")
        return
    gold = row.get("gold_id")
    mark = "" if not gold else ("✓" if row["pred_id"] == gold else f"✗ 정답 {gold}")
    rev = f"  역순 {row['pred_id_rev']}{'' if row['pred_id_rev'] == row['pred_id'] else ' (다름)'}" if "pred_id_rev" in row else ""
    log.debug(f"{head} → {row['pred_id']} {mark}  확신도 {row['confidence']:.3f}{rev}  "
              f"{row['latency_ms']}ms  입력 {row['input_tokens']}  호출 {row.get('llm_calls', 1)}")


def summarize(rows, path, price_in=None, price_out=None, elapsed=None):
    """결과 행 → 터미널 요약. 채점의 기준은 score.py 이고, 이건 실행 직후 눈으로 보는 용도."""
    rows = [r for r in rows if r.get("repeat", 0) == 0]
    ok = [r for r in rows if "error" not in r]
    graded = [r for r in ok if r.get("gold_id")]
    models = sorted({r.get("model") for r in rows if r.get("model")})
    log.info(f"\n[요약] {path}")
    log.info(f"  모델 {', '.join(models) or '?'} · {rows[0].get('config') if rows else '?'} · mention {len(rows)}건 · 오류 {len(rows) - len(ok)}건")
    if graded:
        hit = sum(r["pred_id"] == r["gold_id"] for r in graded)
        log.info(f"  정답률 {hit / len(graded):.1%} ({hit}/{len(graded)})")
        by = defaultdict(list)
        for r in graded:
            by[r["entity_type"]].append(r["pred_id"] == r["gold_id"])
        log.info("  유형별 " + " · ".join(f"{t} {sum(v) / len(v):.1%}({len(v)})" for t, v in sorted(by.items())))
        none = [r for r in graded if r["gold_id"] == "NONE"]
        if none:
            log.info(f"  정답 NONE {len(none)}건 중 맞힘 {sum(r['pred_id'] == 'NONE' for r in none)}")
        wrong = [r for r in graded if r["pred_id"] != r["gold_id"]]
        for r in wrong[:10]:
            log.info(f"    ✗ {r['mention_id']} {r['surface']!r} → {r['pred_id']} (정답 {r['gold_id']}, 확신도 {r['confidence']:.3f})")
        if len(wrong) > 10:
            log.info(f"    ✗ {len(wrong) - 10}건 더")
    elif ok:
        log.info("  정답(gold_id) 없음 — 정답률 생략, 비용 · 시간만")
    if ok:
        conf = Counter("1.0" if r["confidence"] >= 0.9995 else "≥0.95" if r["confidence"] >= 0.95 else "≥0.8" if r["confidence"] >= 0.8 else "<0.8" for r in ok)
        log.info(f"  확신도 분포 " + " · ".join(f"{k} {conf[k]}" for k in ("1.0", "≥0.95", "≥0.8", "<0.8") if conf[k]))
        rv = [r for r in ok if "pred_id_rev" in r]
        if rv:
            log.info(f"  순서 뒤집어도 같은 답 {sum(r['pred_id'] == r['pred_id_rev'] for r in rv) / len(rv):.1%}")
        lat = sorted(r["latency_ms"] for r in ok)
        t_in, t_out = sum(r["input_tokens"] for r in ok), sum(r["output_tokens"] for r in ok)
        calls = sum(r.get("llm_calls", 1) for r in ok)
        log.info(f"  지연 중앙값 {statistics.median(lat):.0f}ms · p95 {lat[min(len(lat) - 1, int(len(lat) * 0.95))]}ms"
                 + (f" · 전체 {elapsed:.0f}s" if elapsed else ""))
        log.info(f"  호출 {calls}회 · 입력 {t_in:,} 토큰 · 출력 {t_out:,} 토큰 (mention 당 입력 {t_in / len(ok):,.0f})")
        if price_in is not None:
            cost = t_in / 1e6 * price_in + t_out / 1e6 * (price_out or 0)
            log.info(f"  비용 ${cost:.4f} (1,000건당 ${cost / len(ok) * 1000:.3f}) — 입력 ${price_in}/1M, 출력 ${price_out or 0}/1M")
    log.info(f"  자세한 채점: python3 score.py <jev결과> {path}")
