"""실제 데이터를 넣기 전에 — API 호출 없이 데이터 형식을 검사하고 LLM 쪽 호출 수 · 토큰을 미리 센다.

Jev 쪽(jev_normalize.py)과 LLM 쪽(llm_normalize.py)이 같은 data 폴더를 그대로 읽을 수 있는지,
선택지 수 때문에 LLM 쪽에 추가 처리(--over20, 숫자 라벨)가 필요한지 실행 전에 알려 준다.

    python3 check_data.py data/real                          # 검사 + k=all, k=5 추정
    python3 check_data.py data/sample                        # 가상 데이터로 동작 확인
    python3 check_data.py data/real --price-in 0.4 --price-out 1.6   # LLM 단가(100만 토큰당 $)를 주면 비용까지

종료 코드: 오류가 있으면 1 (경고만 있으면 0).
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import NONE_ID, build_candidates, gold_in_candidates, load_data, parse_period  # 수정 금지 모듈
from jev_normalize import TYPE_KO, build_request  # 수정 금지 모듈
from llm_normalize import SYSTEM, TOP_N, build_prompt, make_labels

# common.py · jev_normalize.py 가 이름으로 직접 읽는 칸 — 없으면 실행 중 KeyError
MENTION_COLS = ["mention_id", "entity_type", "surface", "gold_id", "split", "ctx_client", "ctx_period", "ctx_role"]
MENTION_OPT = ["variant", "missing_client"]  # 없어도 돌지만 score.py 의 유형별 · 결측별 표가 비게 된다
STANDARD_COLS = ["entity_id", "entity_type", "name", "client_id", "period_start", "period_end", "group", "industry"]
STANDARD_OPT = ["aliases"]
SPLITS = {"dev", "test"}  # jev_normalize.py --split 이 받는 값
JEV_MAX = 255             # common.build_candidates 주석: 공식 문서상 목록을 다 주는 상한
MSG_OVERHEAD = 7          # chat 형식 메시지 2개의 구분 토큰(대략). 실측과 비교해 맞춘 값은 아님


def check(data_dir):
    errors, warns = [], []
    standard, mentions = load_data(data_dir)
    if not mentions or not standard:
        return ["mentions.csv 나 standard.csv 가 비어 있다"], warns, standard, mentions

    for name, rows, need, opt in (("mentions.csv", mentions, MENTION_COLS, MENTION_OPT),
                                  ("standard.csv", standard, STANDARD_COLS, STANDARD_OPT)):
        cols = set(rows[0])
        miss = [c for c in need if c not in cols]
        if miss:
            errors.append(f"{name} 에 필수 칸이 없다: {miss}")
        miss_opt = [c for c in opt if c not in cols]
        if miss_opt:
            warns.append(f"{name} 에 선택 칸이 없다: {miss_opt}")
    if errors:
        return errors, warns, standard, mentions

    by_id = {}
    for s in standard:
        if s["entity_type"] not in TYPE_KO:
            errors.append(f"standard {s['entity_id']}: entity_type '{s['entity_type']}' 은 {list(TYPE_KO)} 중 하나여야 한다")
        if s["entity_id"] in by_id:
            errors.append(f"standard entity_id 중복: {s['entity_id']}")
        if s["entity_id"] == NONE_ID:
            errors.append("standard 에 entity_id 'NONE' 이 있다 — 선택지 NONE 과 겹친다")
        if not s["name"].strip():
            errors.append(f"standard {s['entity_id']}: name 이 비어 있다")
        by_id[s["entity_id"]] = s
    for s in standard:
        if s["entity_type"] == "project":
            c = by_id.get(s["client_id"])
            if not c or c["entity_type"] != "client":
                warns.append(f"standard {s['entity_id']}: client_id '{s['client_id']}' 가 client 가 아니다 → 후보 설명에 고객사가 빈칸")
            if not parse_period(f"{s['period_start']}~{s['period_end']}"):
                warns.append(f"standard {s['entity_id']}: 기간 '{s['period_start']}~{s['period_end']}' 을 못 읽는다 → 겹침 '알 수 없음'")

    seen = set()
    no_period = 0
    for m in mentions:
        mid = m["mention_id"]
        if mid in seen:
            errors.append(f"mention_id 중복: {mid}")
        seen.add(mid)
        if m["entity_type"] not in TYPE_KO:
            errors.append(f"{mid}: entity_type '{m['entity_type']}' 은 {list(TYPE_KO)} 중 하나여야 한다")
            continue
        if not m["surface"].strip():
            errors.append(f"{mid}: surface 가 비어 있다")
        g = m["gold_id"]
        if g != NONE_ID and (g not in by_id or by_id[g]["entity_type"] != m["entity_type"]):
            errors.append(f"{mid}: gold_id '{g}' 가 standard 의 같은 유형({m['entity_type']})에 없다 (없는 실체면 NONE)")
        if m["split"] not in SPLITS:
            errors.append(f"{mid}: split '{m['split']}' 은 {sorted(SPLITS)} 중 하나여야 한다")
        if m["entity_type"] == "project" and m["ctx_period"] and not parse_period(m["ctx_period"]):
            no_period += 1
    if no_period:
        warns.append(f"project mention {no_period}건의 ctx_period 를 못 읽는다 → 겹침 '알 수 없음'")
    return errors, warns, standard, mentions


def estimate(standard, mentions, k, price_in, price_out):
    """LLM 쪽 호출 수와 입력 토큰을 프롬프트를 실제로 만들어 센다 (API 호출 없음)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
    except Exception as e:  # tiktoken 이 없으면 토큰 추정만 건너뛴다
        enc = None
        print(f"  (tiktoken 없음 — 토큰 추정 생략: {e})")
    sys_tok = len(enc.encode(SYSTEM)) if enc else 0
    n_opts = defaultdict(list)
    tok = {"fwd": 0, "rev": 0}
    in_cands = 0
    for m in mentions:
        cands = build_candidates(m, standard, k=k)
        in_cands += gold_in_candidates(m, cands)
        body = build_request(m, cands, model="")
        for q in ("fwd", "rev"):
            qq = body["questions"][f"same_{q}"]
            ids = list(qq["criteria"])
            if q == "fwd":
                n_opts[m["entity_type"]].append(len(ids))
            if enc:
                tok[q] += len(enc.encode(build_prompt(body["state"], qq, ids, make_labels(len(ids))))) + sys_tok + MSG_OVERHEAD
    n = len(mentions)
    label = "k=all" if k is None else f"k={k}"
    print(f"\n[{label}]  mention {n}건, 정답이 후보에 든 비율 {in_cands / n:.1%}")
    over20 = 0
    for t, ns in sorted(n_opts.items()):
        mx = max(ns)
        flag = []
        if mx > TOP_N:
            flag.append(f"LLM: {TOP_N}개 초과 → --over20 logit_bias 필요 (기본 error 면 그 건은 오류)")
        if mx > 26:
            flag.append("LLM: 26개 초과 → 숫자 라벨(1,2,3…)")
        if mx > JEV_MAX:
            flag.append(f"Jev: {JEV_MAX}개 초과")
        print(f"  {t:8s} {len(ns):4d}건  선택지 최대 {mx}개{'  ← ' + ' / '.join(flag) if flag else ''}")
        over20 += sum(x > TOP_N for x in ns)
    # 20개 초과 건은 연쇄 조회로 대략 ceil(n/20)회 — 실제로는 라벨 아닌 토큰이 섞여 조금 더 늘 수 있다
    extra = sum(math.ceil(x / TOP_N) - 1 for ns in n_opts.values() for x in ns if x > TOP_N)
    for rev in (False, True):
        calls = (n + extra) * (2 if rev else 1)
        line = f"  LLM {'--rev   ' if rev else 'fwd 만  '} 호출 약 {calls:,}회"
        if enc:
            t_in = tok["fwd"] + (tok["rev"] if rev else 0)
            t_in = round(t_in * (n + extra) / n)
            line += f", 입력 약 {t_in:,} 토큰 (건당 {t_in / n:,.0f}), 출력 {calls:,} 토큰"
            if price_in is not None:
                cost = t_in / 1e6 * price_in + calls / 1e6 * (price_out or 0)
                line += f", 약 ${cost:.3f}"
        print(line)
    if over20:
        print(f"  ※ 선택지 {TOP_N}개 초과 mention {over20}건 — 호출 수에 연쇄 조회 추정치 포함")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="mentions.csv, standard.csv 가 있는 폴더")
    ap.add_argument("--k", default="all,5", help="추정할 k 목록 (쉼표). Jev 와 같은 값")
    ap.add_argument("--price-in", type=float, default=None, help="LLM 입력 100만 토큰당 $")
    ap.add_argument("--price-out", type=float, default=None, help="LLM 출력 100만 토큰당 $")
    a = ap.parse_args()
    if not (Path(a.data) / "mentions.csv").exists() or not (Path(a.data) / "standard.csv").exists():
        sys.exit(f"{a.data} 에 mentions.csv 와 standard.csv 가 있어야 한다")

    errors, warns, standard, mentions = check(a.data)
    types = Counter(m.get("entity_type") for m in mentions)
    print(f"{a.data}: mention {len(mentions)}건 {dict(types)}, standard {len(standard)}개 "
          f"{dict(Counter(s.get('entity_type') for s in standard))}")
    if mentions and "split" in mentions[0]:
        print(f"  split {dict(Counter(m['split'] for m in mentions))}, "
              f"정답 NONE {sum(m.get('gold_id') == NONE_ID for m in mentions)}건")
    for w in warns[:20]:
        print("  경고:", w)
    if len(warns) > 20:
        print(f"  경고 {len(warns) - 20}개 더")
    if errors:
        print(f"\n오류 {len(errors)}개 — 고치기 전에는 Jev · LLM 둘 다 돌리지 말 것")
        for e in errors[:30]:
            print("  -", e)
        sys.exit(1)
    print("형식 OK — Jev · LLM 둘 다 이 폴더를 그대로 읽을 수 있다")
    for k in a.k.split(","):
        estimate(standard, mentions, None if k.strip() == "all" else int(k), a.price_in, a.price_out)


if __name__ == "__main__":
    main()
