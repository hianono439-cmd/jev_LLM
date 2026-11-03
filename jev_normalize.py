"""정규화 — Jev 쪽 (담당 이헌준).

mention 하나당 요청 하나. 같은 요청 안에 질문 셋을 넣는다(질문은 서로 독립으로 병렬 판정 — 공식 문서):
  same_fwd  Choice  후보를 유사도 순으로 + NONE
  same_rev  Choice  같은 후보를 거꾸로 + NONE   → 순서 편향(공식 약점 「첫 선택지 쪽으로 기운다」) 측정
  any_match Noul    「후보 중 같은 게 있나」     → 공식 문서가 NONE 선택지 대신 권하는 방식

    python3 jev_normalize.py --mock               # 키 없이 파이프라인만 (가짜 확률)
    python3 jev_normalize.py --show-request       # 첫 요청 본문만 찍고 끝
    python3 jev_normalize.py --limit 20           # 실제 호출, 20건만
    python3 jev_normalize.py --k 5                # 후보를 상위 5개로 좁혀서
"""
import argparse
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from common import NONE_ID, ROOT, build_candidates, gold_in_candidates, load_data, load_env

API = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"  # 별칭(jev-latest) 대신 버전 고정 — 문턱값을 맞춘 뒤 모델이 바뀌면 안 된다(공식 문서)
TYPE_KO = {"project": "프로젝트", "client": "고객사", "skill": "기술"}


def describe(c):
    d = {"이름": c["name"]}
    for k_en, k_ko in (("client", "고객사"), ("period", "기간"), ("overlap", "이력 기간과 겹침"),
                       ("group", "그룹"), ("industry", "업종"), ("aliases", "알려진 다른 표기")):
        if c.get(k_en):
            d[k_ko] = c[k_en]
    return d


def build_request(m, cands, model, questions=("fwd", "rev", "noul")):
    t = TYPE_KO[m["entity_type"]]
    state = {"표기": m["surface"], "유형": t, "후보 목록": [c["name"] for c in cands]}
    if m["entity_type"] == "project":
        state["같은 이력"] = {"고객사": m["ctx_client"] or "(적혀 있지 않음)",
                          "기간": m["ctx_period"] or "(적혀 있지 않음)", "역할": m["ctx_role"]}
    instr = (f"`표기` 는 아래 선택지 중 어느 {t}와 같은 대상을 가리키는가? "
             "표기는 띄어쓰기 · 줄임말 · 영문 · 오타 때문에 이름과 다를 수 있다.")
    if m["entity_type"] == "project":
        instr += (" `같은 이력` 의 고객사와 기간을 함께 보라. 같은 고객사라도 단계 · 차수가 다른 사업은 다른 프로젝트다. "
                  "그룹이 같아도 계열사가 다르면 다른 고객사다.")
    none_desc = f"선택지 중 같은 {t}가 없다 — 표준 목록에 없는 새 {t}"
    fwd = {c["id"]: describe(c) for c in cands}
    fwd[NONE_ID] = none_desc
    rev = {c["id"]: describe(c) for c in reversed(cands)}
    rev[NONE_ID] = none_desc
    qs = {
        "same_fwd": {"type": "choice", "instructions": instr, "criteria": fwd},
        "same_rev": {"type": "choice", "instructions": instr, "criteria": rev},
        "any_match": {"type": "noul", "instructions": f"`표기` 는 `후보 목록` 중 하나와 같은 {t}를 가리키는가?"},
    }
    keep = {"fwd": "same_fwd", "rev": "same_rev", "noul": "any_match"}
    return {"model": model, "state": state, "questions": {keep[q]: qs[keep[q]] for q in questions}}


def call_jev(body, key, tries=5):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    for i in range(tries):
        req = urllib.request.Request(API, data=data, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and i < tries - 1:  # 공식 문서: 지수 백오프로 재시도
                time.sleep(2 ** i + random.random())
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")


def fake_response(m, cands, body):
    """--mock — 파이프라인 확인용 가짜 확률. 숫자에 의미 없다."""
    rng = random.Random(m["mention_id"])

    def dist(order):
        logits = {}
        for pos, cid in enumerate(order):
            if cid == NONE_ID:
                top = max([c["sim"] for c in cands] or [0])
                logits[cid] = 4 * (0.55 - top) + rng.gauss(0, 0.5)
            else:
                sim = next(c["sim"] for c in cands if c["id"] == cid)
                logits[cid] = 6 * sim + rng.gauss(0, 0.6) - 0.15 * pos  # 앞자리 가점 = 순서 편향 흉내
        z = max(logits.values())
        ex = {k: math.exp(v - z) for k, v in logits.items()}
        s = sum(ex.values())
        p = {k: v / s for k, v in ex.items()}
        best = max(p, key=p.get)
        n = len(p)
        return {"type": "choice", "choice": best, "probabilities": p,
                "confidence": (p[best] - 1 / n) / (1 - 1 / n)}

    q = body["questions"]
    fwd = dist(list(q["same_fwd"]["criteria"]))
    answers = {"same_fwd": fwd}
    if "same_rev" in q:
        answers["same_rev"] = dist(list(q["same_rev"]["criteria"]))
    if "any_match" in q:
        answers["any_match"] = {"type": "noul", "noul": 1 - fwd["probabilities"][NONE_ID]}
    return {"model": "mock", "answers": answers,
            "usage": {"input_tokens": len(json.dumps(body, ensure_ascii=False)) // 2, "output_tokens": 0}}


def run_one(m, standard, args, key):
    cands = build_candidates(m, standard, k=args.k)
    body = build_request(m, cands, args.model, args.questions)
    t0 = time.perf_counter()
    rec = {k: m[k] for k in ("mention_id", "entity_type", "surface", "gold_id", "variant", "missing_client", "split")}
    rec.update(system="jev", config=f"k={args.k or 'all'}", questions="+".join(args.questions), cand_ids=[c["id"] for c in cands],
               gold_in_cands=gold_in_candidates(m, cands))
    try:
        resp = fake_response(m, cands, body) if args.mock else call_jev(body, key)
    except Exception as e:  # 한 건 실패로 전체를 멈추지 않는다
        rec.update(error=str(e))
        return rec
    a = resp["answers"]
    rec.update(model=resp["model"], latency_ms=round((time.perf_counter() - t0) * 1000),
               pred_id=a["same_fwd"]["choice"], probabilities=a["same_fwd"]["probabilities"],
               confidence=a["same_fwd"]["confidence"],
               input_tokens=resp["usage"]["input_tokens"], output_tokens=resp["usage"].get("output_tokens", 0))
    if "same_rev" in a:
        rec.update(pred_id_rev=a["same_rev"]["choice"], probabilities_rev=a["same_rev"]["probabilities"],
                   confidence_rev=a["same_rev"]["confidence"])
    if "any_match" in a:
        rec["noul_any"] = a["any_match"]["noul"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "sample"))
    ap.add_argument("--k", type=int, default=None, help="후보 상위 k개 (기본: 그 유형 전체)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1, help="같은 요청을 n번 — 반복 일관성(A-3 「같은 판정 반복」 이 의미 있나)")
    ap.add_argument("--questions", default="fwd,rev,noul",
                    help="한 요청에 넣을 질문. 비용 · 지연을 LLM 과 같은 기준으로 잴 때는 fwd 만")
    ap.add_argument("--workers", type=int, default=4, help="동시 요청 수. 지연을 잴 때는 1")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--show-request", action="store_true")
    args = ap.parse_args()
    args.questions = tuple(q.strip() for q in args.questions.split(",") if q.strip())

    load_env()
    key = os.environ.get("TYPESAFE_API_KEY", "")
    standard, mentions = load_data(args.data)
    if args.split != "all":
        mentions = [m for m in mentions if m["split"] == args.split]
    mentions = mentions[: args.limit]

    if args.show_request:
        m = next(x for x in mentions if x["entity_type"] == "project")
        print(json.dumps(build_request(m, build_candidates(m, standard, k=args.k), args.model, args.questions),
                         ensure_ascii=False, indent=2))
        return
    if not args.mock and not key:
        raise SystemExit(".env 에 TYPESAFE_API_KEY 가 없다. 키 없이 돌려 보려면 --mock")

    out_dir = ROOT / "out"
    out_dir.mkdir(exist_ok=True)
    qtag = "" if args.questions == ("fwd", "rev", "noul") else "_" + "+".join(args.questions)
    tag = f"jev{'-mock' if args.mock else ''}_k{args.k or 'all'}{qtag}_{datetime.now():%m%d-%H%M}"
    out = out_dir / f"{tag}.jsonl"
    with open(out, "w", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        for r in range(args.repeat):
            for rec in ex.map(lambda m: run_one(m, standard, args, key), mentions):
                rec["repeat"] = r
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{len(mentions) * args.repeat}건 → {out}")
    print(f"채점: python3 score.py {out}")


if __name__ == "__main__":
    main()
