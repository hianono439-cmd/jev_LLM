"""LLM 정규화 (Jev 비교용) — LLM_HANDOFF.md 규격 구현.

모델에 들어가는 것(후보·순서·state·지시문·선택지)은 jev_normalize.build_request 결과를 그대로 쓰고,
다른 것은 모델 호출 방식뿐이다. 확률은 첫 토큰 위치의 top_logprobs에서 라벨 확률을 읽어 합 1로 정규화한다.

사용 (common.py, jev_normalize.py, score.py 와 같은 폴더에서):
  python3 llm_normalize.py --probe                                   # 모델이 logprobs/top_logprobs=20/temperature=0 을 받는지, 라벨이 한 토큰인지 확인
  python3 llm_normalize.py --data data/sample --k all --limit 20     # 형식 확인용 소량
  python3 llm_normalize.py --data data/sample --k all --rev          # 본 실행 (+ 역순 질문)
  python3 llm_normalize.py --data data/sample --k 5 --rev
.env: OPENAI_API_KEY, OPENAI_MODEL(기본 gpt-4.1-mini)

선택지가 top_logprobs 상한(20)을 넘는 경우(--over20):
  error      : (기본) 해당 mention을 error로 남긴다. 처리 방식은 이헌준과 합의 전.
  logit_bias : 같은 프롬프트를 다시 보내되 이미 읽은 라벨을 logit_bias -100으로 막아 다음 라벨들을 읽는다.
               막힌 뒤의 확률 p' 와 원래 확률 p 는 p = p' × (1 − 이미 읽은 확률 합) 관계라 정확히 복원된다.
               질문·선택지는 바뀌지 않고 호출만 ceil(n/20)회 안팎으로 늘어난다(토큰·지연은 모두 합산 기록). tiktoken 필요.
"""
import argparse
import json
import math
import os
import time

# common / jev_normalize(수정 금지 모듈)는 실제 정규화할 때만 import 한다 — --probe 는 그 파일 없이도 돈다.

HEAD_KEYS = ["mention_id", "entity_type", "surface", "gold_id", "variant", "missing_client", "split"]
SYSTEM = "선택지 라벨 하나만 답하라."
TOP_N = 20


def load_dotenv(path=".env"):
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if v:
                    os.environ.setdefault(k.strip(), v)


def make_labels(n):
    """A~Z(26개까지), 넘으면 1,2,3…(o200k 토크나이저에서 1~999는 한 토큰 — --probe로 확인)."""
    return [chr(ord("A") + i) for i in range(n)] if n <= 26 else [str(i + 1) for i in range(n)]


def build_prompt(state, q, ids, labels):
    options = "\n".join(f"{lab}. {json.dumps(q['criteria'][i], ensure_ascii=False)}" for lab, i in zip(labels, ids))
    return (f"{q['instructions']}\n\n[state]\n{json.dumps(state, ensure_ascii=False)}\n\n[선택지]\n{options}\n\n"
            f"선택지 라벨 하나만 답하라.")


class LLM:
    def __init__(self, model):
        import openai
        self.o = openai
        self.client = openai.OpenAI(max_retries=0, timeout=60)
        self.model = model
        self._enc = None

    def enc(self):
        if self._enc is None:
            import tiktoken
            try:
                self._enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._enc = tiktoken.get_encoding("o200k_base")
        return self._enc

    def first_token(self, prompt, logit_bias=None, attempts=4):
        """반환 (첫 토큰, {토큰: logprob}, usage, ms). 재시도 대기 포함 시간."""
        kw = dict(model=self.model, temperature=0, logprobs=True, top_logprobs=TOP_N, max_completion_tokens=1,
                  messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        if logit_bias:
            kw["logit_bias"] = logit_bias
        t0 = time.perf_counter()
        for a in range(1, attempts + 1):
            try:
                r = self.client.chat.completions.create(**kw)
                break
            except (self.o.RateLimitError, self.o.APIConnectionError, self.o.InternalServerError):
                if a == attempts:
                    raise
                time.sleep(min(2 ** a, 16))
        ms = (time.perf_counter() - t0) * 1000
        lp = r.choices[0].logprobs.content[0]
        return lp.token, {t.token: t.logprob for t in lp.top_logprobs}, r.usage, ms


def label_probs(llm, prompt, labels, over20):
    """라벨별 확률(정규화 전 원래 확률). 반환 (probs{label:p}, first_token, in_tok, out_tok, ms, calls)."""
    tok, top, u, ms = llm.first_token(prompt)
    in_tok, out_tok, calls = u.prompt_tokens, u.completion_tokens, 1
    probs = {lab: math.exp(top[lab]) for lab in labels if lab in top}
    unseen = [lab for lab in labels if lab not in probs]
    if unseen and len(labels) > TOP_N:
        if over20 != "logit_bias":
            raise RuntimeError(f"선택지 {len(labels)}개 > top_logprobs {TOP_N}: 처리 방식 합의 전(--over20)")
        enc, banned_mass = llm.enc(), 0.0
        while unseen:
            banned = [lab for lab in labels if lab in probs]
            bias = {}
            for lab in banned:
                ids = enc.encode(lab)
                if len(ids) != 1:
                    raise RuntimeError(f"라벨 '{lab}' 이 한 토큰이 아님")
                bias[str(ids[0])] = -100
            if len(bias) > 300:
                raise RuntimeError("logit_bias 300개 상한 초과")
            banned_mass = sum(probs.values())
            _, top2, u2, ms2 = llm.first_token(prompt, logit_bias=bias)
            in_tok += u2.prompt_tokens
            out_tok += u2.completion_tokens
            ms += ms2
            calls += 1
            new = {lab: math.exp(top2[lab]) * (1 - banned_mass) for lab in unseen if lab in top2}
            if not new:   # 남은 라벨의 확률이 사실상 0 (다른 토큰에 밀림)
                break
            probs.update(new)
            unseen = [lab for lab in labels if lab not in probs]
    return probs, tok, in_tok, out_tok, ms, calls


def ask(llm, state, q, over20):
    ids = list(q["criteria"])                 # 순서 그대로, 마지막이 NONE
    labels = make_labels(len(ids))
    probs, tok, in_tok, out_tok, ms, calls = label_probs(llm, build_prompt(state, q, ids, labels), labels, over20)
    if tok.strip() != tok or tok not in labels:
        raise RuntimeError(f"첫 토큰이 라벨이 아님: {tok!r}")
    p = {i: probs.get(lab, 0.0) for lab, i in zip(labels, ids)}
    s = sum(p.values())
    if s <= 0:
        raise RuntimeError("라벨 확률 합이 0")
    p = {k: v / s for k, v in p.items()}
    n = len(ids)
    pred = max(p, key=p.get)
    conf = (p[pred] - 1 / n) / (1 - 1 / n) if n > 1 else 1.0
    return pred, {k: round(v, 6) for k, v in p.items()}, round(conf, 6), in_tok, out_tok, ms, calls


def run_one(llm, m, standard, k, rev, over20, config, repeat=0):
    from common import build_candidates, gold_in_candidates
    from jev_normalize import build_request
    row = {key: m.get(key) for key in HEAD_KEYS}
    row.update(system="llm", config=config, model=llm.model, repeat=repeat)
    try:
        cands = build_candidates(m, standard, k=k)
        body = build_request(m, cands, model="")
        row["cand_ids"] = [c["id"] for c in cands]
        row["gold_in_cands"] = gold_in_candidates(m, cands)
        state, qs = body["state"], body["questions"]
        pred, probs, conf, i1, o1, ms, c1 = ask(llm, state, qs["same_fwd"], over20)
        row.update(pred_id=pred, probabilities=probs, confidence=conf)
        tot_in, tot_out, tot_ms, tot_calls = i1, o1, ms, c1
        if rev:
            pr, pb, cf, i2, o2, ms2, c2 = ask(llm, state, qs["same_rev"], over20)
            row.update(pred_id_rev=pr, probabilities_rev=pb, confidence_rev=cf)
            tot_in, tot_out, tot_ms, tot_calls = tot_in + i2, tot_out + o2, tot_ms + ms2, tot_calls + c2
        row.update(latency_ms=round(tot_ms), input_tokens=tot_in, output_tokens=tot_out, llm_calls=tot_calls)
    except Exception as e:  # 실패 건도 지우지 않고 남긴다(채점기가 오류 수를 셈)
        row["error"] = f"{type(e).__name__}: {e}"[:500]
    return row


def probe(llm):
    """모델이 규격에 필요한 파라미터를 받는지 1회 호출로 확인."""
    print("model:", llm.model)
    try:
        tok, top, u, ms = llm.first_token("다음 중 과일은? A. 사과 B. 자동차 C. 연필\n라벨 하나만 답하라.")
        print(f"OK  첫 토큰={tok!r}  top_logprobs {len(top)}개  {ms:.0f}ms  입력 {u.prompt_tokens} 토큰")
        print("    상위:", sorted(top.items(), key=lambda x: -x[1])[:5])
    except Exception as e:
        print("실패:", type(e).__name__, str(e)[:300])
        print("→ 이 모델은 logprobs/top_logprobs/temperature 조합을 지원하지 않을 수 있음(추론 모델). 비추론 모델로 바꿀 것.")
    try:
        enc = llm.enc()
        bad = [s for s in make_labels(26) + make_labels(200) if len(enc.encode(s)) != 1]
        print("한 토큰이 아닌 라벨:", bad[:20] or "없음")
    except Exception as e:
        print("tiktoken 확인 생략:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--k", default="all", help="all 또는 정수. Jev 실행과 같은 값")
    ap.add_argument("--model", default=None)
    ap.add_argument("--rev", action="store_true", help="same_rev(역순)로 한 번 더")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default=None, help="이 split만 (예: dev)")
    ap.add_argument("--over20", choices=["error", "logit_bias"], default="error")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    load_dotenv()
    llm = LLM(a.model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    if a.probe:
        return probe(llm)
    k = None if a.k == "all" else int(a.k)
    config = "k=all" if k is None else f"k={k}"
    from common import load_data
    standard, mentions = load_data(a.data)
    if a.split:
        mentions = [m for m in mentions if m.get("split") == a.split]
    if a.limit:
        mentions = mentions[: a.limit]
    os.makedirs("out", exist_ok=True)
    path = f"out/llm_k{a.k}_{time.strftime('%m%d-%H%M')}.jsonl"
    n_err = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in range(a.repeat):
            for i, m in enumerate(mentions):
                row = run_one(llm, m, standard, k, a.rev, a.over20, config, repeat=r)
                n_err += "error" in row
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 20 == 0:
                    print(f"  {i + 1}/{len(mentions)}  오류 {n_err}")
    print(f"완료 {path}  (오류 {n_err})")


if __name__ == "__main__":
    main()
