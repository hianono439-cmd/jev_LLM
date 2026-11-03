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
               ' A' 처럼 공백 붙은 형태도 함께 막고, 남은 라벨 확률 상한이 1e-4 미만이면 멈춘다(이헌준 조건 2·3).
               사용 전 --verify-chain 10 으로 복원 오차를 숫자로 남긴다(조건 1).
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


STOP_MASS = 1e-4   # 연쇄 조회 멈춤: 남은 라벨 하나가 가질 수 있는 확률 상한이 이보다 작으면 멈춘다


def _label_token_ids(enc, lab):
    """라벨의 모든 형태('A', ' A')의 토큰 ID. 한 토큰인 형태만 막을 수 있으므로 '라벨' 자체가 한 토큰이 아니면 오류."""
    ids = enc.encode(lab)
    if len(ids) != 1:
        raise RuntimeError(f"라벨 '{lab}' 이 한 토큰이 아님")
    out = {lab: ids[0]}
    sp = enc.encode(" " + lab)
    if len(sp) == 1:
        out[" " + lab] = sp[0]
    return out


def label_probs(llm, prompt, labels, over20, first_take=None):
    """라벨별 확률(전체 어휘 기준, 라벨끼리 정규화 전). 반환 (probs{label:p}, first_token, in_tok, out_tok, ms, calls).

    선택지가 top_logprobs 상한(20)을 넘거나 first_take(검증용: 첫 호출에서 상위 몇 개 라벨만 읽은 척)가 있으면
    logit_bias 연쇄 조회를 한다. 호출 r 에서 막힌 토큰들의 실제 확률 합을 B 라 하면, 막힌 뒤 분포 p_r 와 실제 확률 p 는
    p = p_r × (1 − B). B 는 지금까지 읽은 토큰(라벨과 ' 라벨' 형태)의 실제 확률 합이다.
    """
    tok, top, u, ms = llm.first_token(prompt)
    in_tok, out_tok, calls = u.prompt_tokens, u.completion_tokens, 1
    true_p = {t: math.exp(lp) for t, lp in top.items()}          # 토큰 → 실제 확률(전체 어휘 기준)
    probs = {lab: true_p[lab] for lab in labels if lab in true_p}
    if first_take is not None:
        keep = sorted(probs, key=probs.get, reverse=True)[:first_take]
        probs = {lab: probs[lab] for lab in keep}
    unseen = [lab for lab in labels if lab not in probs]
    need_chain = unseen and (len(labels) > TOP_N or first_take is not None)
    if need_chain:
        if over20 != "logit_bias" and first_take is None:
            raise RuntimeError(f"선택지 {len(labels)}개 > top_logprobs {TOP_N}: 처리 방식 합의 전(--over20)")
        enc = llm.enc()
        forms = {lab: _label_token_ids(enc, lab) for lab in labels}
        floor = min(true_p.values()) if true_p else 0.0              # 직전 호출에서 읽힌 가장 작은 확률(=안 읽힌 것의 상한)
        tok_id = {}                                                    # 막을 수 있는(한 토큰으로 되돌아가는) 토큰 → ID
        for f in forms.values():
            tok_id.update(f)
        while unseen and floor >= STOP_MASS:
            # 이미 읽은 토큰은 라벨이든 아니든(top_logprobs 자리를 차지하는 다른 토큰 포함) 모두 막아야 다음 순위가 보인다
            for t in true_p:
                if t not in tok_id:
                    ids = enc.encode(t)
                    if len(ids) == 1:
                        tok_id[t] = ids[0]
            blocked = {t: tok_id[t] for t in true_p if t in tok_id}
            for lab in probs:
                blocked.update(forms[lab])
            if len(blocked) > 300:
                raise RuntimeError("logit_bias 300개 상한 초과")
            B = sum(true_p.get(t, 0.0) for t in blocked)                # 못 본 ' 라벨' 형태는 0으로 근사(floor 미만)
            _, top2, u2, ms2 = llm.first_token(prompt, logit_bias={str(i): -100 for i in blocked.values()})
            in_tok += u2.prompt_tokens
            out_tok += u2.completion_tokens
            ms += ms2
            calls += 1
            scale = max(1.0 - B, 0.0)
            fresh = {t: math.exp(lp) * scale for t, lp in top2.items() if t not in blocked}
            if not fresh:
                break
            for t, v in fresh.items():
                true_p.setdefault(t, v)
            probs.update({lab: true_p[lab] for lab in unseen if lab in true_p})
            unseen = [lab for lab in labels if lab not in probs]
            floor = min(fresh.values())
    return probs, tok, in_tok, out_tok, ms, calls


def verify_chain(llm, data, n=10, first_take=5):
    """헌준 조건 1: 선택지 20개 이하 mention 에서 (가) 한 번에 읽은 확률 vs (나) 상위 first_take 개만 읽고 연쇄로 복원한 확률."""
    from common import build_candidates, load_data
    from jev_normalize import build_request
    standard, mentions = load_data(data)
    done, worst = 0, 0.0
    for m in mentions:
        cands = build_candidates(m, standard, k=None)
        q = build_request(m, cands, model="")["questions"]["same_fwd"]
        ids = list(q["criteria"])
        if len(ids) > TOP_N or len(ids) <= first_take + 2:
            continue
        labels = make_labels(len(ids))
        prompt = build_prompt(build_request(m, cands, model="")["state"], q, ids, labels)
        a = label_probs(llm, prompt, labels, "error")[0]
        b, *_rest, calls = label_probs(llm, prompt, labels, "logit_bias", first_take=first_take)
        diff = max(abs(a.get(l, 0.0) - b.get(l, 0.0)) for l in labels)
        worst = max(worst, diff)
        top3 = sorted(a, key=a.get, reverse=True)[:3]
        print(f"{m['mention_id']} 선택지 {len(ids)}  연쇄 호출 {calls}  최대 절대차 {diff:.2e}  "
              f"상위3 한번에 {[round(a[l], 4) for l in top3]} / 복원 {[round(b.get(l, 0), 4) for l in top3]}")
        done += 1
        if done >= n:
            break
    print(f"검증 {done}건, 라벨 확률 최대 절대차 {worst:.2e}")


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
    ap.add_argument("--verify-chain", type=int, default=0, help="연쇄 조회 복원 검증 n건 (선택지 20개 이하 mention, 상위 5개만 읽고 복원)")
    a = ap.parse_args()
    load_dotenv()
    llm = LLM(a.model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    if a.probe:
        return probe(llm)
    if a.verify_chain:
        return verify_chain(llm, a.data, n=a.verify_chain)
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
