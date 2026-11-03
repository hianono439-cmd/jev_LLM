"""실제 데이터 → 비교 입력(mentions.csv · standard.csv) 변환.

추출 결과는 「이력 한 건 = 한 줄, 7개 항목」 인데, 비교 코드(common.load_data)는 「표기 하나 = 한 줄」 을 읽는다.
이 스크립트가 그 사이를 잇는다. Jev 쪽 · LLM 쪽 모두 이 출력 폴더를 --data 로 그대로 쓴다.

    python3 prepare_real.py --records 추출결과.csv --standard 표준목록.csv --out data/real_full
    python3 prepare_real.py --records 결측세트.csv  --standard 표준목록.csv --out data/real_missing

입력 1. 레코드 (CSV 또는 JSON 배열). 칸 이름은 한글/영문 둘 다 받는다. 빈 칸 = 결측.
  프로젝트명(project) 고객사(client) 역할(role) 기술(skills) 기간(period) 시작일자(start_date) 종료일자(end_date)
  사람ID(person_id, 선택): 같은 사람의 이력을 dev/test 한쪽에 모으는 데만 쓴다. 없으면 레코드마다 따로 나눈다.
  기술은 「, · / ; |」 중 무엇으로 구분해도 된다. JSON 이면 리스트도 된다.
  정답(선택, 있어야 정확도가 나온다):
  정답_프로젝트(gold_project) 정답_고객사(gold_client) 정답_기술(gold_skills: 「표기=ID; 표기=ID」 또는 JSON {표기: ID})
  표준 목록에 없으면 NONE. 정답이 없으면 gold_id 칸이 비고 check_data.py 가 오류로 잡는다.
  변형(variant, 선택): 분석용 분류. 없으면 「실데이터」.

입력 2. 표준 목록 CSV — data/sample/standard.csv 와 같은 칸:
  entity_id entity_type(client|project|skill) name client_id period_start period_end group industry aliases(「|」 구분)

출력: <out>/mentions.csv (record_id 칸 추가 — 다른 코드는 무시함), <out>/standard.csv (검사 후 복사), 요약 출력.
분할(split): 사람ID(없으면 레코드ID)의 해시로 dev/test 반반(재실행해도 같다).
역할은 정규화 대상이 아니다(규격: project · client · skill). 역할은 프로젝트 판정의 문맥(ctx_role)으로만 들어간다.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ALIASES = {
    "name": ["사람ID", "person_id", "사번", "구성원ID", "이름", "name", "성명"],   # dev/test 분할 기준(선택)
    "project": ["프로젝트명", "프로젝트", "project", "project_name"],
    "client": ["고객사", "client", "customer"],
    "role": ["역할", "role"],
    "skills": ["기술", "skills", "skill", "기술스택"],
    "period": ["기간", "period"],
    "start_date": ["시작일자", "시작일", "start_date", "start"],
    "end_date": ["종료일자", "종료일", "end_date", "end"],
    "gold_project": ["정답_프로젝트", "gold_project"],
    "gold_client": ["정답_고객사", "gold_client"],
    "gold_skills": ["정답_기술", "gold_skills"],
    "variant": ["변형", "variant"],
    "record_id": ["record_id", "레코드ID", "id"],
}
STD_COLS = ["entity_id", "entity_type", "name", "client_id", "period_start", "period_end", "group", "industry", "aliases"]
MENTION_COLS = ["mention_id", "entity_type", "surface", "gold_id", "variant", "missing_client",
                "ctx_client", "ctx_period", "ctx_role", "split", "record_id"]
_YM = re.compile(r"(\d{4})\s*[.\-/년]\s*(\d{1,2})|(?<![\d.])(\d{2})\s*[./]\s*(\d{1,2})(?!\d)")
_ONGOING = re.compile(r"현재|재직|진행|now|present", re.I)


def read_records(path):
    p = Path(path)
    if p.suffix.lower() == ".json":
        rows = json.loads(p.read_text(encoding="utf-8"))
    else:
        with open(p, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows):
        g = {}
        for key, names in ALIASES.items():
            v = next((r[n] for n in names if n in r and r[n] not in (None, "")), None)
            g[key] = v.strip() if isinstance(v, str) else v
        g["record_id"] = g["record_id"] or f"R{i + 1:04d}"
        out.append(g)
    return out


def split_skills(v):
    if not v:
        return []
    if isinstance(v, list):
        items = v
    else:
        items = re.split(r"\s*[,·/;|\n]\s*", v)
    seen, out = set(), []
    for s in (x.strip() for x in items):
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def parse_gold_skills(v):
    if not v:
        return {}
    if isinstance(v, dict):
        return {k.strip(): str(x).strip() for k, x in v.items()}
    out = {}
    for part in re.split(r"\s*;\s*", v):
        if "=" in part:
            k, x = part.rsplit("=", 1)
            out[k.strip()] = x.strip()
    return out


def _ym(text):
    m = _YM.search(text or "")
    if not m:
        return None
    y, mo = (m.group(1), m.group(2)) if m.group(1) else ("20" + m.group(3), m.group(4))
    return f"{int(y)}.{int(mo):02d}" if 1 <= int(mo) <= 12 else None


def _months(text):
    """「1년 3개월」 「15개월」 → 개월 수. 4자리 연도가 있으면 기간 길이로 보지 않는다."""
    if not text or re.search(r"\d{4}", text):
        return None
    y = re.search(r"(\d{1,2})\s*년", text)
    m = re.search(r"(\d{1,3})\s*개월", text)
    return ((int(y.group(1)) * 12 if y else 0) + (int(m.group(1)) if m else 0)) or None


def period_text(r):
    """ctx_period 문자열. common.parse_period 가 읽을 수 있게 「YYYY.MM ~ YYYY.MM」 로 맞춘다.
    진행 중이면 「YYYY.MM ~ 현재」(겹침 계산은 「알 수 없음」 이 된다 — 날짜를 지어내지 않는다).
    못 읽으면 원문 그대로."""
    s, e = _ym(r["start_date"]), _ym(r["end_date"])
    raw = r["period"] or ""
    if raw and (not s or not e):
        found = [x for x in (_ym(t) for t in re.split(r"~|–|—|\s-\s|부터|까지", raw)) if x]
        s = s or (found[0] if found else None)
        if not e and len(found) > 1:
            e = found[1]
    ongoing = bool(_ONGOING.search(f"{raw} {r['end_date'] or ''}"))
    dur = _months(raw)
    if s and not e and not ongoing and dur:             # 「시작일 + 15개월」 → 종료 월 계산(코드가 한다, 모델에 시키지 않음)
        y, mo = map(int, s.split("."))
        k = y * 12 + (mo - 1) + dur - 1
        e = f"{k // 12}.{k % 12 + 1:02d}"
    if s and e:
        return f"{s} ~ {e}"
    if s and ongoing:
        return f"{s} ~ 현재"
    return raw or (" ~ ".join(x for x in (r["start_date"], r["end_date"]) if x))


def split_of(name, record_id):
    key = name or record_id
    return "dev" if int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "test"


def check_standard(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    problems = []
    missing_cols = [c for c in STD_COLS if c not in (rows[0].keys() if rows else [])]
    if missing_cols:
        problems.append(f"표준 목록에 칸이 없다: {missing_cols}")
        return rows, problems
    ids = [r["entity_id"] for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        problems.append(f"entity_id 중복: {dup[:5]}")
    bad_type = [r["entity_id"] for r in rows if r["entity_type"] not in ("client", "project", "skill")]
    if bad_type:
        problems.append(f"entity_type 이 client/project/skill 이 아니다: {bad_type[:5]}")
    clients = {r["entity_id"] for r in rows if r["entity_type"] == "client"}
    orphan = [r["entity_id"] for r in rows if r["entity_type"] == "project" and r["client_id"] and r["client_id"] not in clients]
    if orphan:
        problems.append(f"project 의 client_id 가 표준 목록에 없다: {orphan[:5]}")
    if "NONE" in ids:
        problems.append("entity_id 로 NONE 은 쓸 수 없다(「목록에 없음」 선택지 이름)")
    return rows, problems


def build(records, std_ids):
    mentions, warn = [], []
    n = 0

    def add(rec, etype, surface, gold, ctx=None):
        nonlocal n
        n += 1
        if gold and gold != "NONE" and gold not in std_ids:
            warn.append(f"{rec['record_id']} {etype} '{surface}' 정답 {gold} 이 표준 목록에 없음")
        ctx = ctx or {}
        mentions.append({
            "mention_id": f"M{n:05d}", "entity_type": etype, "surface": surface, "gold_id": gold or "",
            "variant": rec["variant"] or "실데이터", "missing_client": "0" if rec["client"] else "1",
            "ctx_client": ctx.get("client", ""), "ctx_period": ctx.get("period", ""), "ctx_role": ctx.get("role", ""),
            "split": split_of(rec["name"], rec["record_id"]), "record_id": rec["record_id"]})

    for r in records:
        if r["client"]:
            add(r, "client", r["client"], r["gold_client"])
        if r["project"]:
            add(r, "project", r["project"], r["gold_project"],
                {"client": r["client"] or "", "period": period_text(r), "role": r["role"] or ""})
        gs = parse_gold_skills(r["gold_skills"])
        for s in split_skills(r["skills"]):
            add(r, "skill", s, gs.get(s))
    return mentions, warn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--standard", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    std_rows, problems = check_standard(a.standard)
    if problems:
        print("표준 목록 문제 — 고친 뒤 다시 실행:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    records = read_records(a.records)
    mentions, warn = build(records, {r["entity_id"] for r in std_rows})

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "standard.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=STD_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(std_rows)
    with open(out / "mentions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MENTION_COLS)
        w.writeheader()
        w.writerows(mentions)

    by_type = {t: sum(m["entity_type"] == t for m in mentions) for t in ("project", "client", "skill")}
    n_gold = sum(bool(m["gold_id"]) for m in mentions)
    n_choices = {t: sum(r["entity_type"] == t for r in std_rows) + 1 for t in ("project", "client", "skill")}
    print(f"레코드 {len(records)}건 → 표기 {len(mentions)}건 {by_type}  (dev {sum(m['split'] == 'dev' for m in mentions)} / test {sum(m['split'] == 'test' for m in mentions)})")
    print(f"결측: 고객사 없는 레코드 {sum(not r['client'] for r in records)}, 프로젝트명 없는 레코드 {sum(not r['project'] for r in records)}, 기술 없는 레코드 {sum(not r['skills'] for r in records)}")
    print(f"정답 있는 표기 {n_gold}/{len(mentions)}" + ("" if n_gold == len(mentions) else "  ← 정답이 빈 표기가 있다. check_data.py 가 오류로 잡으니 정답 칸을 채울 것(목록에 없으면 NONE)"))
    print(f"k=all 선택지 수(NONE 포함): {n_choices}" + ("  ← 20 초과: LLM 은 --over20 logit_bias 필요" if max(n_choices.values()) > 20 else ""))
    for x in warn[:10]:
        print(" 경고:", x)
    if len(warn) > 10:
        print(f" 경고 {len(warn) - 10}건 더")
    print(f"→ {out}/mentions.csv, {out}/standard.csv")


if __name__ == "__main__":
    main()
