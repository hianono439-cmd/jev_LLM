"""Jev 쪽 · LLM 쪽이 같이 쓰는 부분 — 데이터 읽기, 후보 만들기, 기간 계산.

비교가 성립하려면 LLM 쪽과 Jev 쪽이 이 파일의 build_candidates() 결과를 그대로 써야 한다.
같은 후보 · 같은 순서 · 같은 맥락이 아니면 두 숫자를 나란히 놓을 수 없다.
"""
import csv
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).parent
NONE_ID = "NONE"


def load_env(path=ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_data(data_dir):
    data_dir = Path(data_dir)
    standard = read_csv(data_dir / "standard.csv")
    mentions = read_csv(data_dir / "mentions.csv")
    return standard, mentions


# ---------------------------------------------------------------- 기간

_YM = re.compile(r"(\d{2,4})\s*[.\-/년]\s*(\d{1,2})")


def parse_period(text):
    """「2023.3~2024.6」 「23.03~24.06」 「2023년 3월 ~ 2024년 6월」 → ((2023,3),(2024,6)). 못 읽으면 None.
    날짜 비교는 코드가 한다 — Jev 공식 약점(「날짜를 양이 아니라 글자로 읽는다」)."""
    found = _YM.findall(text or "")
    if len(found) < 2:
        return None
    out = []
    for y, m in found[:2]:
        y = int(y)
        y = y + 2000 if y < 100 else y
        out.append((y, int(m)))
    return tuple(out)


def overlaps(a, b):
    """두 기간 ((y,m),(y,m)) 이 하루라도 겹치나."""
    return not (a[1] < b[0] or b[1] < a[0])


# ---------------------------------------------------------------- 후보

def _bigrams(s):
    s = re.sub(r"\s+", "", s.lower())
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def similarity(a, b):
    """문자 2-gram 자카드와 difflib 비율의 평균. 임베딩 없이 돌아가는 기준선."""
    ba, bb = _bigrams(a), _bigrams(b)
    jac = len(ba & bb) / len(ba | bb)
    return (jac + SequenceMatcher(None, a.lower(), b.lower()).ratio()) / 2


def build_candidates(mention, standard, k=None):
    """같은 유형의 표준 실체를 후보로. k=None 이면 그 유형 전체(공식 문서 권고 — 255개까지는 목록을 다 준다),
    k 가 있으면 유사도 상위 k 개. 반환 순서 = 유사도 내림차순(첫 선택지 편향을 재려고 순서를 보존한다)."""
    pool = [s for s in standard if s["entity_type"] == mention["entity_type"]]
    by_id = {s["entity_id"]: s for s in standard}
    scored = []
    for s in pool:
        names = [s["name"]] + [a for a in s.get("aliases", "").split("|") if a]
        scored.append((max(similarity(mention["surface"], n) for n in names), s))
    scored.sort(key=lambda x: -x[0])
    if k:
        scored = scored[:k]
    mp = parse_period(mention.get("ctx_period", ""))
    cands = []
    for sim, s in scored:
        c = {"id": s["entity_id"], "name": s["name"], "sim": round(sim, 3)}
        if s["entity_type"] == "project":
            c["client"] = by_id[s["client_id"]]["name"] if s["client_id"] in by_id else ""
            c["period"] = f'{s["period_start"]}~{s["period_end"]}'
            sp = parse_period(c["period"])
            c["overlap"] = ("예" if overlaps(mp, sp) else "아니오") if mp and sp else "알 수 없음"
        if s["entity_type"] == "client":
            c["group"], c["industry"] = s["group"], s["industry"]
        if s.get("aliases"):
            c["aliases"] = [a for a in s["aliases"].split("|") if a]
        cands.append(c)
    return cands


def gold_in_candidates(mention, cands):
    return mention["gold_id"] == NONE_ID or any(c["id"] == mention["gold_id"] for c in cands)
