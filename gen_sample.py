"""승현 데이터가 오기 전까지 쓸 가상 정규화 세트를 만든다.

표준 목록(standard.csv)에서 표기를 일부러 바꿔 mention 을 만들기 때문에 정답(gold_id)이 저절로 붙는다.
표준 목록에 없는 실체(gold_id=NONE)와 고객사가 빠진 이력(결측 세트)도 섞는다.

    python3 gen_sample.py            # data/sample/ 에 standard.csv · mentions.csv
"""
import csv
import random
from pathlib import Path

SEED = 42
OUT = Path(__file__).parent / "data" / "sample"

# 가상 회사들. 그룹사·계열사가 섞여 있어야 「같은 고객사인가」 가 어려워진다
CLIENTS = [
    # id, 이름, 그룹, 업종, 영문, 짧은 이름
    ("C01", "한빛은행", "한빛금융", "은행", "Hanbit Bank", "한빛"),
    ("C02", "한빛캐피탈", "한빛금융", "캐피탈", "Hanbit Capital", "한빛캐피"),
    ("C03", "누리생명", "누리그룹", "보험", "Nuri Life", "누리"),
    ("C04", "누리손해보험", "누리그룹", "보험", "Nuri Insurance", "누리손보"),
    ("C05", "가람카드", "가람", "카드", "Garam Card", "가람"),
    ("C06", "새벽통신", "새벽", "통신", "Saebyeok Telecom", "새벽"),
    ("C07", "푸른전자", "푸른", "제조", "Pureun Electronics", "푸른"),
    ("C08", "다온유통", "다온", "유통", "Daon Retail", "다온"),
    ("C09", "미르에너지", "미르", "에너지", "Mir Energy", "미르"),
    ("C10", "소담병원", "소담", "의료", "Sodam Hospital", "소담"),
]

PROJECTS = [
    # id, 고객사, 이름, 시작, 끝
    ("P01", "C01", "한빛은행 차세대 시스템 구축", "2022-09", "2024-06"),
    ("P02", "C01", "한빛은행 차세대 2단계 고도화", "2024-07", "2025-12"),
    ("P03", "C01", "한빛은행 모바일뱅킹 앱 리뉴얼", "2023-03", "2023-12"),
    ("P04", "C02", "한빛캐피탈 차세대 여신 시스템 구축", "2023-01", "2024-03"),
    ("P05", "C03", "누리생명 보험금 청구 자동화", "2022-05", "2023-02"),
    ("P06", "C04", "누리손해보험 보상 시스템 클라우드 전환", "2023-06", "2024-09"),
    ("P07", "C05", "가람카드 데이터 플랫폼 구축", "2021-11", "2022-12"),
    ("P08", "C05", "가람카드 이상거래 탐지 고도화", "2023-02", "2023-11"),
    ("P09", "C06", "새벽통신 고객센터 AI 상담 도입", "2024-01", "2024-10"),
    ("P10", "C06", "새벽통신 빌링 시스템 MSA 전환", "2022-03", "2023-08"),
    ("P11", "C07", "푸른전자 스마트팩토리 MES 구축", "2021-06", "2022-09"),
    ("P12", "C07", "푸른전자 설비 예지보전 플랫폼", "2023-04", "2024-05"),
    ("P13", "C08", "다온유통 통합 물류 시스템 구축", "2022-01", "2023-06"),
    ("P14", "C08", "다온유통 추천 엔진 고도화", "2024-02", "2024-12"),
    ("P15", "C09", "미르에너지 ERP 차세대 전환", "2023-01", "2024-12"),
    ("P16", "C10", "소담병원 EMR 클라우드 전환", "2024-03", "2025-06"),
]

SKILLS = [
    # id, 이름, 변형들
    ("S01", "Java", ["java", "JAVA", "자바", "Java 17"]),
    ("S02", "Spring Boot", ["springboot", "스프링부트", "Spring boot", "SpringBoot 3"]),
    ("S03", "Spring Framework", ["Spring", "스프링", "spring framework"]),
    ("S04", "Python", ["python", "파이썬", "Python3"]),
    ("S05", "Kubernetes", ["k8s", "쿠버네티스", "kubernetes", "K8S"]),
    ("S06", "Docker", ["docker", "도커"]),
    ("S07", "Oracle Database", ["Oracle", "오라클", "Oracle DB", "oracle 19c"]),
    ("S08", "PostgreSQL", ["Postgres", "postgresql", "포스트그레스"]),
    ("S09", "React", ["React.js", "리액트", "ReactJS"]),
    ("S10", "Vue.js", ["Vue", "뷰", "VueJS", "vue3"]),
    ("S11", "Apache Kafka", ["Kafka", "카프카", "kafka"]),
    ("S12", "AWS", ["Amazon Web Services", "아마존 웹 서비스", "aws"]),
    ("S13", "PyTorch", ["pytorch", "파이토치", "Torch"]),
    ("S14", "LangChain", ["langchain", "랭체인", "Lang Chain"]),
    ("S15", "Elasticsearch", ["ElasticSearch", "엘라스틱서치", "ES"]),
]

# 표준 목록에 없는 실체 — 정답은 NONE (헷갈리게 기존 이름과 비슷한 것도 넣는다)
NONE_MENTIONS = [
    ("project", "한빛은행 퇴직연금 시스템 구축", "C01"),
    ("project", "가람카드 포인트 플랫폼 개편", "C05"),
    ("project", "하늘항공 예약 시스템 구축", ""),
    ("client", "한빛증권", ""),
    ("client", "하늘항공", ""),
    ("skill", "Rust", ""),
    ("skill", "Spring Batch", ""),
]

PROJECT_SUFFIX = {"구축": ["PJT", "프로젝트", "구축 사업", "구축 PJT"], "고도화": ["고도화 PJT", "고도화 사업"],
                  "전환": ["전환 PJT", "전환 프로젝트", "전환 사업"]}
WORD_EN = {"차세대": "Next-Gen", "시스템": "System", "플랫폼": "Platform", "클라우드": "Cloud", "모바일뱅킹": "Mobile Banking"}
ROLES = ["PM", "PL", "아키텍트", "백엔드 개발", "프론트엔드 개발", "데이터 엔지니어", "QA", "DBA"]


def drop_char(s, rng):
    """오타 — 공백이 아닌 글자 하나를 뺀다."""
    idx = [i for i, ch in enumerate(s) if ch != " "]
    i = rng.choice(idx[1:]) if len(idx) > 2 else idx[0]
    return s[:i] + s[i + 1:]


def project_variants(name, client_short, rng):
    out = []
    for k, subs in PROJECT_SUFFIX.items():
        if name.endswith(k):
            out.append(("접미사", name[: -len(k)] + rng.choice(subs)))
    out.append(("띄어쓰기", name.replace(" ", "")))
    first = name.split(" ")[0]
    out.append(("고객사 줄임", name.replace(first, client_short, 1)))
    en = name
    for ko, e in WORD_EN.items():
        en = en.replace(ko, e)
    if en != name:
        out.append(("한영 혼용", en))
    if "2단계" in name:
        out.append(("단계 표기", name.replace("2단계", rng.choice(["Phase2", "2차", "2nd"]))))
    out.append(("오타", drop_char(name, rng)))
    rest = name.split(" ", 1)[1] if " " in name else name
    out.append(("고객사 생략", rest))  # 고객사가 빠지면 같은 이름의 다른 고객사 사업과 헷갈린다
    return out


def client_variants(name, en, short, rng):
    return [("법인 표기", rng.choice([f"(주){name}", f"{name}(주)", f"주식회사 {name}"])),
            ("영문", en), ("줄임", short), ("띄어쓰기", name[:2] + " " + name[2:])]


def period_text(start, end, rng):
    y1, m1 = start.split("-")
    y2, m2 = end.split("-")
    return rng.choice([f"{start}~{end}", f"{y1}.{int(m1)}~{y2}.{int(m2)}", f"{y1[2:]}.{m1}~{y2[2:]}.{m2}",
                       f"{y1}년 {int(m1)}월 ~ {y2}년 {int(m2)}월"])


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    client = {c[0]: c for c in CLIENTS}

    with open(OUT / "standard.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["entity_id", "entity_type", "name", "client_id", "period_start", "period_end", "group", "industry", "aliases"])
        for cid, name, group, ind, _, _ in CLIENTS:
            w.writerow([cid, "client", name, "", "", "", group, ind, ""])
        for pid, cid, name, s, e in PROJECTS:
            w.writerow([pid, "project", name, cid, s, e, "", "", ""])
        for sid, name, _ in SKILLS:
            w.writerow([sid, "skill", name, "", "", "", "", "", ""])

    rows = []
    for pid, cid, name, s, e in PROJECTS:
        for variant, surface in project_variants(name, client[cid][5], rng):
            for missing in (False, True):  # 결측 세트 — 같은 표기를 고객사 없이도 한 번
                rows.append(dict(entity_type="project", surface=surface, gold_id=pid, variant=variant,
                                 ctx_client="" if missing else client[cid][1], ctx_period=period_text(s, e, rng),
                                 ctx_role=rng.choice(ROLES), missing_client=int(missing)))
    for cid, name, _, _, en, short in CLIENTS:
        for variant, surface in client_variants(name, en, short, rng):
            rows.append(dict(entity_type="client", surface=surface, gold_id=cid, variant=variant,
                             ctx_client="", ctx_period="", ctx_role="", missing_client=0))
    for sid, _, variants in SKILLS:
        for surface in variants:
            rows.append(dict(entity_type="skill", surface=surface, gold_id=sid, variant="표기 변형",
                             ctx_client="", ctx_period="", ctx_role="", missing_client=0))
    for etype, surface, cid in NONE_MENTIONS:
        for missing in ((False, True) if etype == "project" and cid else (False,)):
            rows.append(dict(entity_type=etype, surface=surface, gold_id="NONE", variant="표준 목록에 없음",
                             ctx_client="" if missing or not cid else client[cid][1],
                             ctx_period=period_text("2023-01", "2023-12", rng) if etype == "project" else "",
                             ctx_role=rng.choice(ROLES) if etype == "project" else "", missing_client=int(missing)))

    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["mention_id"] = f"M{i + 1:04d}"
        r["split"] = "dev" if i % 2 == 0 else "test"  # 문턱값은 dev 에서만 고른다
    cols = ["mention_id", "entity_type", "surface", "gold_id", "variant", "missing_client",
            "ctx_client", "ctx_period", "ctx_role", "split"]
    with open(OUT / "mentions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by[r["entity_type"]] = by.get(r["entity_type"], 0) + 1
    print(f"standard {len(CLIENTS) + len(PROJECTS) + len(SKILLS)} · mentions {len(rows)} {by} → {OUT}")


if __name__ == "__main__":
    main()
