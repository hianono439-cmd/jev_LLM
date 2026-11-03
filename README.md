# jev_LLM — LLM 정규화 (Jev 비교용)

엔티티 정규화(프로필 표기 → 표준 목록 ID 또는 NONE)를 LLM으로 풀어, 이헌준의 Jev 정규화와 **같은 입력·같은 채점기**로 비교한다.
규격은 [`LLM_HANDOFF.md`](LLM_HANDOFF.md)(이헌준 작성)를 따른다.

| 파일 | 내용 |
|---|---|
| `llm_normalize.py` | LLM 정규화 실행기. logprobs로 선택지 확률을 읽고 규격 jsonl을 `out/`에 쓴다 |
| `LLM_HANDOFF.md` | Jev 쪽이 정한 규격 (수정하지 않음) |
| `LLM_TO_JEV_NOTES.md` | LLM 쪽 → Jev 쪽 합의 요청 사항 (20개 초과 선택지 처리 등) |
| `common.py` | Jev 쪽과 LLM 쪽 공통: 데이터 읽기, 후보 만들기(`build_candidates`), 기간 해석과 겹침 계산 (이헌준) |
| `jev_normalize.py` | Jev 정규화 실행기. 한 요청에 정순 Choice, 역순 Choice, 「후보 중 같은 것이 있나」 Noul (이헌준) |
| `score.py` | 채점기, Jev와 LLM 공통 (이헌준) |
| `check_parity.py` | 두 결과가 같은 조건(mention, 후보 순서, 선택지, 정답, 분할)인지 검사 (이헌준) |
| `gen_sample.py`, `data/sample/` | 가상 정규화 데이터와 그 생성기. 승현 데이터가 오기 전 임시 (이헌준) |
| `results/` | Jev 실행 결과 jsonl. LLM 결과와 비교할 때 쓴다 (`out/` 은 git 에서 제외라 따로 둔다) |
| `RESULTS_1103.md` | Jev 1차 결과 요약 |
| `prepare_real.py` | 실제 추출 결과 + 팀 표준 목록 → `mentions.csv` · `standard.csv` 변환 |
| `JEV_TO_LLM_REPLY.md` | `LLM_TO_JEV_NOTES.md` 에 대한 답 |

## 실행

`common.py`, `jev_normalize.py`, `score.py`, `check_parity.py`, `data/` 는 이 저장소에 함께 있다. LLM 쪽은 이 파일들을 수정하지 않고 import 해서 쓴다.

```bash
pip install openai tiktoken
cp .env.example .env          # OPENAI_API_KEY 채우기 (커밋 금지)
python3 llm_normalize.py --probe                                  # 모델이 logprobs/top_logprobs=20 을 주는지 확인
python3 llm_normalize.py --data data/sample --k all --limit 20 --rev
python3 llm_normalize.py --data data/sample --k all --rev
python3 llm_normalize.py --data data/sample --k 5 --rev
python3 check_parity.py out/jev_kall_*.jsonl out/llm_kall_*.jsonl
python3 score.py out/jev_kall_*.jsonl out/llm_kall_*.jsonl --price-in <$/1M> --price-out <$/1M>
```

Jev 쪽 (`.env` 의 `TYPESAFE_API_KEY`, 표준 라이브러리만 사용)

```bash
python3 jev_normalize.py --mock                          # 키 없이 파이프라인만 (가짜 확률)
python3 jev_normalize.py --show-request                  # Jev 에 보내는 요청 본문 한 건
python3 jev_normalize.py --limit 20                      # 실제 호출 20건
python3 jev_normalize.py                                 # 전체, 후보 = 그 유형 표준 목록 전체
python3 jev_normalize.py --k 5                           # 후보를 유사도 상위 5개로
python3 jev_normalize.py --questions fwd --workers 1     # LLM 과 같은 기준으로 비용과 지연을 잴 때
```

LLM 결과와 비교할 Jev 결과는 `results/` 에 있다. 질문 구성이 같은 파일끼리 비교한다 (LLM `--rev` 없음이면 `results/jev_kall_fwd_*.jsonl`, `--rev` 면 `results/jev_kall_fwd+rev_*.jsonl`).

## 실제 데이터가 오면

추출 결과가 이력 한 줄 형식이면 먼저 아래 `prepare_real.py` 로 바꾸고, 이미 아래 형식이면 그대로 새 폴더(예: `data/real/`)에 둔다. 코드는 고치지 않는다.

| 파일 | 필수 칸 | 선택 칸 |
|---|---|---|
| `mentions.csv` | `mention_id` `entity_type`(project/client/skill) `surface` `gold_id`(표준 목록에 없으면 `NONE`) `split`(dev/test) `ctx_client` `ctx_period` `ctx_role` (project 외에는 빈칸) | `variant` `missing_client` (없으면 채점표의 유형별 · 결측별 칸이 빈다) |
| `standard.csv` | `entity_id` `entity_type` `name` `client_id` `period_start` `period_end` `group` `industry` (해당 없는 칸은 빈칸) | `aliases` (`\|` 로 구분) |

순서 — 1단계는 API 호출이 없다. 2단계부터 키를 쓴다.

```bash
python3 check_data.py data/real --price-in <$/1M> --price-out <$/1M>   # 1. 형식 검사 + 선택지 수 + LLM 호출 수 · 토큰 · 비용 추정
python3 jev_normalize.py --data data/real --questions fwd,rev --workers 1   # 2. Jev
python3 llm_normalize.py --data data/real --k all --limit 20 --rev          # 3. LLM 소량 확인
python3 llm_normalize.py --data data/real --k all --rev                     # 4. LLM 전체 (선택지 20개 초과가 있으면 --over20 logit_bias)
python3 check_parity.py out/jev_kall_*.jsonl out/llm_kall_*.jsonl           # 5. 같은 조건인지
python3 score.py out/jev_kall_*.jsonl out/llm_kall_*.jsonl --price-in <$/1M> --price-out <$/1M>   # 6. 비교표
```

`check_data.py` 가 알려 주는 것: 필수 칸 누락, `gold_id` 가 표준 목록에 없음, `split` · `entity_type` 값 오류, ID 중복, 기간을 못 읽는 행,
유형별 최대 선택지 수(20개 초과면 LLM 은 `--over20 logit_bias` 필요, 26개 초과면 숫자 라벨), `k=5` 일 때 정답이 후보에 드는 비율.
토큰 추정은 프롬프트를 실제로 만들어 세므로, 가상 데이터 20건 실측(입력 39,860)과 0.4% 안에서 맞았다.

### 이력 한 줄 형식을 바꾸기 (`prepare_real.py`)

추출 결과(이력 한 건 = 한 줄, 프로젝트명 · 고객사 · 역할 · 기술 · 기간 · 시작일자 · 종료일자)와 팀 표준 목록을
비교 코드가 읽는 형식(`mentions.csv` · `standard.csv`)으로 바꾼다. 결측 있는 세트와 없는 세트는 따로 변환해 폴더를 나눈다.

```bash
python3 prepare_real.py --records 추출_결측없음.csv --standard 표준목록.csv --out data/real_full
python3 prepare_real.py --records 추출_결측있음.csv --standard 표준목록.csv --out data/real_missing
python3 jev_normalize.py --data data/real_full --questions fwd,rev --workers 1
python3 llm_normalize.py --data data/real_full --k all --rev
```

- 칸 이름은 한글/영문 모두 받는다. 기술은 `,` `·` `/` `;` `|` 아무거나로 구분. 빈 칸 = 결측(결측 항목은 판정 호출 자체가 없다)
- 정확도를 보려면 정답 칸 `정답_프로젝트` · `정답_고객사` · `정답_기술`(「표기=ID; 표기=ID」, 목록에 없으면 `NONE`)을 채운다. 비어 있으면 `check_data.py` 가 오류로 잡는다
- 정규화 대상은 프로젝트 · 고객사 · 기술. 역할 · 기간은 프로젝트 판정 문맥으로만 쓴다. 사람을 가리키는 칸(`사람ID` 등)이 있으면 dev/test 분할에만 쓰고, 없어도 된다
- 기간은 「YYYY.MM ~ YYYY.MM」 로 맞춰 넣고(시작일+「15개월」 도 계산), 진행 중이면 「~ 현재」. 날짜를 지어내지 않는다
- 실행 끝에 선택지 수를 알려 준다. 20개(NONE 포함)를 넘으면 LLM 은 `--over20 logit_bias` 가 필요하다
- 표준 목록 형식은 `data/sample/standard.csv` 와 같다. 칸 누락 · ID 중복 · 없는 client_id 를 먼저 검사한다

## 상태

- 모델 `gpt-4.1-mini` 확정 (`--probe`: temperature 0 · logprobs · top 20 지원. `gpt-6-luna` 는 temperature 0 거부)
- 가상 데이터 20건 소량 실행: 오류 0, 정답 20/20. 전체 실행 전
- 선택지 20개 초과 시 기본은 error, `--over20 logit_bias` 방식은 이헌준과 합의 후 사용
