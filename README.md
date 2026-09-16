# G-LIFT MVP v0.5.0

전역-coupled 제약최적화 기반 국방 육로수송 AI 배차 시스템 MVP입니다.

이 버전은 다음을 지원합니다.

1. 엑셀 원본 8개를 표준 CSV로 전처리
2. 기존 표준 CSV를 바탕으로 미지원 발생 부족 시나리오 자동 생성
3. OR-Tools CP-SAT Optimizer로 baseline 및 shortage scenario별 후보 배차표 5개 생성
4. 미지원 사유표, 하드 제약 검증표, 시나리오 요약표 생성
5. 후보 배차표 feature 추출
6. 배차반장 베스트표와 후보 배차표 유사도 계산
7. 규칙 기반 Ranker와 Tree-based Ranker 1차 실증
8. 간이 역최적화 방식의 목적함수 가중치 보정표 생성
9. Streamlit에서 Optimizer와 Ranker/역최적화 결과 확인

> 주의: Ranker/역최적화는 현재 소규모 MVP 데이터 기반의 “배차반장 암묵지 학습 실증 개념”입니다. 생산급 학습 성능으로 과장하지 않습니다.

## 1. 설치

프로젝트 루트에서 가상환경을 활성화한 뒤 라이브러리를 설치합니다.

```bash
python -m pip install -r requirements.txt
```

PowerShell 실행정책 오류가 나면 현재 터미널에서만 임시 허용합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. 전처리 실행

```bash
python -m src.preprocess
```

생성 결과:

- `data/processed/requests.csv`
- `data/processed/vehicles.csv`
- `data/processed/drivers.csv`
- `data/processed/vehicle_availability.csv`
- `data/processed/driver_availability.csv`
- `data/processed/expert_schedule.csv`
- `reports/validation/validation_report.csv`
- `reports/validation/validation_report.xlsx`

## 3. 부족 시나리오 자동 생성

본선 시연에서는 미지원 사유표가 실제로 채워지는 장면이 필요합니다. 아래 명령은 원본 데이터를 훼손하지 않고 `data/scenarios/` 아래에 가상 부족 시나리오를 생성합니다.

```bash
python -m src.scenario_generator
```

생성되는 시나리오:

- `baseline`
- `vehicle_shortage`
- `driver_shortage`
- `skill_shortage`
- `other_vehicle_conflict`
- `mixed_shortage`

## 4. Optimizer 후보 배차표 5개 생성

기본 baseline만 실행:

```bash
python -m src.candidate_generator
```

특정 시나리오 실행:

```bash
python -m src.candidate_generator --scenario mixed_shortage
```

전체 시나리오 실행:

```bash
python -m src.candidate_generator --all-scenarios
```

각 시나리오 결과는 다음 위치에 저장됩니다.

```text
outputs/candidates/<scenario_name>/
```

각 폴더에 생성되는 파일:

- `candidate_01_min_unserved.xlsx`
- `candidate_02_fairness.xlsx`
- `candidate_03_vehicle_preservation.xlsx`
- `candidate_04_fatigue.xlsx`
- `candidate_05_expert_style.xlsx`
- `score_breakdown.xlsx`
- `unserved_reasons.xlsx`
- `hard_constraint_check.xlsx`
- `scenario_summary.xlsx`

## 5. Ranker/역최적화 1차 MVP 실행

전체 시나리오 후보 배차표를 대상으로 실행:

```bash
python -m src.ranker --scenario all
```

특정 시나리오만 실행:

```bash
python -m src.ranker --scenario baseline
```

RandomForest 기반 Ranker를 사용하고 싶으면:

```bash
python -m src.ranker --scenario all --model random_forest
```

생성되는 파일:

```text
outputs/ranker/all_scenarios/candidate_feature_matrix.xlsx
outputs/ranker/all_scenarios/before_after_ranking_comparison.xlsx
outputs/ranker/all_scenarios/learned_objective_weights.xlsx
outputs/ranker/all_scenarios/ranker_training_report.xlsx
outputs/ranker/all_scenarios/ranker_feature_importance.xlsx
reports/evaluation/ranker_training_report.xlsx
```

## 6. 테스트

표준 `unittest` 기반 단위 테스트가 `tests/` 폴더에 있습니다. 프로젝트 루트에서 실행합니다.

```bash
python -m unittest discover -s tests
```

`pytest`가 설치돼 있으면 아래도 동일하게 동작합니다.

```bash
python -m pytest tests
```

테스트는 대부분 외부 데이터 파일 없이 인라인 데이터로 자체 완결되도록 작성되어, 전처리 이전에도 바로 실행할 수 있습니다. 커버리지 범위:

- `tests/test_utils.py`: 시간/차량번호/차종/기량코드 파싱·정규화 헬퍼
- `tests/test_other_handler.py`: `Other` 직접운전 판별과 driver_mode 컬럼 생성
- `tests/test_feature_extractor.py`: 슬롯 계산(끝 시간 배타), 기량 판정, 차종 매칭, 가용성 인덱스, fit 점수
- `tests/test_hard_constraint_checker.py`: 독립 하드 제약 검증기(차량 시간 중복, `Other` 오배정, 중복 행 등)
- `tests/test_ranker_evaluation.py`: DCG/NDCG@5, Top-1 일치율
- `tests/test_ranker.py`: 규칙 기반 Ranker, Tree-based Ranker, **K-fold 교차검증 out-of-sample 지표**
- `tests/test_inverse_optimization.py`: 정규화, 가중치 합=1, 역최적화 산출표
- `tests/test_config_weights.py`: 정본 가중치 테이블(`BASELINE_FEATURE_WEIGHTS`)이 규칙 Ranker·역최적화에서 **단일 출처**로 공유되는지 회귀 검증
- `tests/test_optimizer.py`: 프로파일 구조 및 solve 결과의 하드 위반 0 검증

> 참고: `tests/test_optimizer.py`는 OR-Tools(그리고 solve 통합 테스트는 `data/processed/` 입력)를 필요로 하며, 없는 환경에서는 해당 테스트만 자동으로 건너뜁니다(`skipUnless`). 나머지 테스트는 그대로 실행됩니다.

## 7. Streamlit 실행

```bash
streamlit run app.py
```

또는

```bash
python -m streamlit run app.py
```

화면은 배차 담당 실무자가 별도 지식 없이 쓸 수 있도록 **원클릭 가이드형**으로 구성되어 있습니다. 내부 파이프라인(전처리·Optimizer·Ranker·역최적화)은 감춰지고, 세 걸음으로 끝납니다.

1. **오늘 상황 선택** — 평상시 / 차량 부족 / 운전병 부족 / 자격 부족 / 외부 차량 충돌 / 복합 상황 중 하나를 고릅니다.
2. **AI 배차 계획 만들기** — 버튼 한 번으로 후보 배차안 5개 생성과 추천 순위 계산까지 자동 실행됩니다.
3. **결과 확인** — 전체/배차완료/미지원/규정위반 요약과 함께, 관점이 다른 5개 배차안을 쉬운 이름(최대 배차형·공정 배분형·차량 보호형·피로 최소형·배차반장 스타일형)의 탭으로 비교합니다. ⭐ 표시가 AI 추천안이며, 배차표는 한글 컬럼으로 표시됩니다.

내부 단계 제어(원본 데이터 불러오기=전처리, 부족 상황 데이터 재생성)와 전문가용 상세(학습 전/후 순위 비교, 역최적화 가중치, feature importance, 파일 경로)는 화면 맨 아래 **"🔧 전문가 상세 · 데이터 관리"** 를 펼치면 볼 수 있습니다.

> 참고: `data/processed/` 표준 CSV가 이미 있으면 전처리 없이 바로 2·3단계를 실행할 수 있습니다. 원본 엑셀이 DRM 등으로 잠겨 있어 전처리가 실패하는 경우, 잠금을 해제한 파일로 교체한 뒤 '전문가 상세'의 '원본 데이터 불러오기'를 실행하세요.

## 8. 핵심 규칙

- `Other`는 별도 직접운전자를 뜻합니다.
- `Other` 배차에는 운전병을 배정하지 않습니다.
- `Other` 배차는 차량-only 배차로 처리합니다.
- 운전병 기량, 운전병 스케줄, 피로도, 공정성 제약은 `Other`에 적용하지 않습니다.
- 차량 가용성, 차량 시간 중복, 차종, 정비/입고, 탑승/적재량, 시간 제약은 그대로 적용합니다.

## 9. v0.5.0에서 추가된 것

- `src/ranker.py`: 후보 배차표 feature 추출, 규칙 기반 Ranker, Tree-based Ranker, 결과 저장
- `src/inverse_optimization.py`: 간이 역최적화 목적함수 가중치 보정
- `src/ranker_evaluation.py`: NDCG@5, Top-1 match 등 ranking 평가 헬퍼
- `src/feature_extractor.py`: Ranker용 후보 배차표 feature 및 expert similarity 계산 함수 추가
- `app.py`: Ranker/역최적화 실행 버튼 및 결과 화면 추가
- `outputs/ranker/`: Ranker/역최적화 결과 저장 구조 추가
- `reports/evaluation/ranker_training_report.xlsx`: 본선/기획서용 평가 리포트 생성

## 10. 현재 단순화된 부분

- 부족 시나리오는 실제 부대 데이터를 재현한 것이 아니라 본선 시연용 가상 stress test입니다.
- 정기배차 순번은 아직 별도 순번 데이터가 없어 config hook만 준비했습니다.
- 차량 보존은 차량별 세부 상태점수 대신 차량 운행시간 최대값을 낮추는 방식으로 단순화했습니다.
- 피로도는 과거 누적 피로도 없이 당일 운행시간 중심으로 단순화했습니다.
- Ranker는 소규모 후보 데이터 기반의 실증이며, 실제 3년치 배차반장표가 들어오면 재학습 구조로 확장해야 합니다.
- 역최적화는 완전한 수리계획 기반 bilevel inverse optimization이 아니라, feature-target 관계를 이용한 보수적 가중치 보정 방식입니다.

## 11. 가상환경 주의

`.venv` 폴더는 로컬 개발환경이며 zip에 포함하지 않습니다. 다른 컴퓨터에서는 `requirements.txt`로 재설치하세요.
