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

## 6. Streamlit 실행

```bash
streamlit run app.py
```

또는

```bash
python -m streamlit run app.py
```

웹 화면에서 다음 순서대로 실행합니다.

1. 전처리 실행
2. 부족 시나리오 자동 생성
3. 시나리오 선택
4. 선택 시나리오 Optimizer 실행
5. 후보 배차표, 미지원 사유, 하드 제약 검증, Other 배차 처리 결과 확인
6. Ranker/역최적화 실행
7. 학습 전/학습 후 후보 순위 변화 확인
8. 목적함수 가중치 보정표 확인

## 7. 핵심 규칙

- `Other`는 별도 직접운전자를 뜻합니다.
- `Other` 배차에는 운전병을 배정하지 않습니다.
- `Other` 배차는 차량-only 배차로 처리합니다.
- 운전병 기량, 운전병 스케줄, 피로도, 공정성 제약은 `Other`에 적용하지 않습니다.
- 차량 가용성, 차량 시간 중복, 차종, 정비/입고, 탑승/적재량, 시간 제약은 그대로 적용합니다.

## 8. v0.5.0에서 추가된 것

- `src/ranker.py`: 후보 배차표 feature 추출, 규칙 기반 Ranker, Tree-based Ranker, 결과 저장
- `src/inverse_optimization.py`: 간이 역최적화 목적함수 가중치 보정
- `src/ranker_evaluation.py`: NDCG@5, Top-1 match 등 ranking 평가 헬퍼
- `src/feature_extractor.py`: Ranker용 후보 배차표 feature 및 expert similarity 계산 함수 추가
- `app.py`: Ranker/역최적화 실행 버튼 및 결과 화면 추가
- `outputs/ranker/`: Ranker/역최적화 결과 저장 구조 추가
- `reports/evaluation/ranker_training_report.xlsx`: 본선/기획서용 평가 리포트 생성

## 9. 현재 단순화된 부분

- 부족 시나리오는 실제 부대 데이터를 재현한 것이 아니라 본선 시연용 가상 stress test입니다.
- 정기배차 순번은 아직 별도 순번 데이터가 없어 config hook만 준비했습니다.
- 차량 보존은 차량별 세부 상태점수 대신 차량 운행시간 최대값을 낮추는 방식으로 단순화했습니다.
- 피로도는 과거 누적 피로도 없이 당일 운행시간 중심으로 단순화했습니다.
- Ranker는 소규모 후보 데이터 기반의 실증이며, 실제 3년치 배차반장표가 들어오면 재학습 구조로 확장해야 합니다.
- 역최적화는 완전한 수리계획 기반 bilevel inverse optimization이 아니라, feature-target 관계를 이용한 보수적 가중치 보정 방식입니다.

## 10. 가상환경 주의

`.venv` 폴더는 로컬 개발환경이며 zip에 포함하지 않습니다. 다른 컴퓨터에서는 `requirements.txt`로 재설치하세요.
