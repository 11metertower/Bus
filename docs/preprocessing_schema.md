# G-LIFT MVP 데이터 스키마 및 전처리 규칙

## 원본 파일 역할

| 원본 파일 | 표준 테이블 | 용도 |
|---|---|---|
| requests_raw.xlsx | requests | Optimizer 입력: 명일 배차 요청 |
| vehicles.xlsx | vehicles | Optimizer 입력: 차량 마스터 |
| vehicle_schedule.xlsx | vehicle_availability | Optimizer 입력: 차량 시간대별 가용성 |
| vehicle_schedule_assigned.xlsx | vehicle_schedule_assigned_long | 평가/비교: 배차반장표 반영 후 차량 점유 상태 |
| drivers.xlsx | drivers | Optimizer 입력: 운전병 마스터 |
| driver_availability.xlsx | driver_availability | Optimizer 입력: 운전병 시간대별 가용성 |
| driver_schedule_assigned.xlsx | driver_schedule_assigned_long | 평가/비교: 배차반장표 반영 후 운전병 점유 상태 |
| expert_best_schedule.xlsx | expert_schedule | 전문가 정답/Ranker 학습/역최적화 기준 |

## 핵심 규칙: Other 직접운전

- 운전자 열이 `Other`이면 `driver_mode = OTHER_DIRECT_DRIVER`.
- `requires_driver = False`.
- 운전병을 배정하지 않는다.
- 운전병 기량, 운전병 스케줄, 운전병 피로도, 운전병 공정성 제약에서 제외한다.
- 차량 가용성, 차량 시간 중복, 차량 차종, 차량 정비/입고 여부, 탑승/적재량, 배차 시간 제약은 그대로 적용한다.

## 표준 테이블 요약

### requests.csv

- `request_id`: REQ_001 형식의 요청 ID
- `request_type`: 배차 종류
- `requested_vehicle_id`: 신청 단계에서 희망 차량번호가 있는 경우
- `required_vehicle_class`: 표준화된 요청 차종
- `driver_raw`: 원본 운전자 값
- `destination`, `requesting_unit`, `purpose`
- `start_min`, `end_min`, `start_time`, `end_time`, `duration_min`
- `priority_initial`: MVP 초기 중요도
- `is_other`, `requires_driver`, `driver_mode`

### vehicles.csv

- `vehicle_id`, `vehicle_number`
- `vehicle_class`, `vehicle_class_raw`
- `transmission`
- `required_skill_text`
- `notes`
- `is_rental`, `is_electric`, `is_maintenance_or_unavailable`

### drivers.csv

- `driver_id`, `driver_name`
- `skill_raw`, `skill_code`
- `skill_small`, `skill_medium`, `skill_large`
- `skill_valid`, `skill_warning`

### vehicle_availability.csv / driver_availability.csv

- Wide T/F 스케줄을 5분 단위 long format으로 변환한다.
- `slot_index`, `slot_start_min`, `slot_end_min`, `slot_start`, `slot_end`
- `raw_value`, `is_available`

### expert_schedule.csv

- 배차반장이 짠 베스트 배차표를 표준화한 테이블.
- `assigned_vehicle_id`, `assigned_driver_raw`
- `is_other`, `requires_driver`, `driver_mode`
- Ranker 학습 및 역최적화 가중치 보정의 기준으로 사용한다.

## 실행 방법

```bash
python -m src.preprocess
```

## 생성 파일

- `data/processed/*.csv`
- `reports/validation/validation_report.csv`
- `reports/validation/validation_report.xlsx`
