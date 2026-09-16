import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import PROCESSED_DIR, RAW_DIR, SCENARIOS_DIR, VALIDATION_REPORT_DIR
from src.preprocess import run_preprocessing
from src.scenario_generator import generate_all_scenarios
from src.candidate_generator import run_candidate_generation
from src.ranker import run_ranker_pipeline

st.set_page_config(page_title="G-LIFT MVP", layout="wide")

st.title("G-LIFT: 국방 육로수송 AI 배차 시스템 MVP")
st.caption("전처리 + 부족 시나리오 생성 + CP-SAT Optimizer + Ranker/역최적화 실증 · 가상/익명화 데이터 기준")

st.info(
    "버튼을 위에서 아래로 누르면 됩니다. ① 전처리 → ② 부족 시나리오 생성 → "
    "③ 시나리오 선택 → ④ 후보 배차표 5개 생성 → ⑤ Ranker/역최적화 실행. "
    "Other 직접운전은 운전병 없이 차량만 배정하는 차량-only 배차로 처리됩니다."
)

with st.expander("현재 프로젝트 경로", expanded=False):
    st.code(
        f"raw data: {RAW_DIR}\nprocessed csv: {PROCESSED_DIR}\nscenario data: {SCENARIOS_DIR}\nvalidation report: {VALIDATION_REPORT_DIR}",
        language="text",
    )

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("1단계: 전처리 실행", type="primary"):
        try:
            tables = run_preprocessing(RAW_DIR, PROCESSED_DIR, VALIDATION_REPORT_DIR)
            st.session_state["preprocess_summary"] = pd.DataFrame(
                [{"table": name, "rows": df.shape[0], "cols": df.shape[1]} for name, df in tables.items()]
            )
            st.session_state["requests_preview"] = tables["requests"].head(20)
            st.session_state["other_preview"] = tables["requests"].loc[
                tables["requests"]["is_other"],
                ["request_id", "request_type", "required_vehicle_class", "driver_raw", "requires_driver", "driver_mode", "start_time", "end_time"],
            ]
            st.session_state["validation_report"] = tables["validation_report"]
            st.success("전처리가 완료되었습니다.")
        except Exception as exc:
            st.error("전처리 중 오류가 발생했습니다.")
            st.exception(exc)

with col2:
    if st.button("2단계: 부족 시나리오 자동 생성", type="secondary"):
        try:
            with st.spinner("기존 표준 CSV를 복사·변형하여 부족 시나리오를 생성 중입니다..."):
                results = generate_all_scenarios(PROCESSED_DIR, SCENARIOS_DIR)
            st.session_state["scenario_summary"] = pd.DataFrame([
                {
                    "scenario_name": r.scenario_name,
                    "scenario_dir": str(r.scenario_dir),
                    "target_request_ids": ", ".join(r.target_request_ids),
                    "description": r.description,
                }
                for r in results
            ])
            st.success("부족 시나리오 생성이 완료되었습니다.")
        except Exception as exc:
            st.error("시나리오 생성 중 오류가 발생했습니다.")
            st.exception(exc)

with col3:
    known = ["baseline", "vehicle_shortage", "driver_shortage", "skill_shortage", "other_vehicle_conflict", "mixed_shortage"]
    existing = [s for s in known if (SCENARIOS_DIR / s).exists()]
    if not existing:
        existing = ["baseline"]
    selected_scenario = st.selectbox("3단계: 실행할 시나리오 선택", existing, index=0)
    if st.button("4단계: 선택 시나리오 Optimizer 실행", type="secondary"):
        try:
            with st.spinner(f"{selected_scenario} 시나리오에 대해 후보 배차표 5개를 생성 중입니다..."):
                result = run_candidate_generation(scenario_name=selected_scenario)
            st.session_state["optimizer_result"] = result
            st.success(f"{selected_scenario} 후보 배차표 5개 생성이 완료되었습니다.")
        except Exception as exc:
            st.error("Optimizer 실행 중 오류가 발생했습니다.")
            st.exception(exc)

with col4:
    ranker_scope = st.selectbox("5단계: Ranker 실행 범위", ["all", "baseline", "vehicle_shortage", "driver_shortage", "skill_shortage", "other_vehicle_conflict", "mixed_shortage"], index=0)
    if st.button("5단계: Ranker/역최적화 실행", type="secondary"):
        try:
            with st.spinner("후보 배차표 feature 추출, Ranker 학습, 역최적화 가중치 보정 중입니다..."):
                ranker_result = run_ranker_pipeline(scenario=ranker_scope)
            st.session_state["ranker_result"] = ranker_result
            st.success("Ranker/역최적화 실증 실행이 완료되었습니다.")
        except Exception as exc:
            st.error("Ranker/역최적화 실행 중 오류가 발생했습니다. 먼저 후보 배차표 5개가 생성되어 있어야 합니다.")
            st.exception(exc)

st.divider()

if "preprocess_summary" in st.session_state:
    st.subheader("전처리 결과 요약")
    st.dataframe(st.session_state["preprocess_summary"], use_container_width=True)

    with st.expander("requests 미리보기", expanded=False):
        st.dataframe(st.session_state["requests_preview"], use_container_width=True)

    with st.expander("Other 직접운전 처리 확인", expanded=True):
        st.dataframe(st.session_state["other_preview"], use_container_width=True)

    with st.expander("검증 리포트", expanded=False):
        st.dataframe(st.session_state["validation_report"], use_container_width=True)

if "scenario_summary" in st.session_state:
    st.subheader("부족 시나리오 생성 결과")
    st.dataframe(st.session_state["scenario_summary"], use_container_width=True)

    selected_meta = st.selectbox("metadata 확인", st.session_state["scenario_summary"]["scenario_name"].tolist(), key="meta_select")
    meta_path = SCENARIOS_DIR / selected_meta / "scenario_metadata.json"
    if meta_path.exists():
        st.json(json.loads(meta_path.read_text(encoding="utf-8")))

if "optimizer_result" in st.session_state:
    result = st.session_state["optimizer_result"]
    score_breakdown = result["score_breakdown"]
    scenario_name = result["scenario_name"]
    st.subheader(f"후보 배차표 5개 점수 요약: {scenario_name}")
    display_cols = [
        "scenario_name", "candidate_id", "candidate_profile", "solver_status", "assigned_count", "unserved_count",
        "other_assigned_count", "max_driver_minutes", "max_vehicle_minutes", "hard_violation_count",
        "diversity_score_vs_previous", "same_assignment_count_vs_previous",
    ]
    st.dataframe(score_breakdown[[c for c in display_cols if c in score_breakdown.columns]], use_container_width=True)

    st.subheader("후보 배차표 상세")
    tabs = st.tabs([s.score_breakdown["candidate_id"] for s in result["solutions"]])
    for tab, solution in zip(tabs, result["solutions"]):
        with tab:
            st.write(f"**{solution.score_breakdown['candidate_profile']}**: {solution.score_breakdown['description']}")
            st.markdown("**배차표**")
            st.dataframe(solution.schedule, use_container_width=True)

            st.markdown("**Other 직접운전 배차만 보기**")
            other_rows = solution.schedule[solution.schedule["assignment_mode"].eq("OTHER_DIRECT_DRIVER")]
            st.dataframe(other_rows, use_container_width=True)

            st.markdown("**미지원 사유**")
            if solution.unserved_reasons.empty:
                st.success("미지원 배차가 없습니다.")
            else:
                st.dataframe(solution.unserved_reasons, use_container_width=True)

            st.markdown("**하드 제약 검증**")
            st.dataframe(solution.hard_constraint_check, use_container_width=True)

    st.subheader("생성된 Optimizer 파일 경로")
    for name, path in result["output_paths"].items():
        st.write(f"- {name}: `{path}`")
else:
    st.info("전처리와 시나리오 생성을 마친 뒤, 시나리오를 선택하여 Optimizer를 실행하세요.")

if "ranker_result" in st.session_state:
    ranker_result = st.session_state["ranker_result"]
    st.divider()
    st.subheader("Ranker/역최적화 실증 결과")
    st.caption("소규모 MVP 데이터이므로 완전한 학습 성능이 아니라 배차반장 암묵지 학습 구조의 실증 개념으로 해석해야 합니다.")

    st.markdown("**후보 배차표 feature matrix**")
    st.dataframe(ranker_result["feature_matrix"], use_container_width=True)

    st.markdown("**학습 전/학습 후 후보 순위 비교**")
    st.dataframe(ranker_result["comparison"], use_container_width=True)

    st.markdown("**역최적화 가중치 보정표**")
    st.dataframe(ranker_result["learned_weights"], use_container_width=True)

    st.markdown("**Tree-based Ranker feature importance**")
    st.dataframe(ranker_result["feature_importance"], use_container_width=True)

    st.markdown("**Ranker 학습 리포트**")
    st.dataframe(ranker_result["training_report"], use_container_width=True)

    st.subheader("생성된 Ranker/역최적화 파일 경로")
    for name, path in ranker_result["output_paths"].items():
        st.write(f"- {name}: `{path}`")
