"""G-LIFT 배차 도우미 — 배차 담당 실무자용 원클릭 화면.

내부 파이프라인(전처리 / CP-SAT Optimizer / Ranker / 역최적화) 용어를 감추고,
'상황 선택 -> 배차안 만들기 -> 추천안 확인'의 세 걸음으로 단순화한 화면입니다.
전문가용 상세(피처 행렬, 가중치, 파일 경로)는 맨 아래 '전문가 상세'에 접어 둡니다.
"""

import pandas as pd
import streamlit as st

from src.config import PROCESSED_DIR, RAW_DIR, SCENARIOS_DIR, VALIDATION_REPORT_DIR
from src.candidate_generator import run_candidate_generation
from src.preprocess import run_preprocessing
from src.ranker import run_ranker_pipeline
from src.scenario_generator import generate_all_scenarios

st.set_page_config(page_title="G-LIFT 배차 도우미", page_icon="🚚", layout="wide")

# --------------------------------------------------------------------------- #
# 화면에서 쓰는 쉬운 이름 매핑
# --------------------------------------------------------------------------- #

# (내부 시나리오 코드, 화면 라벨, 한 줄 설명)
SITUATIONS = [
    ("baseline", "🟢 평상시", "특별한 부족 없이 정상적인 하루입니다."),
    ("vehicle_shortage", "🚗 차량이 부족해요", "정비·입고 등으로 일부 차량을 쓸 수 없는 상황입니다."),
    ("driver_shortage", "🧑‍✈️ 운전병이 부족해요", "운전병 인원이 부족한 상황입니다."),
    ("skill_shortage", "📄 자격이 부족해요", "필요한 면허·기량을 가진 운전병이 부족한 상황입니다."),
    ("other_vehicle_conflict", "🔁 외부 차량이 겹쳐요", "직접운전(Other) 차량 사용이 서로 겹치는 상황입니다."),
    ("mixed_shortage", "⚠️ 여러 문제가 겹쳐요", "여러 부족이 동시에 발생한 가장 어려운 상황입니다."),
]

# 후보 배차안(candidate_id) -> (아이콘, 쉬운 이름, 한 줄 설명)
CANDIDATE_META = {
    "candidate_01": ("📋", "최대 배차형", "가능한 한 많은 요청을 배차합니다. 미지원을 최소화합니다."),
    "candidate_02": ("⚖️", "공정 배분형", "운전병들의 운행 부담을 고르게 나눕니다."),
    "candidate_03": ("🚗", "차량 보호형", "특정 차량이 혹사되지 않도록 분산합니다."),
    "candidate_04": ("😴", "피로 최소형", "운전병의 장시간 연속 운전을 줄입니다."),
    "candidate_05": ("⭐", "배차반장 스타일형", "기존 배차반장님의 배차 방식과 가장 비슷합니다."),
}

STATUS_LABEL = {"ASSIGNED": "✅ 배차완료", "UNSERVED": "⚠️ 미지원"}
MODE_LABEL = {
    "DRIVER_REQUIRED": "운전병 배정",
    "OTHER_DIRECT_DRIVER": "직접운전(차량만)",
    "UNSERVED": "—",
}

SITUATION_LABELS = {code: label for code, label, _ in SITUATIONS}
SITUATION_DESCS = {code: desc for code, _, desc in SITUATIONS}


# --------------------------------------------------------------------------- #
# 표시용 헬퍼
# --------------------------------------------------------------------------- #

def candidate_title(candidate_id: str) -> str:
    icon, name, _ = CANDIDATE_META.get(candidate_id, ("•", candidate_id, ""))
    return f"{icon} {name}"


def _clean_str(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def friendly_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """배차표를 실무자가 읽기 쉬운 한글 표로 변환한다."""
    if schedule is None or schedule.empty:
        return pd.DataFrame()

    def time_range(row) -> str:
        start, end = _clean_str(row.get("start_time")), _clean_str(row.get("end_time"))
        return f"{start} ~ {end}" if start and end else ""

    def guidance(row) -> str:
        if row.get("status") == "ASSIGNED":
            return _clean_str(row.get("explanation"))
        return _clean_str(row.get("unserved_reason_detail"))

    vehicle = schedule.get("assigned_vehicle_number", schedule.get("assigned_vehicle_id", ""))
    return pd.DataFrame(
        {
            "요청번호": schedule["request_id"].map(_clean_str),
            "요청 유형": schedule.get("request_type", "").map(_clean_str),
            "시간": schedule.apply(time_range, axis=1),
            "목적지": schedule.get("destination", "").map(_clean_str),
            "신청 부대": schedule.get("requesting_unit", "").map(_clean_str),
            "상태": schedule["status"].map(lambda s: STATUS_LABEL.get(s, s)),
            "배차 방식": schedule["assignment_mode"].map(lambda m: MODE_LABEL.get(m, "—")),
            "배정 차량": pd.Series(vehicle, index=schedule.index).map(_clean_str),
            "운전병": schedule.get("assigned_driver_name", "").map(_clean_str),
            "안내": schedule.apply(guidance, axis=1),
        }
    )


def pick_recommended_id(comparison: pd.DataFrame) -> str:
    """랭커 비교표에서 가장 추천되는 후보 배차안의 candidate_id를 고른다."""
    if comparison is None or comparison.empty:
        return "candidate_01"
    for col in ["ml_ranker_score", "rule_based_score", "expert_similarity_score"]:
        if col in comparison.columns and comparison[col].notna().any():
            return str(comparison.sort_values(col, ascending=False).iloc[0]["candidate_id"])
    return str(comparison.iloc[0]["candidate_id"])


def expert_similarity_for(comparison: pd.DataFrame, candidate_id: str):
    if comparison is None or comparison.empty or "expert_similarity_score" not in comparison.columns:
        return None
    match = comparison[comparison["candidate_id"] == candidate_id]
    if match.empty or pd.isna(match.iloc[0]["expert_similarity_score"]):
        return None
    return float(match.iloc[0]["expert_similarity_score"])


def run_full_plan(situation: str) -> dict:
    """상황을 받아 배차안 5개 생성 + 추천 순위 계산까지 한 번에 수행한다."""
    # 부족 상황은 시나리오 데이터가 필요하므로, 없으면 자동으로 만든다.
    if situation != "baseline" and not (SCENARIOS_DIR / situation).exists():
        generate_all_scenarios(PROCESSED_DIR, SCENARIOS_DIR)

    candidate_result = run_candidate_generation(scenario_name=situation)

    # 추천 순위(배차반장 스타일 학습)는 실패해도 배차안 자체는 보여준다.
    comparison = None
    ranker_result = None
    try:
        ranker_result = run_ranker_pipeline(scenario=situation)
        comparison = ranker_result["comparison"]
    except Exception:  # noqa: BLE001 - 데모 안정성 우선
        comparison = None

    return {
        "situation": situation,
        "candidate_result": candidate_result,
        "ranker_result": ranker_result,
        "comparison": comparison,
        "recommended_id": pick_recommended_id(comparison),
    }


# --------------------------------------------------------------------------- #
# 화면
# --------------------------------------------------------------------------- #

st.title("🚚 G-LIFT 배차 도우미")
st.caption("명일 배차 요청을 AI가 검토해, 규정을 지키면서 서로 다른 관점의 배차안 5개를 만들어 드립니다.")

processed_ready = (PROCESSED_DIR / "requests.csv").exists()
if not processed_ready:
    st.error(
        "아직 배차 데이터가 준비되지 않았습니다. 아래 '전문가 상세 · 데이터 관리'에서 "
        "'원본 데이터 불러오기'를 먼저 실행해 주세요."
    )

st.markdown("### 1. 오늘 상황을 골라주세요")
situation = st.radio(
    "오늘 배차 상황",
    options=[code for code, _, _ in SITUATIONS],
    format_func=lambda code: SITUATION_LABELS.get(code, code),
    horizontal=True,
    label_visibility="collapsed",
)
st.caption(SITUATION_DESCS.get(situation, ""))

st.markdown("### 2. 배차안을 만들어 보세요")
run_clicked = st.button(
    "🤖 AI 배차 계획 만들기",
    type="primary",
    use_container_width=True,
    disabled=not processed_ready,
)

if run_clicked:
    with st.status("AI가 배차 계획을 만드는 중입니다...", expanded=True) as status:
        st.write("① 배차 요청과 차량·운전병 현황을 확인하고 있어요.")
        st.write("② 규정을 지키면서 가능한 배차안들을 계산하고 있어요.")
        st.write("③ 배차반장님 방식과 비교해 추천 순위를 매기고 있어요.")
        try:
            st.session_state["plan"] = run_full_plan(situation)
            status.update(label="배차 계획이 완성되었습니다!", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001
            status.update(label="배차 계획을 만들지 못했습니다.", state="error")
            st.exception(exc)

# --------------------------------------------------------------------------- #
# 결과
# --------------------------------------------------------------------------- #

plan = st.session_state.get("plan")
if plan:
    st.divider()
    result = plan["candidate_result"]
    solutions = result["solutions"]
    by_id = {s.score_breakdown["candidate_id"]: s for s in solutions}
    recommended_id = plan["recommended_id"] if plan["recommended_id"] in by_id else list(by_id)[0]
    recommended = by_id[recommended_id]
    rec_sb = recommended.score_breakdown

    total = int(rec_sb.get("assigned_count", 0)) + int(rec_sb.get("unserved_count", 0))
    assigned = int(rec_sb.get("assigned_count", 0))
    unserved = int(rec_sb.get("unserved_count", 0))
    violations = int(rec_sb.get("hard_violation_count", 0))

    st.markdown(f"## 3. 배차 결과 — {SITUATION_LABELS.get(plan['situation'], plan['situation'])}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 요청", f"{total}건")
    c2.metric("배차 완료", f"{assigned}건")
    c3.metric("미지원", f"{unserved}건", delta=None)
    c4.metric("규정 위반", "0건 ✅" if violations == 0 else f"{violations}건 ⚠️")

    # 추천 배차안 배너
    icon, name, desc = CANDIDATE_META.get(recommended_id, ("⭐", recommended_id, ""))
    similarity = expert_similarity_for(plan["comparison"], recommended_id)
    sim_text = f" · 배차반장님 방식과 약 {similarity:.0f}% 유사" if similarity is not None else ""
    st.success(f"**AI 추천: {icon} {name}** — {desc}{sim_text}")
    st.caption("아래 탭에서 5가지 관점의 배차안을 비교해 보세요. ⭐ 표시가 AI 추천안입니다.")

    # 후보 배차안 탭
    ordered_ids = [cid for cid in CANDIDATE_META if cid in by_id]
    tab_labels = [
        f"{candidate_title(cid)}{'  ⭐' if cid == recommended_id else ''}" for cid in ordered_ids
    ]
    tabs = st.tabs(tab_labels)
    for tab, cid in zip(tabs, ordered_ids):
        solution = by_id[cid]
        sb = solution.score_breakdown
        with tab:
            _, cname, cdesc = CANDIDATE_META.get(cid, ("", cid, ""))
            st.markdown(f"**{cname}** — {cdesc}")
            if cid == recommended_id:
                st.info("이 배차안이 AI 추천안입니다.")

            m1, m2, m3, m4 = st.columns(4)
            c_total = int(sb.get("assigned_count", 0)) + int(sb.get("unserved_count", 0))
            m1.metric("배차 완료", f"{int(sb.get('assigned_count', 0))} / {c_total}건")
            m2.metric("미지원", f"{int(sb.get('unserved_count', 0))}건")
            m3.metric("직접운전(차량만)", f"{int(sb.get('other_assigned_count', 0))}건")
            m4.metric("규정 위반", "0건 ✅" if int(sb.get("hard_violation_count", 0)) == 0 else f"{int(sb.get('hard_violation_count', 0))}건 ⚠️")

            table = friendly_schedule(solution.schedule)
            only_unserved = st.checkbox("미지원 요청만 보기", key=f"unserved_{cid}")
            if only_unserved:
                table = table[table["상태"] == STATUS_LABEL["UNSERVED"]]
            st.dataframe(table, use_container_width=True, hide_index=True)

            if not solution.unserved_reasons.empty:
                with st.expander(f"⚠️ 미지원 {int(sb.get('unserved_count', 0))}건 — 왜 배차하지 못했나요?"):
                    reasons = solution.unserved_reasons
                    show = pd.DataFrame(
                        {
                            "요청번호": reasons["request_id"].map(_clean_str),
                            "요청 유형": reasons.get("request_type", "").map(_clean_str),
                            "미지원 사유": reasons.get("unserved_reason_detail", "").map(_clean_str),
                            "해결 힌트": reasons.get("possible_relaxation", "").map(_clean_str),
                        }
                    )
                    st.dataframe(show, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# 전문가 상세 · 데이터 관리 (접어 둠)
# --------------------------------------------------------------------------- #

with st.expander("🔧 전문가 상세 · 데이터 관리", expanded=False):
    st.caption(
        "내부 파이프라인 제어와 상세 분석 결과입니다. 일반 사용 시에는 열지 않아도 됩니다."
    )

    st.markdown("**원본 데이터 관리**")
    dc1, dc2 = st.columns(2)
    with dc1:
        if st.button("원본 데이터 불러오기 (전처리)"):
            try:
                with st.spinner("원본 엑셀을 표준 데이터로 변환 중입니다..."):
                    run_preprocessing(RAW_DIR, PROCESSED_DIR, VALIDATION_REPORT_DIR)
                st.success("원본 데이터를 불러왔습니다.")
            except Exception as exc:  # noqa: BLE001
                st.error("원본 데이터 불러오기에 실패했습니다. 엑셀 파일이 DRM 등으로 잠겨 있지 않은지 확인하세요.")
                st.exception(exc)
    with dc2:
        if st.button("부족 상황 데이터 다시 만들기"):
            try:
                with st.spinner("부족 상황 시나리오를 재생성 중입니다..."):
                    generate_all_scenarios(PROCESSED_DIR, SCENARIOS_DIR)
                st.success("부족 상황 데이터를 다시 만들었습니다.")
            except Exception as exc:  # noqa: BLE001
                st.error("부족 상황 데이터 생성에 실패했습니다.")
                st.exception(exc)

    if plan and plan.get("ranker_result"):
        rr = plan["ranker_result"]
        st.markdown("**배차반장 스타일 학습(Ranker/역최적화) 상세**")
        st.caption("소규모 MVP 데이터 기반의 실증 개념입니다. 완전한 학습 성능으로 해석하지 마세요.")

        st.markdown("학습 전/후 후보 순위 비교")
        st.dataframe(rr["comparison"], use_container_width=True)

        st.markdown("역최적화 가중치 보정표")
        st.dataframe(rr["learned_weights"], use_container_width=True)

        st.markdown("Tree 기반 Ranker feature importance")
        st.dataframe(rr["feature_importance"], use_container_width=True)

        st.markdown("Ranker 학습 리포트")
        st.dataframe(rr["training_report"], use_container_width=True)

    if plan:
        st.markdown("**생성된 파일 경로**")
        for name, path in plan["candidate_result"]["output_paths"].items():
            st.write(f"- {name}: `{path}`")
        if plan.get("ranker_result"):
            for name, path in plan["ranker_result"]["output_paths"].items():
                st.write(f"- {name}: `{path}`")
