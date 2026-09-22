"""
포트폴리오 파이프라인을 시각적으로 보여주는 Streamlit 대시보드.

실행 방법:
    streamlit run streamlit_app.py

(로컬 venv 안에서, ANTHROPIC_API_KEY 환경변수가 설정된 상태로 실행하세요.
 orchestration.py, llm_client.py, data_sources.py와 같은 폴더에 둬야 합니다.)
"""

import pandas as pd
import streamlit as st

from orchestration import DEFAULT_LIMITS, build_graph
from data_sources import fetch_company_detail, fetch_price_history

st.set_page_config(page_title="안정형 포트폴리오 빌더", layout="wide")


def _check_password() -> bool:
    """온라인에 배포했을 때 아무나 못 누르게 막는 간단한 비밀번호 게이트.

    Streamlit Cloud의 Secrets에 APP_PASSWORD를 설정하면 그때부터 활성화됩니다.
    로컬 컴퓨터에서 그냥 실행할 때는 secrets 자체가 없으니 이 게이트를 건너뜁니다
    (본인 컴퓨터에서만 도는 거라 안전).
    """
    try:
        password_required = "APP_PASSWORD" in st.secrets
    except Exception:
        # secrets.toml 자체가 없는 로컬 실행 환경 - 비밀번호 없이 통과시킵니다.
        password_required = False

    if not password_required:
        return True

    if st.session_state.get("_authenticated"):
        return True

    st.title("🔒 접속 확인")
    pw = st.text_input("비밀번호를 입력하세요", type="password")
    if pw:
        if pw == st.secrets["APP_PASSWORD"]:
            st.session_state["_authenticated"] = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다.")
    return False


if not _check_password():
    st.stop()

st.title("📊 안정형 포트폴리오 빌더")
st.caption(
    "섹터를 지정하면 Claude가 실제로 웹 검색해 종목을 찾고, "
    "안정성·밸류에이션 기준으로 걸러 포트폴리오를 구성합니다."
)

# ─────────────────────────────────────────────────────────────
# 사이드바: 설정
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    sector = st.text_input("관심 섹터", value="헬스케어")
    budget = st.number_input("예산 ($)", min_value=1000, value=10_000, step=1000)

    st.divider()
    st.subheader("안정성 기준")
    min_holdings = st.slider("최소 보유 종목 수", 1, 20, DEFAULT_LIMITS["min_holdings"])
    max_beta = st.slider("베타 상한", 0.5, 3.0, DEFAULT_LIMITS["max_beta"], step=0.1)
    min_growth_years = st.slider("최소 연속 이익성장 연수", 0, 5, DEFAULT_LIMITS["min_years_profit_growth"])
    max_debt = st.slider("부채비율 상한", 0.1, 5.0, DEFAULT_LIMITS["max_debt_to_equity"], step=0.1)

    st.divider()
    st.subheader("포트폴리오 구성 기준")
    max_single_weight = st.slider(
        "단일 종목 최대 비중", 0.05, 0.5, DEFAULT_LIMITS["max_single_stock_weight"], step=0.01
    )

    st.divider()
    run_button = st.button("🚀 포트폴리오 만들기", type="primary", width="stretch")
    st.caption("한 번 실행에 웹 검색 + LLM 호출 비용이 발생합니다 (보통 몇 센트~수십 센트 수준).")

if "result" not in st.session_state:
    st.session_state.result = None

if run_button:
    limits = dict(DEFAULT_LIMITS)
    limits.update({
        "min_holdings": min_holdings,
        "max_beta": max_beta,
        "min_years_profit_growth": min_growth_years,
        "max_debt_to_equity": max_debt,
        "max_single_stock_weight": max_single_weight,
    })

    initial_state = {
        "sector": sector,
        "budget": float(budget),
        "limits": limits,
        "candidates": [],
        "stable_candidates": [],
        "scored_candidates": [],
        "portfolio": [],
        "trade_plan": [],
        "current_holdings": [],
        "rebalance_actions": [],
        "stage1_audit": [],
        "stage2_audit": [],
        "status": "시작 전",
    }

    with st.spinner(f"'{sector}' 섹터를 검색하고 분석하는 중입니다... (1~2분 소요될 수 있어요)"):
        try:
            app = build_graph()
            st.session_state.result = app.invoke(initial_state)
        except Exception as e:
            st.error(f"파이프라인 실행 중 오류가 발생했습니다: {e}")
            st.session_state.result = None

result = st.session_state.result

if result is None:
    st.info("왼쪽에서 섹터와 조건을 설정하고 '포트폴리오 만들기'를 눌러주세요.")
    st.stop()

st.success(result["status"])

tab1, tab2, tab3 = st.tabs(["1️⃣ 종목 발굴", "2️⃣ 안정성 필터링", "3️⃣ 최종 포트폴리오"])

# ─────────────────────────────────────────────────────────────
# 탭 1: 종목 발굴 (제안 vs 검증 통과/탈락)
# ─────────────────────────────────────────────────────────────
with tab1:
    audit1 = result.get("stage1_audit", [])
    passed1 = [a for a in audit1 if a["passed"]]
    failed1 = [a for a in audit1 if not a["passed"]]

    col1, col2, col3 = st.columns(3)
    col1.metric("AI가 제안한 종목", len(audit1))
    col2.metric("실제 검증 통과", len(passed1))
    col3.metric("검증 탈락", len(failed1))

    st.subheader("✅ 검증 통과 (실제 존재 + 시가총액 기준 충족)")
    if passed1:
        df = pd.DataFrame([
            {
                "티커": a["ticker"],
                "회사명": a.get("company") or "-",
                "시가총액": f"${(a.get('market_cap') or 0) / 1e9:.1f}B" if a.get("market_cap") else "-",
                "AI가 제안한 이유": a.get("llm_reason") or "-",
            }
            for a in passed1
        ])
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.write("없음")

    st.subheader("❌ 검증 탈락 (지어낸 티커 또는 소형주)")
    if failed1:
        df = pd.DataFrame([
            {"티커": a["ticker"], "회사명": a.get("company") or "-", "탈락 이유": a.get("reject_reason")}
            for a in failed1
        ])
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.write("없음")

# ─────────────────────────────────────────────────────────────
# 탭 2: 안정성 필터링 (기준별 통과/탈락)
# ─────────────────────────────────────────────────────────────
with tab2:
    audit2 = result.get("stage2_audit", [])
    passed2 = [a for a in audit2 if a["passed"]]

    col1, col2 = st.columns(2)
    col1.metric("안정성 필터 대상", len(audit2))
    col2.metric("통과", len(passed2))

    for a in sorted(audit2, key=lambda x: not x["passed"]):
        icon = "✅" if a["passed"] else "❌"
        with st.expander(f"{icon} {a['ticker']}", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("베타", a.get("beta") if a.get("beta") is not None else "N/A")
            c2.metric("부채비율", a.get("debt_to_equity") if a.get("debt_to_equity") is not None else "N/A")
            c3.metric(
                "연속 이익성장",
                f"{a['profit_growth_years']}년" if a.get("profit_growth_years") is not None else "N/A",
            )

            if a["passed"]:
                st.info(f"**통과 근거**: {a.get('stability_rationale') or '-'}")
            else:
                st.warning("**탈락 이유**")
                for reason in a.get("failed_criteria", []):
                    st.write(f"- {reason}")

# ─────────────────────────────────────────────────────────────
# 탭 3: 최종 포트폴리오 (종목별 상세 정보)
# ─────────────────────────────────────────────────────────────
with tab3:
    portfolio = result.get("portfolio", [])

    if not portfolio:
        st.warning(
            "최소 보유 종목 수 조건을 만족하지 못해 포트폴리오가 구성되지 않았습니다. "
            "왼쪽에서 '최소 보유 종목 수'를 낮추거나 섹터를 넓혀보세요."
        )
    else:
        total_weight = sum(h.get("weight", 0) for h in portfolio)
        if total_weight < 0.99:
            st.warning(
                f"종목 수({len(portfolio)}개)가 부족해 단일 종목 상한을 지키는 선에서는 "
                f"예산의 {(1 - total_weight) * 100:.1f}%가 미배분 상태입니다."
            )

        st.subheader("비중 구성")
        chart_df = pd.DataFrame(
            [{"티커": h["ticker"], "비중": h.get("weight", 0)} for h in portfolio]
        ).set_index("티커")
        st.bar_chart(chart_df)

        st.subheader("종목별 상세")
        for h in sorted(portfolio, key=lambda x: -x.get("weight", 0)):
            ticker = h["ticker"]
            with st.expander(f"{ticker} — 비중 {h.get('weight', 0) * 100:.1f}%", expanded=False):
                try:
                    detail = fetch_company_detail(ticker)
                    history = fetch_price_history(ticker)
                except Exception as e:
                    st.error(f"상세 정보를 가져오지 못했습니다: {e}")
                    detail = {}
                    history = pd.DataFrame()

                st.markdown(f"### {detail.get('long_name', ticker)}")
                if detail.get("sector"):
                    st.caption(f"{detail.get('sector')} · {detail.get('industry', '')}")

                if not history.empty:
                    st.line_chart(history["Close"])
                else:
                    st.caption("차트 데이터를 가져오지 못했습니다.")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("현재가", f"${detail['current_price']:.2f}" if detail.get("current_price") else "N/A")
                c2.metric("베타", h.get("beta", "N/A"))
                c3.metric("부채비율", h.get("debt_to_equity", "N/A"))
                dy = detail.get("dividend_yield")
                c4.metric("배당수익률", f"{dy * 100:.2f}%" if dy else "N/A")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("PER", round(h["per"], 1) if h.get("per") is not None else "N/A")
                c2.metric("PBR", round(h["pbr"], 1) if h.get("pbr") is not None else "N/A")
                c3.metric("PEG", round(h["peg"], 2) if h.get("peg") is not None else "N/A")
                roe = h.get("roe")
                c4.metric("ROE", f"{roe * 100:.1f}%" if roe is not None else "N/A")

                st.markdown("**✅ 안정성 통과 근거**")
                st.info(h.get("stability_rationale") or "-")

                if detail.get("analyst_target_mean"):
                    st.markdown("**📈 애널리스트 목표주가 컨센서스**")
                    st.caption(
                        "Yahoo Finance가 집계한 제3자 애널리스트 의견입니다. "
                        "이 파이프라인이나 Claude가 만든 예측이 아니며, 실제 주가를 보장하지 않습니다."
                    )
                    st.write(
                        f"낮음 ${detail['analyst_target_low']:.2f} · "
                        f"평균 ${detail['analyst_target_mean']:.2f} · "
                        f"높음 ${detail['analyst_target_high']:.2f} "
                        f"({detail.get('num_analyst_opinions', '?')}명 의견 기준)"
                    )

                if h.get("high_correlation_with"):
                    st.markdown("**⚠️ 상관관계 주의**")
                    for c in h["high_correlation_with"]:
                        st.write(f"- {c['ticker']}와 상관계수 {c['correlation']} (함께 크게 움직이는 경향)")

        st.divider()
        st.subheader("🔔 리밸런싱 / 손절 제안")
        actions = result.get("rebalance_actions", [])
        if actions:
            for a in actions:
                action_label = {
                    "initial_buy": "최초 매수",
                    "stop_loss_review": "⚠️ 손절 검토",
                    "trim": "비중 축소",
                    "add": "비중 확대",
                }.get(a.get("action"), a.get("action"))
                st.write(f"**{a['ticker']}** — {action_label}")
                if a.get("rationale"):
                    st.caption(a["rationale"])
        else:
            st.write("제안 없음")