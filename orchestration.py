"""
안정적 우상향(Quality Growth) 목표에 맞춘 주식 분석 에이전트 파이프라인.
"""

from __future__ import annotations

import json
import re

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from llm_client import call_llm, call_llm_with_web_search
from data_sources import (
    fetch_fundamentals_batch,
    fetch_valuation_batch,
    compute_valuation_scores,
    fetch_price_returns,
    compute_correlation_matrix,
    find_high_correlation_pairs,
    fetch_current_prices,
    verify_ticker_and_market_cap,
)


MODEL_FAST = "claude-haiku-4-5-20251001"
MODEL_MID = "claude-sonnet-5"
MODEL_TOP = "claude-opus-5"


class RiskLimits(TypedDict):
    max_beta: float
    min_years_profit_growth: int
    max_debt_to_equity: float
    max_single_stock_weight: float
    max_single_sector_weight: float
    min_holdings: int
    correlation_ceiling: float
    stop_loss_pct: float
    min_market_cap: float


DEFAULT_LIMITS: RiskLimits = {
    "max_beta": 1.3,
    "min_years_profit_growth": 3,
    "max_debt_to_equity": 1.0,
    "max_single_stock_weight": 0.10,
    "max_single_sector_weight": 0.25,
    "min_holdings": 12,
    "correlation_ceiling": 0.7,
    "stop_loss_pct": -0.15,
    "min_market_cap": 10_000_000_000,
}


class PortfolioState(TypedDict):
    sector: str
    budget: float
    limits: RiskLimits

    candidates: list[dict]
    stable_candidates: list[dict]
    scored_candidates: list[dict]
    portfolio: list[dict]
    trade_plan: list[dict]

    current_holdings: list[dict]
    rebalance_actions: list[dict]

    stage1_audit: list[dict]
    stage2_audit: list[dict]

    status: str


_SECTOR_DISCOVERY_SYSTEM_PROMPT = """너는 보수적인 주식 리서처다. 웹 검색으로 실제 정보를 확인한 뒤,
주어진 섹터에서 실제로 존재하고 미국 또는 주요 거래소에 상장된 중대형주만 후보로 제시하라.

반드시 지켜야 할 규칙:
- 최근 상장(IPO)한 지 얼마 안 됐거나, 적자가 지속되거나, 투기성이 강한 소형주/신생 산업은 제외한다.
- 지어내지 말고, 실제로 검색해서 확인한 회사와 티커만 제시한다.
- 최소 15개, 최대 25개의 후보를 제시한다.

마지막에 반드시 다른 설명 없이 아래 형식의 JSON 코드블록만 출력하라:
```json
[{"ticker": "실제 거래소 티커", "company": "회사명", "reason": "한 줄 이유"}]
```"""


def _extract_json_list(text: str) -> list[dict]:
    matches = re.findall(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    if not matches:
        matches = re.findall(r"(\[\s*\{.*?\}\s*\])", text, re.DOTALL)
    if not matches:
        raise ValueError("응답에서 JSON 리스트를 찾지 못했습니다.")
    return json.loads(matches[-1])


def sector_researcher_and_screener(state: PortfolioState) -> PortfolioState:
    """1단계: 섹터 리서처 + 티커 스크리너."""
    if state.get("candidates"):
        state["stage1_audit"] = [
            {"ticker": c["ticker"], "company": c.get("company"), "market_cap": c.get("market_cap"),
             "llm_reason": "수동 지정", "passed": True, "reject_reason": None}
            for c in state["candidates"]
        ]
        state["status"] = f"1단계 완료: 수동 지정 종목 사용 ({len(state['candidates'])}종목, 웹 검색 건너뜀)"
        print(f'>>> {state["status"]}')
        return state

    sector = state["sector"]
    min_market_cap = state["limits"]["min_market_cap"]

    try:
        raw_response = call_llm_with_web_search(
            model=MODEL_MID,
            system_prompt=_SECTOR_DISCOVERY_SYSTEM_PROMPT,
            user_prompt=f"섹터: {sector}",
        )
        proposed = _extract_json_list(raw_response)
    except Exception as e:
        state["candidates"] = []
        state["stage1_audit"] = []
        state["status"] = f"1단계 실패: 섹터 리서치 오류로 후보 0종목 ({e})"
        print(f'>>> {state["status"]}')
        return state

    verified: list[dict] = []
    audit: list[dict] = []
    for item in proposed:
        ticker = item.get("ticker")
        if not ticker:
            continue
        check = verify_ticker_and_market_cap(ticker, min_market_cap)
        if check:
            entry = {**check, "company": item.get("company"), "llm_reason": item.get("reason")}
            verified.append(entry)
            audit.append({**entry, "passed": True, "reject_reason": None})
        else:
            audit.append({
                "ticker": ticker, "company": item.get("company"), "market_cap": None,
                "llm_reason": item.get("reason"), "passed": False,
                "reject_reason": f"티커 미존재 또는 시가총액 {min_market_cap/1e9:.0f}억 달러 미만",
            })

    state["candidates"] = verified
    state["stage1_audit"] = audit
    state["status"] = (
        f"1단계 완료: 섹터 리서치로 {len(proposed)}개 제안 -> 실제 검증 통과 {len(verified)}종목 "
        f"(시가총액 미달/티커 오류 등으로 탈락 {len(proposed) - len(verified)}종목)"
    )
    print(f'>>> {state["status"]}')
    return state


_RATIONALE_SCHEMA = {
    "type": "object",
    "properties": {"rationale": {"type": "string"}},
    "required": ["rationale"],
    "additionalProperties": False,
}


def stability_screener(state: PortfolioState) -> PortfolioState:
    """2단계: 안정성 필터링."""
    limits = state["limits"]
    tickers = [c["ticker"] for c in state["candidates"]]

    fetched, failed = fetch_fundamentals_batch(tickers)

    passed: list[dict] = []
    audit: list[dict] = []

    for f in fetched:
        failed_criteria = []
        beta = f.get("beta")
        debt = f.get("debt_to_equity")
        growth = f.get("profit_growth_years", 0)

        if beta is None or beta > limits["max_beta"]:
            failed_criteria.append(f"베타 {beta} (기준: {limits['max_beta']} 이하)")
        if debt is None or debt > limits["max_debt_to_equity"]:
            failed_criteria.append(f"부채비율 {debt} (기준: {limits['max_debt_to_equity']} 이하)")
        if growth < limits["min_years_profit_growth"]:
            failed_criteria.append(f"연속 이익성장 {growth}년 (기준: {limits['min_years_profit_growth']}년 이상)")

        entry = dict(f)
        entry["passed"] = len(failed_criteria) == 0
        entry["failed_criteria"] = failed_criteria
        audit.append(entry)
        if entry["passed"]:
            passed.append(entry)

    for ticker in failed:
        audit.append({
            "ticker": ticker, "beta": None, "debt_to_equity": None, "profit_growth_years": None,
            "passed": False, "failed_criteria": ["데이터 조회 실패"],
        })

    for stock in passed:
        try:
            result = call_llm(
                model=MODEL_MID,
                system_prompt=(
                    "너는 보수적인 안정성 심사역이다. 아래 지표만 근거로 삼아 "
                    "이 종목이 왜 안정성 기준을 통과했는지 한국어 한 문장으로 설명하라. "
                    "지표에 없는 내용을 추측하거나 과장하지 마라."
                ),
                user_prompt=(
                    f"티커: {stock['ticker']}\n"
                    f"베타: {stock['beta']}\n"
                    f"부채비율: {stock['debt_to_equity']}\n"
                    f"연속 이익성장 연수: {stock['profit_growth_years']}년\n"
                    f"시가총액: {stock.get('market_cap')}"
                ),
                json_schema=_RATIONALE_SCHEMA,
            )
            stock["stability_rationale"] = result["rationale"]
        except Exception:
            stock["stability_rationale"] = None

    state["stable_candidates"] = passed
    state["stage2_audit"] = audit
    state["status"] = (
        f"2단계 완료: 안정성 필터 통과 {len(passed)}종목 "
        f"(데이터 조회 실패로 자동 탈락 {len(failed)}종목: {failed})"
    )
    print(f'>>> {state["status"]}')
    return state


def valuation_and_risk_analyst(state: PortfolioState) -> PortfolioState:
    """3단계: 밸류에이션 스코어링 + 상관관계 기반 리스크 분석."""
    stable = state["stable_candidates"]
    tickers = [c["ticker"] for c in stable]

    valuation_metrics = fetch_valuation_batch(tickers)
    valuation_metrics = compute_valuation_scores(valuation_metrics)
    valuation_by_ticker = {m["ticker"]: m for m in valuation_metrics}

    scored = []
    for c in stable:
        v = valuation_by_ticker.get(c["ticker"], {})
        merged = {**c, **{k: val for k, val in v.items() if k != "ticker"}}
        scored.append(merged)

    high_corr_pairs: list[tuple[str, str, float]] = []
    if len(tickers) >= 2:
        try:
            returns = fetch_price_returns(tickers)
            if not returns.empty:
                corr = compute_correlation_matrix(returns)
                high_corr_pairs = find_high_correlation_pairs(
                    corr, state["limits"]["correlation_ceiling"]
                )
        except Exception:
            high_corr_pairs = []

    for ticker_a, ticker_b, corr_value in high_corr_pairs:
        for holding in scored:
            if holding["ticker"] in (ticker_a, ticker_b):
                other = ticker_b if holding["ticker"] == ticker_a else ticker_a
                holding.setdefault("high_correlation_with", []).append(
                    {"ticker": other, "correlation": corr_value}
                )

    state["scored_candidates"] = scored
    state["status"] = (
        f"3단계 완료: 밸류에이션 스코어링 {len(scored)}종목, "
        f"상관관계 {state['limits']['correlation_ceiling']} 이상 쌍 {len(high_corr_pairs)}개: {high_corr_pairs}"
    )
    print(f'>>> {state["status"]}')
    return state


def compute_weights(scored: list[dict], max_single_weight: float) -> list[dict]:
    """valuation_score를 상대적 비중으로 변환하고 단일 종목 상한을 적용합니다."""
    n = len(scored)
    if n == 0:
        return scored
    if n == 1:
        scored[0]["weight"] = round(min(1.0, max_single_weight), 4)
        return scored

    scores = {h["ticker"]: h.get("valuation_score", 0.0) for h in scored}
    min_score = min(scores.values())
    shifted = {t: s - min_score + 0.01 for t, s in scores.items()}
    total = sum(shifted.values())
    weights = {t: v / total for t, v in shifted.items()}

    capped: set[str] = set()
    for _ in range(len(weights)):
        over = {t: w for t, w in weights.items() if t not in capped and w > max_single_weight}
        if not over:
            break
        for t in over:
            capped.add(t)
            weights[t] = max_single_weight

        remaining = {t: w for t, w in weights.items() if t not in capped}
        remaining_total = sum(remaining.values())
        budget_left = 1.0 - sum(weights[t] for t in capped)
        if remaining_total > 0:
            for t in remaining:
                weights[t] = (remaining[t] / remaining_total) * budget_left

    for h in scored:
        h["weight"] = round(weights[h["ticker"]], 4)
    return scored


def allocator_and_trade_planner(state: PortfolioState) -> PortfolioState:
    """4단계: 비중 배분 + 분할 매수 계획."""
    limits = state["limits"]
    portfolio = compute_weights(state["scored_candidates"], limits["max_single_stock_weight"])

    state["portfolio"] = portfolio
    state["trade_plan"] = []

    invested_weight = sum(h["weight"] for h in portfolio)
    status = "4단계 완료: 포트폴리오 구성"
    if invested_weight < 0.99:
        cash_pct = round((1 - invested_weight) * 100, 1)
        status += (
            f" (경고: 종목 수 부족으로 예산의 {cash_pct}%가 미배분 상태 - "
            f"단일 종목 상한 {limits['max_single_stock_weight']*100:.0f}%를 지키려면 "
            f"최소 {int(1 / limits['max_single_stock_weight'])}종목이 필요합니다)"
        )
    state["status"] = status
    print(f'>>> {state["status"]}')
    return state


def rebalancing_monitor(state: PortfolioState) -> PortfolioState:
    """5단계: 정기 리밸런싱 + 손절 조건 체크."""
    current_holdings = state.get("current_holdings") or []
    target_weights = {h["ticker"]: h["weight"] for h in state["portfolio"]}

    if not current_holdings:
        state["rebalance_actions"] = [
            {"ticker": t, "action": "initial_buy", "target_weight": w}
            for t, w in target_weights.items()
        ]
        state["status"] = "5단계 완료: 기존 보유분 없음 -> 최초 매수 계획으로 대체"
        print(f'>>> {state["status"]}')
        return state

    tickers = [h["ticker"] for h in current_holdings]
    current_prices = fetch_current_prices(tickers)

    total_value = 0.0
    for h in current_holdings:
        price = current_prices.get(h["ticker"])
        h["current_price"] = price
        h["market_value"] = (price or 0.0) * h["shares"]
        h["return_pct"] = ((price / h["entry_price"]) - 1) if price and h.get("entry_price") else None
        total_value += h["market_value"]

    stop_loss_pct = state["limits"]["stop_loss_pct"]
    drift_threshold = 0.05

    actions: list[dict] = []
    for h in current_holdings:
        ticker = h["ticker"]
        current_weight = (h["market_value"] / total_value) if total_value else 0.0
        target_weight = target_weights.get(ticker, 0.0)
        drift = current_weight - target_weight

        if h["return_pct"] is not None and h["return_pct"] <= stop_loss_pct:
            actions.append({
                "ticker": ticker,
                "action": "stop_loss_review",
                "return_pct": round(h["return_pct"], 4),
                "current_weight": round(current_weight, 4),
            })
        elif abs(drift) >= drift_threshold:
            actions.append({
                "ticker": ticker,
                "action": "trim" if drift > 0 else "add",
                "current_weight": round(current_weight, 4),
                "target_weight": round(target_weight, 4),
                "drift": round(drift, 4),
            })

    for action in actions:
        try:
            result = call_llm(
                model=MODEL_TOP,
                system_prompt=(
                    "너는 신중한 포트폴리오 리밸런싱 어드바이저다. 주어진 수치만 근거로 "
                    "이 조치가 왜 제안됐는지 한국어로 한두 문장으로 설명하라. "
                    "확정적인 예측이나 과장된 표현은 쓰지 마라."
                ),
                user_prompt=json.dumps(action, ensure_ascii=False),
                json_schema=_RATIONALE_SCHEMA,
            )
            action["rationale"] = result["rationale"]
        except Exception:
            action["rationale"] = None

    state["rebalance_actions"] = actions
    state["status"] = f"5단계 완료: 리밸런싱/손절 제안 {len(actions)}건"
    print(f'>>> {state["status"]}')
    return state


def stability_gate(state: PortfolioState) -> Literal["continue", "abort"]:
    if len(state["stable_candidates"]) < state["limits"]["min_holdings"]:
        return "abort"
    return "continue"


def build_graph():
    graph = StateGraph(PortfolioState)

    graph.add_node("discover", sector_researcher_and_screener)
    graph.add_node("stability_filter", stability_screener)
    graph.add_node("valuation_risk", valuation_and_risk_analyst)
    graph.add_node("allocate", allocator_and_trade_planner)
    graph.add_node("monitor", rebalancing_monitor)

    graph.set_entry_point("discover")
    graph.add_edge("discover", "stability_filter")

    graph.add_conditional_edges(
        "stability_filter",
        stability_gate,
        {
            "continue": "valuation_risk",
            "abort": END,
        },
    )

    graph.add_edge("valuation_risk", "allocate")
    graph.add_edge("allocate", "monitor")
    graph.add_edge("monitor", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    initial_state: PortfolioState = {
        "sector": "반도체",
        "budget": 10_000.0,
        "limits": DEFAULT_LIMITS,
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

    result = app.invoke(initial_state)
    print(result["status"])
    print(f"최종 포트폴리오 종목 수: {len(result['portfolio'])}")
