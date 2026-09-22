"""
안정적 우상향(Quality Growth) 목표에 맞춘 주식 분석 에이전트 파이프라인.

전체 흐름: 종목 발굴·검증 -> 안정성 필터링 -> 밸류에이션·리스크 분석
          -> 포트폴리오 구성 -> 모니터링·리밸런싱

각 단계는 실제로는 Claude API를 호출하는 에이전트지만, 이 골격에서는
call_llm()이 플레이스홀더입니다. 실제 프롬프트/모델을 넣고 나서
API 키를 연결하면 바로 동작하는 구조로 짰습니다.
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


# ---------------------------------------------------------------------------
# 1. 모델 티어 설정
#    - 실제 모델 문자열은 Anthropic 문서(docs.claude.com)에서 최신 값을 확인하세요.
#    - 호출 빈도가 높은 단계는 저가/고속 모델, 종합 판단 단계는 최상위 모델을 씁니다.
# ---------------------------------------------------------------------------

MODEL_FAST = "claude-haiku-4-5-20251001"   # 대량 스캔, 티커 검증 등 고빈도 작업
MODEL_MID = "claude-sonnet-5"              # 밸류에이션 해석, 재무 분석 등 중간 추론
MODEL_TOP = "claude-opus-5"                # 리스크 종합, 배분, 최종 리밸런싱 판단


# ---------------------------------------------------------------------------
# 2. 안정성 기준값 - 여기 숫자들이 "안정적 우상향" 목표를 실제로 구현하는 부분입니다.
#    프롬프트 안에 이 값을 그대로 박아 넣어서 에이전트가 임의로 완화하지 못하게 하세요.
# ---------------------------------------------------------------------------

class RiskLimits(TypedDict):
    max_beta: float                 # 이 값을 넘는 종목은 안정성 필터에서 탈락
    min_years_profit_growth: int    # 최소 연속 이익 성장 연수
    max_debt_to_equity: float       # 부채비율 상한 (업종 평균 대비 배수)
    max_single_stock_weight: float  # 단일 종목 최대 비중
    max_single_sector_weight: float # 단일 섹터 최대 비중
    min_holdings: int               # 최소 보유 종목 수
    correlation_ceiling: float      # 이 상관계수 이상인 종목 쌍은 동시 편입 제한
    stop_loss_pct: float            # 매수가 대비 이 비율 하락 시 즉시 재검토
    min_market_cap: float           # 1단계 티커 스크리너 통과 기준 (중대형주 하한선)


DEFAULT_LIMITS: RiskLimits = {
    "max_beta": 1.3,
    "min_years_profit_growth": 3,
    "max_debt_to_equity": 1.0,          # 업종 평균 이하
    "max_single_stock_weight": 0.10,
    "max_single_sector_weight": 0.25,
    "min_holdings": 12,
    "correlation_ceiling": 0.7,
    "stop_loss_pct": -0.15,
    "min_market_cap": 10_000_000_000,   # 100억 달러 - 소형주/신생 산업 배제 기준
}


# ---------------------------------------------------------------------------
# 3. 파이프라인 상태 정의
#    LangGraph는 이 State 딕셔너리를 노드 간에 전달하며 계속 채워나갑니다.
# ---------------------------------------------------------------------------

class PortfolioState(TypedDict):
    sector: str                      # 사용자가 지정한 관심 섹터
    budget: float                    # 총 투자 예산
    limits: RiskLimits               # 안정성/리스크 상한값

    candidates: list[dict]           # 1단계: 발굴 + 티커 검증 통과 종목
    stable_candidates: list[dict]    # 2단계: 안정성 필터 통과 종목
    scored_candidates: list[dict]    # 3단계: 밸류에이션+리스크 스코어링 결과
    portfolio: list[dict]            # 4단계: 최종 비중까지 정해진 포트폴리오
    trade_plan: list[dict]           # 4단계: 분할 매수 주문 초안

    current_holdings: list[dict]     # 5단계 입력: 실제로 보유 중인 종목 (ticker, shares, entry_price)
    rebalance_actions: list[dict]    # 5단계 출력: 리밸런싱/손절 제안

    stage1_audit: list[dict]         # 1단계: 제안된 전체 종목 + 통과/탈락 사유 (UI용 감사 기록)
    stage2_audit: list[dict]         # 2단계: 평가된 전체 종목 + 통과/탈락 사유 (UI용 감사 기록)

    status: str                      # 파이프라인 진행 상태 메시지 (디버깅용)


# ---------------------------------------------------------------------------
# 4. call_llm은 llm_client.py에서 가져옵니다 (실제 Anthropic API 연결).
#    사용 전 ANTHROPIC_API_KEY 환경변수를 설정하세요.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. 단계별 노드 함수
#    각 함수는 "역할 하나만" 담당하고, 결과를 State에 채워서 돌려줍니다.
# ---------------------------------------------------------------------------

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
    """LLM 응답 텍스트에서 마지막 JSON 코드블록(또는 JSON 배열)을 찾아 파싱합니다."""
    matches = re.findall(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    if not matches:
        matches = re.findall(r"(\[\s*\{.*?\}\s*\])", text, re.DOTALL)
    if not matches:
        raise ValueError("응답에서 JSON 리스트를 찾지 못했습니다.")
    return json.loads(matches[-1])


def sector_researcher_and_screener(state: PortfolioState) -> PortfolioState:
    """1단계: 섹터 리서처 + 티커 스크리너. 중대형주 위주 후보군 발굴.

    1) Claude에게 웹 검색 도구를 주고 지정 섹터의 실제 중대형주 후보를 찾게 합니다.
    2) LLM이 준 티커를 그대로 믿지 않고, yfinance로 실제 존재 여부와 시가총액을
       재검증합니다 - 최종 통과/탈락 판단은 항상 실제 데이터가 내립니다.

    run_local.py에서 candidates를 미리 채워 넘기면(빠른 반복 테스트용) 이 웹 검색을
    건너뛰고 그 값을 그대로 씁니다. 실제 자동 발굴을 보려면 run_local.py의
    CANDIDATE_TICKERS를 비워두세요.
    """
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
        # 웹 검색/파싱이 실패해도 전체 파이프라인이 죽지 않게, 빈 후보로 안전하게 넘어갑니다.
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
    """2단계: 안정성 필터링 - 이번 목표의 핵심 게이트.

    1) yfinance로 실제 베타/부채비율/연속 이익성장 데이터를 가져옵니다 (코드 실행).
    2) 숫자 기준은 LLM 판단이 아니라 코드로 강제 필터링합니다 - 여기서 타협하면
       "밸류에이션은 좋은데 변동성이 큰 종목"이 다음 단계로 새어 나갑니다.
    3) 통과한 종목에 대해서만 Sonnet으로 짧은 근거 문장을 덧붙입니다.
    """
    limits = state["limits"]
    tickers = [c["ticker"] for c in state["candidates"]]

    fetched, failed = fetch_fundamentals_batch(tickers)

    import json
    print("전체 조회 결과:", json.dumps(fetched, indent=2, ensure_ascii=False))

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

    # 데이터 조회 자체에 실패한 종목도 감사 기록에 남깁니다 (자동 탈락 처리).
    for ticker in failed:
        audit.append({
            "ticker": ticker, "beta": None, "debt_to_equity": None, "profit_growth_years": None,
            "passed": False, "failed_criteria": ["데이터 조회 실패"],
        })

    # 통과 종목마다 짧은 근거를 생성합니다.
    # 종목 수가 많아지면 비용 절감을 위해 여러 종목을 한 번의 호출로 묶는 걸 추천합니다.
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
            # LLM 근거 생성이 실패해도 필터링 결과 자체는 유지합니다.
            stock["stability_rationale"] = None

    # audit에도 rationale을 동기화 (passed 리스트와 같은 dict 객체를 공유하므로 이미 반영됨)
    state["stable_candidates"] = passed
    state["stage2_audit"] = audit
    state["status"] = (
        f"2단계 완료: 안정성 필터 통과 {len(passed)}종목 "
        f"(데이터 조회 실패로 자동 탈락 {len(failed)}종목: {failed})"
    )
    print(f'>>> {state["status"]}')
    return state


def valuation_and_risk_analyst(state: PortfolioState) -> PortfolioState:
    """3단계: 밸류에이션 스코어링 + 상관관계 기반 리스크 분석.

    1) PER/PBR/PEG/ROE를 가져와 다중 팩터 점수(valuation_score)를 계산합니다.
    2) 최근 1년 일별 수익률로 상관관계 매트릭스를 만들고, 상관계수가
       correlation_ceiling 이상인 종목 쌍을 찾아둡니다 (배분 단계에서
       "이 둘을 동시에 많이 담지 말라"는 근거로 씁니다).
    """
    stable = state["stable_candidates"]
    tickers = [c["ticker"] for c in stable]

    # --- 밸류에이션 스코어링 ---
    valuation_metrics = fetch_valuation_batch(tickers)
    valuation_metrics = compute_valuation_scores(valuation_metrics)
    valuation_by_ticker = {m["ticker"]: m for m in valuation_metrics}

    scored = []
    for c in stable:
        v = valuation_by_ticker.get(c["ticker"], {})
        merged = {**c, **{k: val for k, val in v.items() if k != "ticker"}}
        scored.append(merged)

    # --- 상관관계 기반 리스크 분석 ---
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
            # 가격 히스토리 조회 실패는 전체 파이프라인을 막을 정도는 아니므로
            # 상관관계 분석만 건너뛰고 계속 진행합니다.
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
    """valuation_score를 상대적 비중으로 변환하고 단일 종목 상한을 적용합니다.

    - 종목이 1개뿐이면 비교 대상이 없으므로 상한값 그대로를 비중으로 씁니다
      (실전에서는 min_holdings 게이트가 이 상황 자체를 막아줍니다).
    - 점수가 높을수록 비중을 더 주되, 상한을 넘는 초과분은 나머지 종목에
      비례 재분배합니다 (waterfall capping).
    """
    n = len(scored)
    if n == 0:
        return scored
    if n == 1:
        scored[0]["weight"] = round(min(1.0, max_single_weight), 4)
        return scored

    scores = {h["ticker"]: h.get("valuation_score", 0.0) for h in scored}
    min_score = min(scores.values())
    # 전부 양수로 이동시켜서 비중 계산이 가능하게 함 (0.01은 최저점 종목도 소액은 배분받게 하는 여유값)
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
    """4단계: 비중 배분 + 분할 매수 계획. 비중 상한 규칙을 강제 적용."""
    limits = state["limits"]
    portfolio = compute_weights(state["scored_candidates"], limits["max_single_stock_weight"])

    state["portfolio"] = portfolio
    # TODO: call_llm(MODEL_MID, ...) 로 분할 매수(예: 3회 분할) 주문 초안 작성
    state["trade_plan"] = []

    invested_weight = sum(h["weight"] for h in portfolio)
    status = "4단계 완료: 포트폴리오 구성"
    if invested_weight < 0.99:
        # 종목 수 × 단일 종목 상한 < 100% 인 경우 발생. 버그가 아니라
        # "상한을 지키려면 이 종목 수로는 예산을 다 못 채운다"는 신호입니다.
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
    """5단계: 정기 리밸런싱 + 손절 조건 체크.

    - 손절 여부와 비중 이탈(drift) 판단은 LLM이 아니라 코드로 계산합니다.
      가격 트리거를 LLM 판단에 맡기면 같은 숫자를 두고도 실행마다 다른 결론이
      나올 수 있어서, 여기서는 숫자 계산은 100% 코드가 하고 Claude는 "왜"만 설명합니다.
    - current_holdings가 비어 있으면(아직 실제로 아무것도 안 산 상태) 4단계에서 나온
      목표 비중을 그대로 "최초 매수 계획"으로 돌려줍니다.
    """
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
    drift_threshold = 0.05  # 목표 비중과 5%포인트 이상 벌어지면 리밸런싱 대상으로 표시

    actions: list[dict] = []
    for h in current_holdings:
        ticker = h["ticker"]
        current_weight = (h["market_value"] / total_value) if total_value else 0.0
        target_weight = target_weights.get(ticker, 0.0)
        drift = current_weight - target_weight

        if h["return_pct"] is not None and h["return_pct"] <= stop_loss_pct:
            # 손절 조건이 비중 이탈보다 우선순위가 높습니다 - 리스크 신호가 더 급함
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

    # 판단(숫자)은 이미 끝났으니, 각 액션에 대한 설명 한 줄만 Claude가 생성합니다.
    import json as _json
    for action in actions:
        try:
            result = call_llm(
                model=MODEL_TOP,
                system_prompt=(
                    "너는 신중한 포트폴리오 리밸런싱 어드바이저다. 주어진 수치만 근거로 "
                    "이 조치가 왜 제안됐는지 한국어로 한두 문장으로 설명하라. "
                    "확정적인 예측이나 과장된 표현은 쓰지 마라."
                ),
                user_prompt=_json.dumps(action, ensure_ascii=False),
                json_schema=_RATIONALE_SCHEMA,
            )
            action["rationale"] = result["rationale"]
        except Exception:
            action["rationale"] = None

    state["rebalance_actions"] = actions
    state["status"] = f"5단계 완료: 리밸런싱/손절 제안 {len(actions)}건"
    print(f'>>> {state["status"]}')
    return state


# ---------------------------------------------------------------------------
# 6. 조건부 게이트
#    안정성 필터를 통과한 종목이 하나도 없으면 뒷단으로 넘어가지 않고 종료합니다.
#    이게 "밸류에이션은 좋은데 변동성이 큰 종목"이 살아남는 걸 막는 하드 게이트입니다.
# ---------------------------------------------------------------------------

def stability_gate(state: PortfolioState) -> Literal["continue", "abort"]:
    if len(state["stable_candidates"]) < state["limits"]["min_holdings"]:
        return "abort"
    return "continue"


# ---------------------------------------------------------------------------
# 7. 그래프 조립
# ---------------------------------------------------------------------------

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
            "abort": END,  # 후보가 최소 종목 수에 못 미치면 여기서 중단
        },
    )

    graph.add_edge("valuation_risk", "allocate")
    graph.add_edge("allocate", "monitor")
    graph.add_edge("monitor", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 8. 실행 예시
# ---------------------------------------------------------------------------

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
        "status": "시작 전",
    }

    result = app.invoke(initial_state)
    print(result["status"])
    print(f"최종 포트폴리오 종목 수: {len(result['portfolio'])}")

    import json
    print(json.dumps(result["portfolio"], indent=2, ensure_ascii=False))

    print()
    print(f"리밸런싱/손절 제안 상세 ({len(result['rebalance_actions'])}건):")
    print(json.dumps(result["rebalance_actions"], indent=2, ensure_ascii=False))