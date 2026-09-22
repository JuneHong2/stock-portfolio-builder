"""
개인 테스트용 실행 스크립트.

이 파일에 있는 값들(종목 리스트, min_holdings, 보유종목)은 orchestration.py를
새로 받아서 교체하셔도 사라지지 않습니다. orchestration.py는 "로직"만 담당하고,
"내가 지금 뭘 테스트하고 싶은지"는 이 파일에서만 관리하세요.

실행: python3 run_local.py
"""

import json

from orchestration import build_graph, DEFAULT_LIMITS

# ── 여기부터 자유롭게 수정하세요 ──────────────────────────────────────

# 비워두면(빈 리스트) 1단계가 실제로 웹 검색을 해서 종목을 자동 발굴합니다.
# 웹 검색 없이 빠르게 뒷단(2~5단계)만 반복 테스트하고 싶으면, 여기에 티커를
# 직접 채워 넣으세요 - 그러면 1단계가 검색을 건너뛰고 이 목록을 그대로 씁니다.
CANDIDATE_TICKERS: list[str] = [
    # "AAPL", "MSFT", "GOOGL", ...
]

# 실제 발굴을 테스트할 때 어떤 섹터를 찾아볼지 (CANDIDATE_TICKERS가 비어있을 때만 사용됨)
SECTOR_LABEL = "헬스케어"

# 테스트 중엔 낮춰서 전체 흐름을 확인하고, 실전에서는 12 이상(기본값)을 권장합니다.
MIN_HOLDINGS_OVERRIDE = 6

# 실제로 보유 중인 종목이 있다면 여기 채워 넣으세요.
# 비워두면 5단계가 "최초 매수 계획"을 대신 보여줍니다.
CURRENT_HOLDINGS = [
    # {"ticker": "MSFT", "shares": 10, "entry_price": 400.0},
]

BUDGET = 10_000.0

# ── 여기까지만 수정하면 됩니다 ────────────────────────────────────────


def main() -> None:
    limits = dict(DEFAULT_LIMITS)
    limits["min_holdings"] = MIN_HOLDINGS_OVERRIDE

    initial_state = {
        "sector": SECTOR_LABEL,
        "budget": BUDGET,
        "limits": limits,
        "candidates": [{"ticker": t} for t in CANDIDATE_TICKERS],
        "stable_candidates": [],
        "scored_candidates": [],
        "portfolio": [],
        "trade_plan": [],
        "current_holdings": CURRENT_HOLDINGS,
        "rebalance_actions": [],
        "stage1_audit": [],
        "stage2_audit": [],
        "status": "시작 전",
    }

    app = build_graph()
    result = app.invoke(initial_state)

    print()
    print("=" * 60)
    print(result["status"])
    print(f"최종 포트폴리오 종목 수: {len(result['portfolio'])}")
    print(json.dumps(result["portfolio"], indent=2, ensure_ascii=False))

    print()
    print(f"리밸런싱/손절 제안 상세 ({len(result['rebalance_actions'])}건):")
    print(json.dumps(result["rebalance_actions"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()