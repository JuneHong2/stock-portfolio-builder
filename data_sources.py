"""
안정성 필터링에 필요한 재무/변동성 데이터를 Yahoo Finance(yfinance)에서 가져옵니다.

주의: 이 모듈은 실제 네트워크 요청(query1/query2.finance.yahoo.com)을 합니다.
      샌드박스처럼 아웃바운드 도메인이 제한된 환경에서는 동작하지 않을 수 있으니,
      본인 로컬 환경이나 서버에서 실행하세요.
"""

from __future__ import annotations

import time

import pandas as pd
import yfinance as yf


class FundamentalsUnavailable(Exception):
    """데이터를 가져오지 못했을 때. 안정성 필터에서는 이런 종목을 기본적으로 탈락시키세요."""


def fetch_company_detail(ticker: str) -> dict:
    """UI 상세보기용 부가 정보 - 풀네임, 섹터, 배당, 애널리스트 목표주가 컨센서스 등.

    analyst_target_* 필드는 Yahoo Finance가 집계한 제3자 애널리스트 컨센서스입니다.
    Claude나 이 파이프라인이 만든 예측이 아니며, 실제 주가를 보장하지 않습니다.
    """
    info = yf.Ticker(ticker).info
    return {
        "ticker": ticker,
        "long_name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "current_price": info.get("regularMarketPrice") or info.get("previousClose"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "dividend_yield": info.get("dividendYield"),
        "analyst_target_low": info.get("targetLowPrice"),
        "analyst_target_mean": info.get("targetMeanPrice"),
        "analyst_target_high": info.get("targetHighPrice"),
        "analyst_recommendation": info.get("recommendationKey"),
        "num_analyst_opinions": info.get("numberOfAnalystOpinions"),
        "business_summary": info.get("longBusinessSummary"),
    }


def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """차트용 종가 히스토리. 실패하면 빈 DataFrame을 돌려줍니다 (UI가 알아서 숨기게)."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
        return hist[["Close"]] if not hist.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def verify_ticker_and_market_cap(ticker: str, min_market_cap: float) -> dict | None:
    """티커가 실제로 존재하고 시가총액이 기준 이상인지 확인합니다.

    LLM이 제안한 후보는 티커를 잘못 알거나(예: 상장폐지, 오타) 아예 지어낼 수도 있으므로,
    반드시 이 함수로 실제 시장 데이터와 대조한 뒤에만 다음 단계로 넘깁니다.
    """
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get("regularMarketPrice") is None:
            return None
        market_cap = info.get("marketCap")
        if market_cap is None or market_cap < min_market_cap:
            return None
        return {"ticker": ticker, "market_cap": market_cap}
    except Exception:
        return None


def fetch_fundamentals(ticker: str, retries: int = 2, retry_wait: float = 1.5) -> dict:
    """한 종목의 베타, 부채비율, 연속 이익성장 연수, 시가총액을 가져옵니다.

    Returns:
        {
            "ticker": str,
            "beta": float | None,
            "debt_to_equity": float | None,   # 0.5 = 부채가 자기자본의 50%
            "profit_growth_years": int,        # 최근부터 몇 년 연속 순이익이 늘었는지
            "market_cap": float | None,
        }
    """
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            t = yf.Ticker(ticker)
            info = t.info

            if not info or info.get("regularMarketPrice") is None:
                raise FundamentalsUnavailable(f"{ticker}: 유효한 시세 정보 없음 (상장폐지/오타 가능성)")

            debt_to_equity_raw = info.get("debtToEquity")  # yfinance는 %단위로 줌 (예: 45.2 -> 45.2%)
            debt_to_equity = (debt_to_equity_raw / 100) if debt_to_equity_raw is not None else None

            profit_growth_years = _consecutive_profit_growth_years(t.financials)

            return {
                "ticker": ticker,
                "beta": info.get("beta"),
                "debt_to_equity": debt_to_equity,
                "profit_growth_years": profit_growth_years,
                "market_cap": info.get("marketCap"),
            }

        except FundamentalsUnavailable:
            raise
        except Exception as e:  # 네트워크 오류, 일시적 API 오류 등
            last_error = e
            if attempt < retries:
                time.sleep(retry_wait * (attempt + 1))

    raise FundamentalsUnavailable(f"{ticker}: 데이터 조회 {retries + 1}회 모두 실패 ({last_error})")


def _consecutive_profit_growth_years(financials: pd.DataFrame) -> int:
    """연간 손익계산서에서 순이익이 최근 연도부터 몇 년 연속 증가했는지 셉니다.

    yfinance의 t.financials는 컬럼이 회계연도, 행이 계정과목인 DataFrame입니다.
    컬럼(연도) 순서가 보장되지 않을 수 있어 날짜 기준으로 명시적으로 정렬합니다.
    """
    if financials is None or financials.empty or "Net Income" not in financials.index:
        return 0

    net_income = financials.loc["Net Income"]
    net_income = net_income.reindex(sorted(net_income.index, reverse=True))  # 최신 -> 과거
    values = list(net_income.dropna())

    if len(values) < 2:
        return 0

    streak = 0
    for newer, older in zip(values, values[1:]):
        if newer > older:
            streak += 1
        else:
            break
    return streak


def fetch_valuation_metrics(ticker: str) -> dict:
    """PER, PBR, PEG, ROE를 가져옵니다. 값이 없으면 None으로 채웁니다."""
    info = yf.Ticker(ticker).info
    return {
        "ticker": ticker,
        "per": info.get("trailingPE"),
        "pbr": info.get("priceToBook"),
        "peg": info.get("pegRatio") or info.get("trailingPegRatio"),
        "roe": info.get("returnOnEquity"),
    }


def fetch_valuation_batch(tickers: list[str]) -> list[dict]:
    results = []
    for ticker in tickers:
        try:
            results.append(fetch_valuation_metrics(ticker))
        except Exception:
            results.append({"ticker": ticker, "per": None, "pbr": None, "peg": None, "roe": None})
    return results


def compute_valuation_scores(metrics: list[dict]) -> list[dict]:
    """PER/PBR/PEG는 낮을수록, ROE는 높을수록 좋다고 보고 z-score를 합산합니다.

    값이 없는(None) 종목은 해당 지표에서 중립(0)으로 처리합니다 - 데이터 결측이
    자동으로 유리하거나 불리하게 작용하지 않게 하기 위함입니다.
    """
    if not metrics:
        return metrics

    lower_is_better = ["per", "pbr", "peg"]
    higher_is_better = ["roe"]

    for field in lower_is_better + higher_is_better:
        values = [m[field] for m in metrics if m.get(field) is not None]
        if len(values) < 2:
            continue  # 비교할 데이터가 부족하면 이 지표는 스코어링에서 건너뜁니다
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stdev = variance ** 0.5 or 1e-9
        sign = -1 if field in lower_is_better else 1
        for m in metrics:
            if m.get(field) is not None:
                m.setdefault("_z_sum", 0.0)
                m["_z_sum"] += sign * (m[field] - mean) / stdev

    for m in metrics:
        m["valuation_score"] = round(m.pop("_z_sum", 0.0), 3)

    return metrics


def fetch_price_returns(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """일별 수익률 DataFrame (컬럼=티커, 인덱스=날짜)을 반환합니다."""
    if not tickers:
        return pd.DataFrame()

    data = yf.download(tickers, period=period, progress=False)["Close"]
    if isinstance(data, pd.Series):  # 티커가 1개뿐이면 Series로 반환됨
        data = data.to_frame(name=tickers[0])
    return data.pct_change(fill_method=None).dropna(how="all")


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def find_high_correlation_pairs(
    corr: pd.DataFrame, ceiling: float
) -> list[tuple[str, str, float]]:
    """상관계수가 ceiling 이상인 종목 쌍을 찾습니다 (자기 자신과의 상관관계=1은 제외)."""
    pairs = []
    tickers = corr.columns.tolist()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            value = corr.iloc[i, j]
            if pd.notna(value) and value >= ceiling:
                pairs.append((tickers[i], tickers[j], round(float(value), 3)))
    return pairs


def fetch_current_prices(tickers: list[str]) -> dict[str, float | None]:
    """각 티커의 현재가(또는 전일 종가)를 가져옵니다. 리밸런싱 모니터가 사용합니다."""
    prices: dict[str, float | None] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            prices[ticker] = info.get("regularMarketPrice") or info.get("previousClose")
        except Exception:
            prices[ticker] = None
    return prices


def fetch_fundamentals_batch(tickers: list[str]) -> tuple[list[dict], list[str]]:
    """여러 종목을 순회하며 조회합니다.

    Returns:
        (성공한 종목들의 데이터 리스트, 실패한 티커 리스트)
        실패한 티커는 안정성 필터에서 자동 탈락 처리하는 게 안전합니다
        (데이터가 없다는 것 자체를 리스크 신호로 취급).
    """
    results: list[dict] = []
    failed: list[str] = []

    for ticker in tickers:
        try:
            results.append(fetch_fundamentals(ticker))
        except FundamentalsUnavailable:
            failed.append(ticker)

    return results, failed