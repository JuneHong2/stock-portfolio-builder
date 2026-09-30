"""
SnapTrade Personal API를 통해 실제 Wealthsimple 계좌의 보유 종목을 가져옵니다.

사용 전 준비:
- pip install snaptrade-python-sdk
- 환경변수(또는 Streamlit secrets): SNAPTRADE_CLIENT_ID, SNAPTRADE_CONSUMER_KEY
  (SnapTrade Dashboard > API Keys 화면에서 확인한 Client ID / Consumer Key)

중요 - 의도적인 설계 원칙:
- 이 모듈은 읽기 전용 엔드포인트(계좌 목록, 포지션 조회)만 사용합니다.
- 거래 실행(trading) 관련 함수는 여기에 아예 만들지 않습니다.
  실수로라도 이 파일을 통해 매매 주문이 나가는 일이 없게 하기 위한 설계입니다.
- Personal API key 모드이므로 SnapTrade user_id/user_secret은 필요 없습니다
  (Client ID/Consumer Key 자체가 계정 본인을 식별합니다).
"""

from __future__ import annotations

import json
import os

from snaptrade_client import SnapTrade, SnapTradeAuth


def _get_client() -> SnapTrade:
    client_id = os.environ.get("SNAPTRADE_CLIENT_ID")
    consumer_key = os.environ.get("SNAPTRADE_CONSUMER_KEY")

    if not client_id or not consumer_key:
        raise RuntimeError(
            "SNAPTRADE_CLIENT_ID / SNAPTRADE_CONSUMER_KEY 환경변수(또는 Streamlit secrets)가 "
            "설정되어 있지 않습니다."
        )

    return SnapTrade(
        auth=SnapTradeAuth.personal_api_key(
            consumer_key=consumer_key,
            client_id=client_id,
        )
    )


def _as_parsed(body):
    """SnapTrade 응답의 .body가 이미 파싱된 리스트/딕셔너리면 그대로,
    혹시 JSON 문자열로 온 경우엔 파싱해서 돌려줍니다.

    SDK 버전/설정에 따라 .body가 str로 올 수도 있어서, 이 함수 없이
    바로 .get()을 호출하면 "'str' object has no attribute 'get'" 에러가 납니다.
    """
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    if isinstance(body, str):
        return json.loads(body)
    return body


def fetch_wealthsimple_holdings() -> tuple[list[dict], list[dict]]:
    """연결된 모든 계좌(Wealthsimple non-registered 포함)의 보유 종목을 가져옵니다.

    반환값: (holdings, unparsed)
    - holdings: current_holdings 형식의 정상 파싱된 종목 리스트
        [{"ticker": str, "shares": float, "entry_price": float | None, "account": str}, ...]
    - unparsed: 티커를 못 뽑아낸 원본 포지션들 [{"account": str, "raw": dict}, ...]
      (비어있지 않으면 화면에 원본을 보여줘서 파싱 로직을 보정할 근거로 씁니다)

    - 현금성 포지션(cash_equivalent)은 제외합니다 - 이 파이프라인은 주식/ETF만 다룹니다.
    - entry_price는 SnapTrade가 주는 총 매입원가(cost_basis)를 수량으로 나눠 계산합니다.
    """
    client = _get_client()

    accounts_response = client.account_information.list_user_accounts()
    accounts = _as_parsed(accounts_response.body) or []

    if not isinstance(accounts, list):
        raise RuntimeError(
            f"계좌 목록 응답이 예상과 다른 형식입니다 (type={type(accounts).__name__}). "
            f"원본 일부: {str(accounts)[:300]}"
        )

    holdings: list[dict] = []
    unparsed: list[dict] = []

    for account in accounts:
        if not isinstance(account, dict):
            print(f"경고: 계좌 항목이 dict가 아님 (type={type(account).__name__}): {account}")
            continue

        account_id = account.get("id")
        account_label = account.get("name") or account.get("number") or account_id
        if not account_id:
            continue

        try:
            positions_response = client.account_information.get_all_account_positions(
                account_id=account_id
            )
        except Exception as e:
            print(f"경고: 계좌 '{account_label}' 포지션 조회 실패: {e}")
            continue

        positions_body = _as_parsed(positions_response.body) or {}
        if isinstance(positions_body, list):
            # 일부 응답 형태는 {"results": [...]}가 아니라 바로 리스트로 올 수 있음
            positions = positions_body
        elif isinstance(positions_body, dict):
            positions = positions_body.get("results", [])
        else:
            print(
                f"경고: 계좌 '{account_label}' 포지션 응답이 예상과 다른 형식"
                f"(type={type(positions_body).__name__}): {str(positions_body)[:300]}"
            )
            continue

        for pos in positions:
            if not isinstance(pos, dict):
                print(f"경고: 포지션 항목이 dict가 아님 (계좌 '{account_label}', type={type(pos).__name__}): {pos}")
                continue

            if pos.get("cash_equivalent"):
                continue

            ticker = _extract_ticker(pos)
            units = _to_float(pos.get("units"))
            cost_basis = _to_float(pos.get("cost_basis"))

            if not ticker or not units:
                unparsed.append({"account": account_label, "raw": pos})
                continue

            entry_price = (cost_basis / units) if cost_basis and units else None

            holdings.append({
                "ticker": ticker,
                "shares": units,
                "entry_price": entry_price,
                "account": account_label,
            })

    return holdings, unparsed


def _to_float(value) -> float | None:
    """SnapTrade가 숫자를 문자열로 줄 때가 있어서, 안전하게 float로 변환합니다."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_ticker(pos: dict) -> str | None:
    """포지션에서 티커를 최대한 유연하게 뽑아냅니다.

    SnapTrade가 브로커리지마다(그리고 엔드포인트마다) symbol을 문자열로 주기도 하고
    중첩된 객체로 주기도 해서, 두 경우 다 대응하도록 여러 경로를 순서대로 시도합니다.
    """
    instrument = pos.get("instrument")

    # 경우 1: instrument가 아예 문자열(티커 자체)인 경우
    if isinstance(instrument, str):
        return instrument

    if isinstance(instrument, dict):
        symbol_field = instrument.get("symbol")
        # 경우 2: instrument.symbol이 문자열인 경우
        if isinstance(symbol_field, str):
            return symbol_field
        # 경우 3: instrument.symbol이 중첩 객체인 경우
        if isinstance(symbol_field, dict):
            ticker = symbol_field.get("symbol") or symbol_field.get("raw_symbol")
            if isinstance(ticker, str):
                return ticker
        # 경우 4: instrument 바로 아래에 raw_symbol이 있는 경우
        raw = instrument.get("raw_symbol")
        if isinstance(raw, str):
            return raw

    # 경우 5: instrument 없이 최상위에 바로 symbol/ticker가 있는 경우
    for key in ("symbol", "ticker", "raw_symbol"):
        value = pos.get(key)
        if isinstance(value, str):
            return value

    return None
