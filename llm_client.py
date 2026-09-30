"""
Claude API 호출 래퍼.

- ANTHROPIC_API_KEY 환경변수가 설정되어 있어야 합니다.
- json_schema를 넘기면 output_config로 스키마를 강제해서 dict를 돌려줍니다.
- json_schema를 안 넘기면 그냥 텍스트(str)를 돌려줍니다.
"""

from __future__ import annotations

import json
import os
import time

from anthropic import Anthropic, APIStatusError, APIConnectionError

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY 환경변수가 설정되어 있지 않습니다. "
                "터미널에서 export ANTHROPIC_API_KEY=sk-ant-... 로 설정하세요."
            )
        _client = Anthropic(api_key=api_key)
    return _client


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: dict | None = None,
    max_tokens: int = 2048,
    max_retries: int = 3,
) -> str | dict:
    """Claude API를 호출하고 결과를 돌려줍니다."""
    client = _get_client()

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if json_schema is not None:
        kwargs["output_config"] = {
            "format": {"type": "json_schema", "schema": json_schema}
        }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(**kwargs)
            text_block = next(b for b in response.content if b.type == "text")

            if json_schema is not None:
                return json.loads(text_block.text)
            return text_block.text

        except (APIStatusError, APIConnectionError) as e:
            last_error = e
            wait_seconds = 2 ** attempt
            time.sleep(wait_seconds)
        except (json.JSONDecodeError, StopIteration) as e:
            raise RuntimeError(f"응답 파싱 실패 (모델={model}): {e}") from e

    raise RuntimeError(f"call_llm 재시도 {max_retries}회 모두 실패 (모델={model}): {last_error}")


def call_llm_with_web_search(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_searches: int = 5,
    max_tokens: int = 3000,
) -> str:
    """웹 검색 도구를 켜고 Claude를 호출합니다. 최종 텍스트를 그대로 돌려줍니다."""
    client = _get_client()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": max_searches}],
    )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    return "\n".join(text_blocks)
