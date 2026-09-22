"""
가장 먼저 실행해볼 스크립트.
API 키가 제대로 설정됐는지, anthropic 패키지가 잘 깔렸는지만 빠르게 확인합니다.
여기가 통과해야 orchestration.py 전체를 돌리는 의미가 있습니다.
"""

import os

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ ANTHROPIC_API_KEY 환경변수가 설정되어 있지 않습니다.")
    print("   터미널에서 export ANTHROPIC_API_KEY='sk-ant-...' 를 먼저 실행하세요.")
    raise SystemExit(1)

print(f"✅ 환경변수 확인됨 (키 앞부분: {api_key[:12]}...)")

from anthropic import Anthropic

client = Anthropic(api_key=api_key)

print("Claude API에 실제로 요청을 보내는 중...")
response = client.messages.create(
    model="claude-haiku-4-5-20251001",  # 연결 테스트는 가장 저렴한 모델로
    max_tokens=50,
    messages=[{"role": "user", "content": "한국어로 짧게 인사해줘"}],
)

text = next(b.text for b in response.content if b.type == "text")
print("✅ API 응답 성공!")
print(f"   Claude의 응답: {text}")
print(f"   사용된 토큰: input={response.usage.input_tokens}, output={response.usage.output_tokens}")
