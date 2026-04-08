# 동네 엑스레이 AI 에이전트

LangGraph + OpenAI 기반 대화형 데이터 분석 에이전트

## 배포 URL

- **Agent API**: https://agent-3lht.onrender.com
- **Dashboard**: (Streamlit Cloud 배포 후 업데이트)
- **API Health**: https://agent-3lht.onrender.com/health
- **API Docs**: https://agent-3lht.onrender.com/docs

## 구조

```
agent/
├── agent/
│   ├── graph.py          # LangGraph StateGraph (메인 그래프)
│   ├── prompts.py        # 프롬프트 템플릿
│   ├── cost_tracker.py   # 비용 추적 모듈
│   ├── nodes/            # 노드 확장용
│   └── tools/
│       └── data_tools.py # DuckDB 기반 데이터 도구 8개
├── tests/
├── requirements.txt
└── .env.example
```

## 3-Tier 모델 라우팅

| Tier | 모델 | 용도 | 건당 비용 |
|------|------|------|----------|
| Router | gpt-4.1-nano | 질문 분류 | ~$0.00007 |
| Tier 1 | gpt-4.1-nano | 단순 조회 | ~$0.0002 |
| Tier 2 | gpt-4.1-mini | 분석/비교 | ~$0.004 |
| Tier 3 | gpt-4.1 | 복합 추론 | ~$0.02 |

## 설정

```bash
cp .env.example .env
# .env에 OPENAI_API_KEY 설정
pip install -r requirements.txt
```

## 사용

```python
from agent.graph import agent

result = agent.invoke({
    "messages": [("user", "신당동 유동인구 알려줘")],
    "selected_district": "신당동",
    "selected_month": "202506",
    "current_tab": "프로파일",
    "cost_log": [],
})

print(result["response_text"])
print(f"비용: ${sum(c['cost'] for c in result['cost_log']):.4f}")
```
