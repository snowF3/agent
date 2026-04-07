# AI 에이전트 기획안 — LangGraph + OpenAI

## 1. 개요

**목표**: 대시보드에 통합된 대화형 AI 에이전트. 사용자가 자연어로 질문하면 데이터를 분석하여 차트 + 인사이트를 반환.

**프레임워크**: LangGraph v0.3+ (StateGraph 기반)
**LLM**: OpenAI GPT 모델 (비용 최적화 라우팅)

---

## 2. 비용 전략: 3-Tier 모델 라우팅

### 핵심 원칙: "싼 모델로 시작, 필요할 때만 비싼 모델"

```
┌─────────────────────────────────────────────────────────────┐
│                    사용자 질문 입력                           │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
              ┌─── 라우터 (nano) ───┐
              │  질문 복잡도 분류     │  ← gpt-5.4-nano ($0.20/$1.25)
              └──┬────┬────┬────────┘
                 │    │    │
     ┌───────────┘    │    └───────────┐
     ▼                ▼                ▼
 [Tier 1]         [Tier 2]         [Tier 3]
 단순 조회         분석/비교         복합 추론
 gpt-5.4-nano    gpt-5.4-mini     gpt-5.4
 $0.20/$1.25     $0.75/$4.50      $2.50/$15.00
                                                   
 "신당동 인구?"   "중구 vs 서초구"   "카페 차리면
  → SQL 1개       → SQL 2~3개       매출 얼마?"
  → 숫자 답변     → 비교 테이블      → 시뮬레이션
                  → 차트 생성        → 인사이트
```

### 모델별 역할

| Tier | 모델 | Input/Output 단가 | 용도 | 예상 토큰 | 건당 비용 |
|------|------|-------------------|------|----------|----------|
| Router | gpt-5.4-nano | $0.20/$1.25 | 질문 분류 (3줄 응답) | ~200 in / ~20 out | **~$0.00007** |
| Tier 1 | gpt-5.4-nano | $0.20/$1.25 | 단순 조회 SQL + 답변 | ~500 in / ~100 out | **~$0.0002** |
| Tier 2 | gpt-5.4-mini | $0.75/$4.50 | 분석/비교/차트 생성 | ~2000 in / ~500 out | **~$0.004** |
| Tier 3 | gpt-5.4 | $2.50/$15.00 | 복합 추론, 시뮬레이션 해석 | ~3000 in / ~1000 out | **~$0.02** |

### 예상 월간 비용 (일 50건 질의 기준)

```
질의 분포: Tier1 60% + Tier2 30% + Tier3 10% = 50건/일

일간:
  Router:  50건 × $0.00007  = $0.0035
  Tier 1:  30건 × $0.0002   = $0.006
  Tier 2:  15건 × $0.004    = $0.06
  Tier 3:   5건 × $0.02     = $0.10
  ─────────────────────────────
  일 합계:                    ≈ $0.17

월간 (22일): $0.17 × 22 = $3.74/월

캐시 적용 시 (반복 질문 40% 할인):
  → 약 $2.24/월
```

### 비용 추적 구현

```python
# 모든 LLM 호출에 콜백으로 비용 추적
class CostTracker:
    PRICING = {
        "gpt-5.4-nano":  {"input": 0.20, "output": 1.25, "cached": 0.02},
        "gpt-5.4-mini":  {"input": 0.75, "output": 4.50, "cached": 0.075},
        "gpt-5.4":       {"input": 2.50, "output": 15.00, "cached": 0.25},
    }
    
    def track(self, model, input_tokens, output_tokens, cached_tokens=0):
        p = self.PRICING[model]
        cost = (
            (input_tokens - cached_tokens) / 1_000_000 * p["input"]
            + cached_tokens / 1_000_000 * p["cached"]
            + output_tokens / 1_000_000 * p["output"]
        )
        self.session_total += cost
        self.history.append({
            "model": model, "input": input_tokens,
            "output": output_tokens, "cost": cost,
            "timestamp": datetime.now()
        })
        return cost
```

---

## 3. LangGraph 아키텍처

### StateGraph 설계

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from operator import add

class AgentState(TypedDict):
    # 대화 히스토리
    messages: Annotated[list, add]
    
    # 질문 분류 결과
    query_type: str          # "simple" | "analysis" | "complex"
    query_intent: str        # "lookup" | "compare" | "trend" | "simulate" | "recommend"
    
    # SQL 실행 결과
    sql_query: str
    sql_result: dict
    
    # 대시보드 컨텍스트
    current_tab: str         # 현재 보고 있는 탭
    selected_district: str   # 선택된 법정동
    selected_month: str      # 선택된 기준월
    
    # 응답
    response_text: str
    chart_data: dict         # 차트 렌더링용 데이터
    
    # 비용 추적
    cost_log: Annotated[list, add]
```

### 그래프 구조

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         ▼
                ┌────────────────┐
                │  context_node  │  대시보드 컨텍스트 수집
                │  (no LLM)     │  (현재 탭, 선택 지역, 기준월)
                └────────┬──────┘
                         ▼
                ┌────────────────┐
                │  router_node   │  질문 복잡도 + 의도 분류
                │  (gpt-5.4-nano)│
                └───┬────┬───┬──┘
                    │    │   │
        ┌───────────┘    │   └───────────┐
        ▼                ▼               ▼
  ┌───────────┐  ┌────────────┐  ┌────────────┐
  │ simple    │  │ analysis   │  │ complex    │
  │ _node     │  │ _node      │  │ _node      │
  │(nano)     │  │(mini)      │  │(gpt-5.4)   │
  └─────┬─────┘  └─────┬──────┘  └─────┬──────┘
        │               │               │
        ▼               ▼               ▼
  ┌───────────┐  ┌────────────┐  ┌────────────┐
  │ sql_node  │  │ sql_node   │  │ simulate   │
  │ (실행)    │  │ (실행)     │  │ _node      │
  └─────┬─────┘  └─────┬──────┘  └─────┬──────┘
        │               │               │
        └───────┬───────┘───────┬───────┘
                ▼               
        ┌────────────────┐
        │ response_node  │  최종 응답 포맷팅
        │ (same tier)    │  (텍스트 + 차트 데이터)
        └────────┬───────┘
                 ▼
        ┌────────────────┐
        │  cost_node     │  비용 기록 (no LLM)
        │  (tracker)     │
        └────────┬───────┘
                 ▼
              ┌──────┐
              │  END │
              └──────┘
```

### 노드 상세 설계

#### 1. Router Node (gpt-5.4-nano)

```python
ROUTER_PROMPT = """당신은 질문 분류기입니다. 아래 질문을 분류하세요.

query_type:
- simple: 단일 지역/지표 조회 ("신당동 인구 알려줘")
- analysis: 비교/트렌드/패턴 분석 ("중구 vs 서초구 매출 비교")  
- complex: 시뮬레이션/예측/다중 데이터 교차 ("카페 차리면 매출 얼마?")

query_intent:
- lookup: 특정 값 조회
- compare: 2개 이상 비교
- trend: 시계열 추이
- simulate: 가상 시나리오
- recommend: 추천/랭킹

JSON으로만 답변: {"query_type": "...", "query_intent": "..."}
"""
```

#### 2. SQL Node (Text-to-SQL)

```python
SCHEMA_PROMPT = """
사용 가능한 테이블:

1. population_agg (유동인구 - 법정동×월별)
   - DISTRICT_CODE (str): 법정동 코드 (8자리)
   - STANDARD_YEAR_MONTH (int): 기준년월 (202301~202506)
   - RESIDENTIAL_POPULATION (float): 거주인구
   - WORKING_POPULATION (float): 직장인구
   - VISITING_POPULATION (float): 방문인구

2. card_sales_agg (카드매출 - 법정동×월별)
   - DISTRICT_CODE, STANDARD_YEAR_MONTH
   - TOTAL_SALES, FOOD_SALES, COFFEE_SALES, ENTERTAINMENT_SALES,
     BEAUTY_SALES, MEDICAL_SALES, EDUCATION_ACADEMY_SALES, ...
   - 각 _SALES에 대응하는 _COUNT도 있음

3. income_agg (자산소득 - 법정동×월별)
   - DISTRICT_CODE, STANDARD_YEAR_MONTH, total_customers
   - AVERAGE_INCOME, MEDIAN_INCOME, AVERAGE_SCORE (신용점수)
   - RATE_INCOME_UNDER_20M ~ RATE_INCOME_OVER_70M (소득구간 비율)
   - RATE_MODEL_GROUP_LARGE_COMPANY_EMPLOYEE 등 (직업군 비율)

4. region_master (법정동 마스터)
   - district_code, city_kor (시군구), district_kor (법정동)

5. realestate (부동산시세 - 리치고, 3개구만)
   - BJD_CODE, YYYYMMDD, MEME_PRICE_PER_SUPPLY_PYEONG, 
     JEONSE_PRICE_PER_SUPPLY_PYEONG, TOTAL_HOUSEHOLDS

조인 방법:
- population/card/income ↔ region_master: DISTRICT_CODE = district_code
- realestate ↔ region_master: BJD_CODE 앞 8자리 = district_code

현재 컨텍스트:
- 선택된 지역: {selected_district}
- 기준 월: {selected_month}
- 데이터 범위: 서울 중구, 영등포구, 서초구 (118개 법정동)

DuckDB SQL 문법을 사용하세요. SQL만 반환하세요.
"""
```

#### 3. Simulate Node (Tier 3 전용)

```python
SIMULATE_PROMPT = """
당신은 상권 분석 전문가입니다.

아래 데이터를 기반으로 시뮬레이션 결과를 분석하세요:

[유동인구 데이터]
{population_data}

[카드매출 데이터]  
{card_sales_data}

[소득 데이터]
{income_data}

사용자 시나리오: {user_scenario}

다음을 포함하여 답변:
1. 예상 수치 (매출, 고객수 등) — 범위로 제시
2. 시간대별 분석
3. 리스크 요인
4. 추천 사항

데이터에 기반한 근거를 반드시 명시하세요.
"""
```

---

## 4. 도구(Tools) 정의

```python
from langchain_core.tools import tool

@tool
def query_population(district_name: str, year_month: int = None) -> str:
    """법정동의 유동인구 데이터 조회 (거주/직장/방문)"""
    ...

@tool  
def query_card_sales(district_name: str, year_month: int = None, category: str = "TOTAL") -> str:
    """법정동의 카드매출 데이터 조회 (20개 업종)"""
    ...

@tool
def query_income(district_name: str, year_month: int = None) -> str:
    """법정동의 소득/자산/신용 데이터 조회"""
    ...

@tool
def compare_districts(district_names: list[str], metric: str = "total_population") -> str:
    """2~3개 법정동 지표 비교"""
    ...

@tool
def get_trend(district_name: str, metric: str, months: int = 6) -> str:
    """법정동의 특정 지표 N개월 추이"""
    ...

@tool
def simulate_business(district_name: str, business_type: str, scale: int = 30) -> str:
    """특정 위치에 업종 시뮬레이션 — 매출 추정"""
    ...

@tool
def get_hotplace_ranking(top_n: int = 10) -> str:
    """핫플 스코어 랭킹 조회"""
    ...

@tool
def run_sql(query: str) -> str:
    """DuckDB SQL 직접 실행 (Text-to-SQL 결과)"""
    ...
```

---

## 5. LangGraph 구현 코드 구조

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

# 모델 초기화 (3-Tier)
model_nano = ChatOpenAI(model="gpt-5.4-nano", temperature=0)
model_mini = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
model_full = ChatOpenAI(model="gpt-5.4", temperature=0.1)

# 도구 바인딩
tools = [query_population, query_card_sales, query_income,
         compare_districts, get_trend, simulate_business,
         get_hotplace_ranking, run_sql]

# 그래프 빌드
graph = StateGraph(AgentState)

# 노드 추가
graph.add_node("context", collect_context)       # 대시보드 컨텍스트
graph.add_node("router", route_query)            # 질문 분류 (nano)
graph.add_node("simple", handle_simple)          # Tier 1 (nano + tools)
graph.add_node("analysis", handle_analysis)      # Tier 2 (mini + tools)
graph.add_node("complex", handle_complex)        # Tier 3 (full + tools)
graph.add_node("tools", ToolNode(tools))         # 도구 실행
graph.add_node("respond", format_response)       # 응답 포맷팅
graph.add_node("track_cost", log_cost)           # 비용 기록

# 엣지 정의
graph.add_edge(START, "context")
graph.add_edge("context", "router")

# 조건부 라우팅
graph.add_conditional_edges("router", route_by_complexity, {
    "simple": "simple",
    "analysis": "analysis", 
    "complex": "complex",
})

# 각 Tier → tools → respond
for tier in ["simple", "analysis", "complex"]:
    graph.add_conditional_edges(tier, should_use_tool, {
        "tools": "tools",
        "respond": "respond",
    })
    graph.add_edge("tools", tier)  # 도구 결과 → 다시 해당 Tier로

graph.add_edge("respond", "track_cost")
graph.add_edge("track_cost", END)

# 컴파일
agent = graph.compile()
```

---

## 6. Streamlit 통합

### 채팅 UI 위치

```
┌─────────────────────┬────────────────────────────────────────┐
│  📰 뉴스 타임라인    │  [현재 탭 컨텐츠]                       │
│  (사이드바)          │                                        │
│                     │                                        │
│                     │                                        │
│                     │                                        │
│                     ├────────────────────────────────────────┤
│                     │  🤖 AI 에이전트                         │
│                     │  ┌──────────────────────────────────┐  │
│                     │  │ 💬 신당동에 카페 차리면 매출이?    │  │
│                     │  │                                  │  │
│                     │  │ 🤖 신당동 카페 시뮬레이션 결과:   │  │
│                     │  │    월 예상 매출: 4,200~5,800만원  │  │
│                     │  │    [시간대별 차트]                │  │
│                     │  │    비용: $0.018                  │  │
│                     │  │                                  │  │
│                     │  │ [입력창____________] [전송]       │  │
│                     │  └──────────────────────────────────┘  │
│  💰 세션 비용: $0.05 │                                        │
└─────────────────────┴────────────────────────────────────────┘
```

### 구현 방식

```python
# 모든 페이지 하단에 공통 AI 채팅 패널
def render_ai_chat(current_tab, selected_district=None, selected_month=None):
    st.divider()
    st.subheader("🤖 AI 에이전트")
    
    # 세션 상태
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "cost_tracker" not in st.session_state:
        st.session_state.cost_tracker = CostTracker()
    
    # 대화 히스토리 표시
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "chart" in msg:
                st.plotly_chart(msg["chart"])
            if "cost" in msg:
                st.caption(f"비용: ${msg['cost']:.4f}")
    
    # 입력
    if prompt := st.chat_input("무엇이든 물어보세요..."):
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # LangGraph 에이전트 실행
        result = agent.invoke({
            "messages": [("user", prompt)],
            "current_tab": current_tab,
            "selected_district": selected_district,
            "selected_month": selected_month,
            "cost_log": [],
        })
        
        # 응답 추가
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["response_text"],
            "chart": result.get("chart_data"),
            "cost": sum(c["cost"] for c in result["cost_log"]),
        })
        st.rerun()
    
    # 사이드바에 누적 비용 표시
    with st.sidebar:
        total = st.session_state.cost_tracker.session_total
        st.metric("💰 세션 비용", f"${total:.4f}")
```

---

## 7. 비용 대시보드

### 사이드바 하단 실시간 표시

```
💰 세션 비용
├─ 총 비용: $0.052
├─ 질의 수: 12건
├─ Tier 분포: 🟢7 🟡4 🔴1
└─ 평균 비용: $0.004/건
```

### 관리자용 비용 리포트 (별도 페이지)

| 지표 | 값 |
|------|-----|
| 오늘 총 비용 | $0.34 |
| 이번 주 | $1.87 |
| 이번 달 | $7.23 |
| 평균 질의 비용 | $0.003 |
| 가장 비싼 질의 | "서초구 전체 상권 분석" ($0.04) |
| Tier 1 비율 | 62% |
| 캐시 히트율 | 38% |

---

## 8. 프로젝트 구조 (추가 파일)

```
app/
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph StateGraph 정의
│   ├── nodes.py          # 각 노드 구현 (router, simple, analysis, complex)
│   ├── tools.py          # 도구 정의 (SQL 실행, 시뮬레이션 등)
│   ├── prompts.py        # 프롬프트 템플릿
│   ├── cost_tracker.py   # 비용 추적 모듈
│   └── chat_ui.py        # Streamlit 채팅 UI 컴포넌트
├── main.py
├── ...
```

---

## 9. 필요 패키지

```
langgraph>=0.3.0
langchain-openai>=0.3.0
langchain-core>=0.3.0
duckdb>=1.0.0
```

---

## 10. 구현 계획

### Phase 4-1: 기반 (Day 1~2)
- [ ] DuckDB 연동 (parquet 직접 쿼리)
- [ ] 도구(Tools) 8개 구현
- [ ] CostTracker 구현
- [ ] 환경변수 설정 (OPENAI_API_KEY)

### Phase 4-2: LangGraph 그래프 (Day 2~3)
- [ ] StateGraph 정의
- [ ] Router 노드 (nano)
- [ ] Simple/Analysis/Complex 노드
- [ ] 조건부 엣지 라우팅
- [ ] Tool 실행 루프

### Phase 4-3: Streamlit 통합 (Day 3~4)
- [ ] 채팅 UI 컴포넌트
- [ ] 모든 탭에 AI 패널 추가
- [ ] 대시보드 컨텍스트 → 에이전트 전달
- [ ] 차트 응답 렌더링

### Phase 4-4: 비용 최적화 (Day 4~5)
- [ ] 프롬프트 캐싱 적용
- [ ] 비용 대시보드 페이지
- [ ] 응답 품질 vs 비용 A/B 테스트
- [ ] 안전장치: 일일 비용 한도 설정

---

## 11. 예상 질문-응답 시나리오

### Tier 1 (nano, ~$0.0002)
```
Q: "신당동 이번 달 유동인구 얼마야?"
→ SQL: SELECT * FROM population_agg WHERE ... 
→ A: "신당동 2025년 6월 총 유동인구는 45,935명입니다.
      (거주 9,819 / 직장 27,562 / 방문 8,554)"
```

### Tier 2 (mini, ~$0.004)
```
Q: "중구에서 커피 매출 가장 높은 동네 Top 5 알려줘"
→ SQL: SELECT r.district_kor, c.COFFEE_SALES FROM ...
→ A: "중구 커피 매출 Top 5:
      1. 명동 - 8.2억원
      2. 을지로 - 5.1억원
      3. 충무로 - 3.8억원
      ..."
→ [바 차트 렌더링]
```

### Tier 3 (gpt-5.4, ~$0.02)
```
Q: "서초동에 30석 규모 카페를 차리면 수익성이 어떨까?"
→ [유동인구 조회] + [카드매출 조회] + [소득 조회]
→ [시뮬레이션 실행]
→ A: "서초동 카페 시뮬레이션 결과:
      월 예상 매출: 4,800~6,200만원
      주 고객층: 30대 직장인 (42%)
      피크 시간: 오전 9~12시
      리스크: 주말 유동인구 -58%
      추천: 테이크아웃 중심, 오전 프로모션"
→ [시간대별 매출 차트] + [고객 프로파일 차트]
```
