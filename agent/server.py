"""
FastAPI 서버 — AI 에이전트 API 엔드포인트
"""
import os
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage as LCAIMessage

from agent.graph import agent
from agent.cost_tracker import CostTracker

# 세션별 비용 추적 (메모리, 프로덕션에서는 Redis/DB)
cost_tracker = CostTracker(daily_limit=float(os.getenv("DAILY_COST_LIMIT", "1.0")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🤖 AI 에이전트 서버 시작")
    yield
    print("🤖 AI 에이전트 서버 종료")


app = FastAPI(
    title="동네 엑스레이 AI 에이전트",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청/응답 모델 ──

class ChatRequest(BaseModel):
    query: str = Field(..., description="사용자 질문")
    selected_district: str = Field(default="", description="선택된 법정동")
    selected_month: str = Field(default="", description="기준 년월 (YYYYMM)")
    current_tab: str = Field(default="", description="현재 탭")
    chat_history: list[dict] = Field(default=[], description="이전 대화 이력 (role/content)")


class CostDetail(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    cost: float


class ChatResponse(BaseModel):
    answer: str
    query_type: str
    query_intent: str
    chart_data: dict | None = None
    costs: list[CostDetail]
    total_cost: float
    session_total_cost: float


class CostSummaryResponse(BaseModel):
    total_cost: float
    total_queries: int
    avg_cost: float
    by_model: dict
    total_input_tokens: int
    total_output_tokens: int
    tier_distribution: dict


class HealthResponse(BaseModel):
    status: str
    uptime: float


_start_time = time.time()


# ── 엔드포인트 ──

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", uptime=time.time() - _start_time)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """메인 채팅 엔드포인트"""
    if cost_tracker.is_over_limit():
        raise HTTPException(429, f"일일 비용 한도 초과 (${cost_tracker.daily_limit})")

    # 이전 대화 이력을 LangChain 메시지로 변환
    history_messages = []
    for msg in req.chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(LCAIMessage(content=content))

    try:
        result = agent.invoke({
            "messages": history_messages + [HumanMessage(content=req.query)],
            "query_type": "",
            "query_intent": "",
            "selected_district": req.selected_district,
            "selected_month": req.selected_month,
            "current_tab": req.current_tab,
            "response_text": "",
            "chart_data": {},
            "cost_log": [],
        })
    except Exception as e:
        raise HTTPException(500, f"에이전트 실행 오류: {str(e)}")

    # 비용 집계
    cost_log = result.get("cost_log", [])
    costs = []
    total_cost = 0
    for entry in cost_log:
        c = CostDetail(
            model=entry.get("model", "unknown"),
            input_tokens=entry.get("input_tokens", 0),
            output_tokens=entry.get("output_tokens", 0),
            cost=entry.get("cost", 0),
        )
        costs.append(c)
        total_cost += c.cost
        # 글로벌 트래커에도 기록
        cost_tracker.track(
            model=c.model,
            input_tokens=c.input_tokens,
            output_tokens=c.output_tokens,
        )

    # 차트 데이터 파싱
    chart_data = result.get("chart_data")
    if isinstance(chart_data, str):
        import json
        try:
            chart_data = json.loads(chart_data)
        except (json.JSONDecodeError, TypeError):
            chart_data = None

    return ChatResponse(
        answer=result.get("response_text", "응답 생성 실패"),
        query_type=result.get("query_type", "unknown"),
        query_intent=result.get("query_intent", "unknown"),
        chart_data=chart_data,
        costs=costs,
        total_cost=round(total_cost, 6),
        session_total_cost=round(cost_tracker.session_total, 6),
    )


@app.get("/costs", response_model=CostSummaryResponse)
async def get_costs():
    """세션 비용 요약"""
    summary = cost_tracker.get_summary()
    tier_dist = cost_tracker.get_tier_distribution()
    return CostSummaryResponse(
        total_cost=summary["total_cost"],
        total_queries=summary["total_queries"],
        avg_cost=summary["avg_cost"],
        by_model=summary["by_model"],
        total_input_tokens=summary["total_input_tokens"],
        total_output_tokens=summary["total_output_tokens"],
        tier_distribution=tier_dist,
    )


@app.post("/costs/reset")
async def reset_costs():
    """비용 추적 리셋"""
    cost_tracker.history.clear()
    cost_tracker.session_total = 0.0
    return {"status": "reset"}
