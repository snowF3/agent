"""
LangGraph StateGraph 정의 — 3-Tier 라우팅 에이전트
"""
import json
from typing import TypedDict, Annotated, Literal
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from agent.prompts import ROUTER_SYSTEM, SQL_SYSTEM, RESPONSE_SYSTEM, SIMULATE_SYSTEM
from agent.tools import ALL_TOOLS
from agent.cost_tracker import CostTracker


# ── State 정의 ──
class AgentState(TypedDict):
    messages: Annotated[list, add]
    query_type: str          # simple | analysis | complex
    query_intent: str        # lookup | compare | trend | simulate | recommend
    selected_district: str
    selected_month: str
    current_tab: str
    page_context: str        # 현재 페이지에서 사용자가 보고 있는 상세 컨텍스트
    response_text: str
    chart_data: dict
    cost_log: Annotated[list, add]


# ── 모델 초기화 ──
MODEL_NANO = "gpt-4.1-nano"    # Tier 1 + Router (가장 저렴, 실제 배포 시 gpt-5.4-nano로 교체)
MODEL_MINI = "gpt-4.1-mini"    # Tier 2 (분석/비교)
MODEL_FULL = "gpt-4.1"         # Tier 3 (복합 추론)


def _get_model(tier: str) -> ChatOpenAI:
    """Tier별 모델 반환"""
    model_map = {
        "nano": MODEL_NANO,
        "mini": MODEL_MINI,
        "full": MODEL_FULL,
    }
    model_name = model_map.get(tier, MODEL_NANO)
    return ChatOpenAI(model=model_name, temperature=0)


def _track_cost(state: AgentState, model: str, response) -> dict:
    """응답에서 토큰 사용량 추출하여 비용 기록"""
    tracker = CostTracker()
    usage = getattr(response, "usage_metadata", None) or {}
    entry = tracker.track(
        model=model,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cached_tokens=usage.get("input_token_details", {}).get("cached", 0),
    )
    return entry


# ══════════════════════════════════════════════
# 노드 함수들
# ══════════════════════════════════════════════

def context_node(state: AgentState) -> dict:
    """대시보드 컨텍스트 수집 (LLM 미사용)"""
    # 이미 state에 주입되어 있으므로 패스스루
    return {}


def router_node(state: AgentState) -> dict:
    """질문 복잡도 + 의도 분류 (nano)"""
    model = _get_model("nano")
    user_msg = state["messages"][-1] if state["messages"] else ""

    if isinstance(user_msg, HumanMessage):
        content = user_msg.content
    elif isinstance(user_msg, tuple):
        content = user_msg[1]
    else:
        content = str(user_msg)

    response = model.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=content),
    ])

    cost_entry = _track_cost(state, MODEL_NANO, response)

    try:
        parsed = json.loads(response.content)
        query_type = parsed.get("query_type", "simple")
        query_intent = parsed.get("query_intent", "lookup")
    except (json.JSONDecodeError, AttributeError):
        query_type = "simple"
        query_intent = "lookup"

    return {
        "query_type": query_type,
        "query_intent": query_intent,
        "cost_log": [cost_entry],
    }


def simple_node(state: AgentState) -> dict:
    """Tier 1: 단순 조회 (nano + tools)"""
    model = _get_model("nano").bind_tools(ALL_TOOLS)
    user_msg = state["messages"][-1]

    context = f"지역: {state.get('selected_district', '미선택')}, 기준월: {state.get('selected_month', '최신')}"
    page_ctx = state.get('page_context', '')

    system_prompt = (
        f"당신은 데이터 조회 도우미입니다. 도구를 사용해 정확한 데이터를 조회하세요.\n"
        f"컨텍스트: {context}\n"
        f"[사용자가 현재 보고 있는 화면]\n{page_ctx}\n\n"
        f"[필수 규칙]\n"
        f"- 절대로 사용자에게 지역이나 기준월을 되묻지 마세요. 이전 대화 또는 현재 화면 컨텍스트에서 파악하세요.\n"
        f"- 사용자가 '이거', '여기', '이 동네' 등 대명사를 쓰면 현재 화면의 지역/데이터를 참조하세요.\n"
        f"- 정보가 부족하면 도구를 사용해 직접 조회하세요.\n"
        f"- 분석 시 유동인구 + 카드매출 + 소득 데이터를 함께 교차 분석하세요.\n"
        f"- 업종 추천 시 해당 지역의 카드매출 업종별 비중을 반드시 조회하세요."
    )
    response = model.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"],
    ])

    cost_entry = _track_cost(state, MODEL_NANO, response)

    return {
        "messages": [response],
        "cost_log": [cost_entry],
    }


def analysis_node(state: AgentState) -> dict:
    """Tier 2: 분석/비교 (mini + tools)"""
    model = _get_model("mini").bind_tools(ALL_TOOLS)

    context = f"지역: {state.get('selected_district', '미선택')}, 기준월: {state.get('selected_month', '최신')}"
    page_ctx = state.get('page_context', '')

    system_prompt = (
        f"당신은 데이터 분석 전문가입니다. 도구를 사용해 데이터를 조회하고 비교/분석하세요.\n"
        f"컨텍스트: {context}\n"
        f"[사용자가 현재 보고 있는 화면]\n{page_ctx}\n\n"
        f"[필수 규칙]\n"
        f"- 절대로 사용자에게 지역이나 기준월을 되묻지 마세요. 이전 대화 또는 현재 화면 컨텍스트에서 파악하세요.\n"
        f"- 사용자가 '이거', '여기', '이 동네' 등 대명사를 쓰면 현재 화면의 지역/데이터를 참조하세요.\n"
        f"- 정보가 부족하면 도구를 사용해 직접 조회하세요.\n"
        f"- 분석 시 유동인구 + 카드매출 + 소득 데이터를 함께 교차 분석하세요.\n"
        f"- 업종 추천 시 해당 지역의 카드매출 업종별 비중을 반드시 조회하세요."
    )
    response = model.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"],
    ])

    cost_entry = _track_cost(state, MODEL_MINI, response)

    return {
        "messages": [response],
        "cost_log": [cost_entry],
    }


def complex_node(state: AgentState) -> dict:
    """Tier 3: 복합 추론/시뮬레이션 (full + tools)"""
    model = _get_model("full").bind_tools(ALL_TOOLS)

    context = f"지역: {state.get('selected_district', '미선택')}, 기준월: {state.get('selected_month', '최신')}"
    page_ctx = state.get('page_context', '')

    system_prompt = (
        f"당신은 상권/부동산/마케팅 데이터 분석 전문가입니다. 도구를 적극 활용하여 종합 분석, 시뮬레이션, 예측을 수행하세요.\n"
        f"컨텍스트: {context}\n"
        f"[사용자가 현재 보고 있는 화면]\n{page_ctx}\n\n"
        f"[필수 규칙]\n"
        f"- 절대로 사용자에게 지역이나 기준월을 되묻지 마세요. 이전 대화 또는 현재 화면 컨텍스트에서 파악하세요.\n"
        f"- 사용자가 '이거', '여기', '이 동네' 등 대명사를 쓰면 현재 화면의 지역/데이터를 참조하세요.\n"
        f"- 정보가 부족하면 도구를 사용해 직접 조회하세요.\n"
        f"- 분석 시 유동인구 + 카드매출 + 소득 데이터를 함께 교차 분석하세요.\n"
        f"- 업종 추천 시 해당 지역의 카드매출 업종별 비중을 반드시 조회하세요.\n"
        f"- 시뮬레이션 시 여러 도구를 순차적으로 호출하세요: 1) query_population 2) query_card_sales 3) query_income 4) simulate_business\n"
        f"- 시뮬레이션 결과가 0이면 직접 유동인구 x 업종비율 x 포획률로 계산하세요."
    )
    response = model.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"],
    ])

    cost_entry = _track_cost(state, MODEL_FULL, response)

    return {
        "messages": [response],
        "cost_log": [cost_entry],
    }


def respond_node(state: AgentState) -> dict:
    """최종 응답 포맷팅"""
    last_msg = state["messages"][-1] if state["messages"] else None
    if last_msg and isinstance(last_msg, AIMessage):
        return {
            "response_text": last_msg.content,
        }
    return {"response_text": "응답을 생성하지 못했습니다."}


# ══════════════════════════════════════════════
# 라우팅 함수들
# ══════════════════════════════════════════════

def route_by_complexity(state: AgentState) -> Literal["simple", "analysis", "complex"]:
    """질문 복잡도에 따라 Tier 라우팅"""
    qt = state.get("query_type", "simple")
    if qt == "complex":
        return "complex"
    elif qt == "analysis":
        return "analysis"
    return "simple"


def should_continue(state: AgentState) -> Literal["tools", "respond"]:
    """도구 호출이 필요한지 판단"""
    last_msg = state["messages"][-1] if state["messages"] else None
    if last_msg and isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return "respond"


# ══════════════════════════════════════════════
# 그래프 빌드
# ══════════════════════════════════════════════

def build_graph():
    """LangGraph 에이전트 그래프 빌드 & 컴파일"""
    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("context", context_node)
    graph.add_node("router", router_node)
    graph.add_node("simple", simple_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("complex", complex_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("respond", respond_node)

    # 엣지
    graph.add_edge(START, "context")
    graph.add_edge("context", "router")

    # 라우터 → Tier별 분기
    graph.add_conditional_edges("router", route_by_complexity, {
        "simple": "simple",
        "analysis": "analysis",
        "complex": "complex",
    })

    # 각 Tier → 도구 사용 여부 판단
    for tier in ["simple", "analysis", "complex"]:
        graph.add_conditional_edges(tier, should_continue, {
            "tools": "tools",
            "respond": "respond",
        })

    # 도구 실행 후 → 원래 Tier로 복귀
    def route_back_from_tools(state: AgentState) -> str:
        return state.get("query_type", "simple")

    graph.add_conditional_edges("tools", route_back_from_tools, {
        "simple": "simple",
        "analysis": "analysis",
        "complex": "complex",
    })

    graph.add_edge("respond", END)

    return graph.compile()


# 싱글턴 에이전트
agent = build_graph()
