"""
비용 추적 모듈 — 모든 LLM 호출의 토큰/비용을 실시간 기록
"""
from datetime import datetime
from typing import Optional


# per 1M tokens 기준 (USD)
PRICING = {
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
}


class CostTracker:
    def __init__(self, daily_limit: float = 1.0):
        self.history: list[dict] = []
        self.session_total: float = 0.0
        self.daily_limit = daily_limit

    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        metadata: Optional[dict] = None,
    ) -> dict:
        """LLM 호출 비용 기록. 반환: 비용 상세 dict"""
        pricing = PRICING.get(model)
        if not pricing:
            # 알 수 없는 모델 → 가장 비싼 가격으로 추정
            pricing = PRICING["gpt-5.4"]

        uncached_input = input_tokens - cached_tokens
        cost = (
            uncached_input / 1_000_000 * pricing["input"]
            + cached_tokens / 1_000_000 * pricing["cached_input"]
            + output_tokens / 1_000_000 * pricing["output"]
        )

        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "cost": round(cost, 6),
            "metadata": metadata or {},
        }
        self.history.append(entry)
        self.session_total += cost
        return entry

    def is_over_limit(self) -> bool:
        """일일 비용 한도 초과 여부"""
        return self.session_total >= self.daily_limit

    def get_summary(self) -> dict:
        """세션 비용 요약"""
        if not self.history:
            return {
                "total_cost": 0, "total_queries": 0,
                "avg_cost": 0, "by_model": {},
                "total_input_tokens": 0, "total_output_tokens": 0,
            }

        by_model = {}
        total_input = 0
        total_output = 0
        for entry in self.history:
            m = entry["model"]
            by_model[m] = by_model.get(m, 0) + entry["cost"]
            total_input += entry["input_tokens"]
            total_output += entry["output_tokens"]

        return {
            "total_cost": round(self.session_total, 6),
            "total_queries": len(self.history),
            "avg_cost": round(self.session_total / len(self.history), 6),
            "by_model": by_model,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
        }

    def get_tier_distribution(self) -> dict:
        """Tier별 질의 분포"""
        tiers = {"nano": 0, "mini": 0, "full": 0}
        for entry in self.history:
            model = entry["model"]
            if "nano" in model:
                tiers["nano"] += 1
            elif "mini" in model:
                tiers["mini"] += 1
            else:
                tiers["full"] += 1
        return tiers
