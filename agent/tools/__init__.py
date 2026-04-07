"""도구 모듈"""
from agent.tools.data_tools import (
    query_population,
    query_card_sales,
    query_income,
    compare_districts,
    get_trend,
    get_hotplace_ranking,
    run_sql,
    simulate_business,
)

ALL_TOOLS = [
    query_population,
    query_card_sales,
    query_income,
    compare_districts,
    get_trend,
    get_hotplace_ranking,
    run_sql,
    simulate_business,
]
