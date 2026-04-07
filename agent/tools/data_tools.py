"""
데이터 조회/분석 도구 — DuckDB parquet 직접 쿼리
"""
import os
import json
import duckdb
import pandas as pd
from langchain_core.tools import tool
from pathlib import Path

# 데이터 디렉토리 (환경변수 또는 기본값)
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent.parent / "processed_data"))


def _get_conn():
    """DuckDB 인메모리 커넥션"""
    return duckdb.connect(":memory:")


def _parquet_path(name: str) -> str:
    """parquet 파일 경로"""
    return str(DATA_DIR / f"{name}.parquet")


def _resolve_district(conn, district_name: str) -> str:
    """동네 이름 → district_code 변환"""
    rm_path = _parquet_path("region_master")
    result = conn.execute(f"""
        SELECT district_code FROM read_parquet('{rm_path}')
        WHERE district_kor = ? OR city_kor || ' ' || district_kor = ?
        LIMIT 1
    """, [district_name, district_name]).fetchone()
    return result[0] if result else None


@tool
def query_population(district_name: str, year_month: int = 0) -> str:
    """법정동의 유동인구 데이터를 조회합니다 (거주/직장/방문 인구). district_name: 법정동 이름 (예: '신당동' 또는 '중구 신당동'), year_month: 기준년월 YYYYMM (0이면 최신)"""
    conn = _get_conn()
    dc = _resolve_district(conn, district_name)
    if not dc:
        return f"'{district_name}'에 해당하는 법정동을 찾을 수 없습니다. 중구/영등포구/서초구 내 법정동만 가능합니다."

    pop_path = _parquet_path("population_agg")
    if year_month == 0:
        ym_filter = f"STANDARD_YEAR_MONTH = (SELECT MAX(STANDARD_YEAR_MONTH) FROM read_parquet('{pop_path}'))"
    else:
        ym_filter = f"STANDARD_YEAR_MONTH = {year_month}"

    result = conn.execute(f"""
        SELECT STANDARD_YEAR_MONTH, RESIDENTIAL_POPULATION, WORKING_POPULATION, VISITING_POPULATION,
               RESIDENTIAL_POPULATION + WORKING_POPULATION + VISITING_POPULATION as TOTAL
        FROM read_parquet('{pop_path}')
        WHERE DISTRICT_CODE = '{dc}' AND {ym_filter}
    """).fetchdf()

    if result.empty:
        return f"'{district_name}' ({dc})에 대한 유동인구 데이터가 없습니다."
    return result.to_json(orient="records", force_ascii=False)


@tool
def query_card_sales(district_name: str, year_month: int = 0, category: str = "TOTAL") -> str:
    """법정동의 카드매출 데이터를 조회합니다. category: 업종 (TOTAL/FOOD/COFFEE/BEAUTY/MEDICAL 등)"""
    conn = _get_conn()
    dc = _resolve_district(conn, district_name)
    if not dc:
        return f"'{district_name}'을 찾을 수 없습니다."

    card_path = _parquet_path("card_sales_agg")
    if year_month == 0:
        ym_filter = f"STANDARD_YEAR_MONTH = (SELECT MAX(STANDARD_YEAR_MONTH) FROM read_parquet('{card_path}'))"
    else:
        ym_filter = f"STANDARD_YEAR_MONTH = {year_month}"

    sales_col = f"{category}_SALES" if category != "TOTAL" else "TOTAL_SALES"
    count_col = f"{category}_COUNT" if category != "TOTAL" else "TOTAL_COUNT"

    result = conn.execute(f"""
        SELECT STANDARD_YEAR_MONTH, {sales_col} as sales, {count_col} as count
        FROM read_parquet('{card_path}')
        WHERE DISTRICT_CODE = '{dc}' AND {ym_filter}
    """).fetchdf()

    if result.empty:
        return f"데이터 없음"
    return result.to_json(orient="records", force_ascii=False)


@tool
def query_income(district_name: str, year_month: int = 0) -> str:
    """법정동의 소득/자산/신용 데이터를 조회합니다."""
    conn = _get_conn()
    dc = _resolve_district(conn, district_name)
    if not dc:
        return f"'{district_name}'을 찾을 수 없습니다."

    inc_path = _parquet_path("income_agg")
    if year_month == 0:
        ym_filter = f"STANDARD_YEAR_MONTH = (SELECT MAX(STANDARD_YEAR_MONTH) FROM read_parquet('{inc_path}'))"
    else:
        ym_filter = f"STANDARD_YEAR_MONTH = {year_month}"

    result = conn.execute(f"""
        SELECT STANDARD_YEAR_MONTH, total_customers, AVERAGE_INCOME, MEDIAN_INCOME,
               AVERAGE_SCORE, AVERAGE_ASSET_AMOUNT, RATE_HIGHEND
        FROM read_parquet('{inc_path}')
        WHERE DISTRICT_CODE = '{dc}' AND {ym_filter}
    """).fetchdf()

    if result.empty:
        return f"데이터 없음"
    return result.to_json(orient="records", force_ascii=False)


@tool
def compare_districts(district_names: str, metric: str = "total_population") -> str:
    """2~3개 법정동의 지표를 비교합니다. district_names: 쉼표로 구분 (예: '신당동,서초동,여의도동'), metric: 비교 지표 (total_population/total_sales/average_income)"""
    conn = _get_conn()
    names = [n.strip() for n in district_names.split(",")]
    codes = []
    for name in names:
        dc = _resolve_district(conn, name)
        if dc:
            codes.append((name, dc))

    if len(codes) < 2:
        return f"비교 가능한 법정동이 2개 미만입니다. 찾은 법정동: {[c[0] for c in codes]}"

    pop_path = _parquet_path("population_agg")
    card_path = _parquet_path("card_sales_agg")
    inc_path = _parquet_path("income_agg")

    dc_list = ",".join(f"'{c[1]}'" for c in codes)

    results = conn.execute(f"""
        SELECT r.district_kor, r.city_kor,
               p.RESIDENTIAL_POPULATION + p.WORKING_POPULATION + p.VISITING_POPULATION as total_pop,
               c.TOTAL_SALES,
               i.AVERAGE_INCOME, i.AVERAGE_SCORE
        FROM read_parquet('{pop_path}') p
        JOIN read_parquet('{_parquet_path("region_master")}') r ON p.DISTRICT_CODE = r.district_code
        LEFT JOIN read_parquet('{card_path}') c ON p.DISTRICT_CODE = c.DISTRICT_CODE AND p.STANDARD_YEAR_MONTH = c.STANDARD_YEAR_MONTH
        LEFT JOIN read_parquet('{inc_path}') i ON p.DISTRICT_CODE = i.DISTRICT_CODE AND p.STANDARD_YEAR_MONTH = i.STANDARD_YEAR_MONTH
        WHERE p.DISTRICT_CODE IN ({dc_list})
          AND p.STANDARD_YEAR_MONTH = (SELECT MAX(STANDARD_YEAR_MONTH) FROM read_parquet('{pop_path}'))
    """).fetchdf()

    return results.to_json(orient="records", force_ascii=False)


@tool
def get_trend(district_name: str, metric: str = "total_population", months: int = 6) -> str:
    """법정동의 특정 지표 월별 추이를 조회합니다. metric: total_population/total_sales/coffee_sales/average_income"""
    conn = _get_conn()
    dc = _resolve_district(conn, district_name)
    if not dc:
        return f"'{district_name}'을 찾을 수 없습니다."

    pop_path = _parquet_path("population_agg")
    card_path = _parquet_path("card_sales_agg")
    inc_path = _parquet_path("income_agg")

    if metric == "total_population":
        result = conn.execute(f"""
            SELECT STANDARD_YEAR_MONTH,
                   RESIDENTIAL_POPULATION + WORKING_POPULATION + VISITING_POPULATION as value
            FROM read_parquet('{pop_path}')
            WHERE DISTRICT_CODE = '{dc}'
            ORDER BY STANDARD_YEAR_MONTH DESC LIMIT {months}
        """).fetchdf()
    elif "sales" in metric.lower():
        col = metric.upper() if "_SALES" in metric.upper() else f"{metric.upper()}_SALES"
        if col == "TOTAL_POPULATION_SALES":
            col = "TOTAL_SALES"
        result = conn.execute(f"""
            SELECT STANDARD_YEAR_MONTH, {col} as value
            FROM read_parquet('{card_path}')
            WHERE DISTRICT_CODE = '{dc}'
            ORDER BY STANDARD_YEAR_MONTH DESC LIMIT {months}
        """).fetchdf()
    else:
        col = metric.upper()
        result = conn.execute(f"""
            SELECT STANDARD_YEAR_MONTH, {col} as value
            FROM read_parquet('{inc_path}')
            WHERE DISTRICT_CODE = '{dc}'
            ORDER BY STANDARD_YEAR_MONTH DESC LIMIT {months}
        """).fetchdf()

    if result.empty:
        return "데이터 없음"
    return result.to_json(orient="records", force_ascii=False)


@tool
def get_hotplace_ranking(top_n: int = 10) -> str:
    """방문인구 증가율 기반 핫플레이스 랭킹을 조회합니다."""
    conn = _get_conn()
    pop_path = _parquet_path("population_agg")
    rm_path = _parquet_path("region_master")

    result = conn.execute(f"""
        WITH recent AS (
            SELECT DISTRICT_CODE,
                   AVG(VISITING_POPULATION) as recent_avg
            FROM read_parquet('{pop_path}')
            WHERE STANDARD_YEAR_MONTH >= (SELECT MAX(STANDARD_YEAR_MONTH) - 6 FROM read_parquet('{pop_path}'))
            GROUP BY DISTRICT_CODE
        ),
        prev AS (
            SELECT DISTRICT_CODE,
                   AVG(VISITING_POPULATION) as prev_avg
            FROM read_parquet('{pop_path}')
            WHERE STANDARD_YEAR_MONTH < (SELECT MAX(STANDARD_YEAR_MONTH) - 6 FROM read_parquet('{pop_path}'))
              AND STANDARD_YEAR_MONTH >= (SELECT MAX(STANDARD_YEAR_MONTH) - 12 FROM read_parquet('{pop_path}'))
            GROUP BY DISTRICT_CODE
        )
        SELECT r.city_kor || ' ' || r.district_kor as name,
               ROUND((recent.recent_avg - prev.prev_avg) / NULLIF(prev.prev_avg, 0) * 100, 1) as growth_pct,
               ROUND(recent.recent_avg, 0) as current_visitors
        FROM recent
        JOIN prev ON recent.DISTRICT_CODE = prev.DISTRICT_CODE
        JOIN read_parquet('{rm_path}') r ON recent.DISTRICT_CODE = r.district_code
        WHERE prev.prev_avg > 0
        ORDER BY growth_pct DESC
        LIMIT {top_n}
    """).fetchdf()

    return result.to_json(orient="records", force_ascii=False)


@tool
def run_sql(query: str) -> str:
    """DuckDB SQL을 직접 실행합니다. parquet 파일 경로는 read_parquet('processed_data/파일명.parquet') 형식입니다."""
    conn = _get_conn()
    # 경로 치환: processed_data/ → 실제 DATA_DIR
    adjusted = query.replace("processed_data/", str(DATA_DIR) + "/")
    try:
        result = conn.execute(adjusted).fetchdf()
        if len(result) > 50:
            result = result.head(50)
        return result.to_json(orient="records", force_ascii=False)
    except Exception as e:
        return f"SQL 실행 오류: {str(e)}"


@tool
def simulate_business(district_name: str, business_type: str = "카페", seats: int = 30) -> str:
    """특정 법정동에 업종을 개업했을 때 예상 매출을 시뮬레이션합니다. business_type: 카페/음식점/미용실/편의점"""
    conn = _get_conn()
    dc = _resolve_district(conn, district_name)
    if not dc:
        return f"'{district_name}'을 찾을 수 없습니다."

    pop_path = _parquet_path("population_time_agg")
    card_path = _parquet_path("card_sales_agg")
    inc_path = _parquet_path("income_agg")

    biz_map = {"카페": "COFFEE", "음식점": "FOOD", "미용실": "BEAUTY", "편의점": "SMALL_RETAIL_STORE"}
    price_map = {"카페": 5000, "음식점": 12000, "미용실": 25000, "편의점": 8000}
    biz_key = biz_map.get(business_type, "COFFEE")
    avg_price = price_map.get(business_type, 5000)

    # 시간대별 유동인구
    pop_time = conn.execute(f"""
        SELECT TIME_SLOT,
               SUM(RESIDENTIAL_POPULATION + WORKING_POPULATION + VISITING_POPULATION) as total_pop
        FROM read_parquet('{pop_path}')
        WHERE DISTRICT_CODE = '{dc}'
          AND STANDARD_YEAR_MONTH = (SELECT MAX(STANDARD_YEAR_MONTH) FROM read_parquet('{pop_path}'))
          AND WEEKDAY_WEEKEND = 'W'
        GROUP BY TIME_SLOT
        ORDER BY TIME_SLOT
    """).fetchdf()

    # 업종 매출 비중
    card = conn.execute(f"""
        SELECT {biz_key}_SALES as biz_sales, TOTAL_SALES
        FROM read_parquet('{card_path}')
        WHERE DISTRICT_CODE = '{dc}'
          AND STANDARD_YEAR_MONTH = (SELECT MAX(STANDARD_YEAR_MONTH) FROM read_parquet('{card_path}'))
    """).fetchdf()

    # 업종 매출 비중 (최소 3%)
    biz_ratio = 0.05
    if not card.empty and card["TOTAL_SALES"].values[0] > 0:
        biz_ratio = card["biz_sales"].values[0] / card["TOTAL_SALES"].values[0]
    biz_ratio = max(biz_ratio, 0.03)

    # population_time_agg가 비어있으면 population_agg에서 폴백
    if pop_time.empty:
        pop_agg_path = _parquet_path("population_agg")
        pop_agg = conn.execute(f"""
            SELECT RESIDENTIAL_POPULATION + WORKING_POPULATION + VISITING_POPULATION as total_pop
            FROM read_parquet('{pop_agg_path}')
            WHERE DISTRICT_CODE = '{dc}'
              AND STANDARD_YEAR_MONTH = (SELECT MAX(STANDARD_YEAR_MONTH) FROM read_parquet('{pop_agg_path}'))
        """).fetchdf()

        if not pop_agg.empty:
            total_daily_pop = int(pop_agg["total_pop"].values[0])
            num_slots = 7
            per_slot = total_daily_pop // num_slots
            pop_time = pd.DataFrame({
                "TIME_SLOT": [f"T{i}" for i in range(num_slots)],
                "total_pop": [per_slot] * num_slots,
            })

    capture_rate = 0.05
    time_results = []
    for _, row in pop_time.iterrows():
        customers = int(row["total_pop"] * biz_ratio * capture_rate)
        time_results.append({
            "time_slot": row["TIME_SLOT"],
            "population": int(row["total_pop"]),
            "estimated_customers": customers,
            "revenue": customers * avg_price,
        })

    daily = sum(r["revenue"] for r in time_results)
    monthly = daily * 22 + int(daily * 0.6 * 8)

    return json.dumps({
        "district": district_name,
        "business_type": business_type,
        "seats": seats,
        "biz_ratio": round(biz_ratio, 3),
        "daily_revenue_weekday": daily,
        "monthly_revenue_estimate": monthly,
        "time_detail": time_results,
    }, ensure_ascii=False)
