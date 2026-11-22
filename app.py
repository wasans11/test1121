# app.py
import json
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostRegressor

from weather_auto import (
    calc_discomfort_index,
    find_nearest_branch,
    get_future_weather_calendar,
    now_kst_naive,
    resolve_branch_coords,
)

try:
    from streamlit_geolocation import streamlit_geolocation
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

st.set_page_config(page_title="무잉 발주 추천 프로토타입", page_icon="🧊", layout="wide")

MODEL_DIR = "."

DEFAULT_PACK = {
    "하드": 12,
    "쭈쭈바": 10,
    "콘": 6,
    "샌드": 8,
    "파인트": 4,
    "과자": 20,
    "젤리": 12,
    "음료": 12,
    "초콜릿": 12,
}
WEEKDAY_KR = ["월", "화", "수", "목", "금", "토", "일"]


def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return default


@st.cache_resource
def load_models():
    mean_model = CatBoostRegressor()
    low_model = CatBoostRegressor()
    high_model = CatBoostRegressor()
    mean_model.load_model(f"{MODEL_DIR}/poisson_mean.cbm")
    low_model.load_model(f"{MODEL_DIR}/quantile_10.cbm")
    high_model.load_model(f"{MODEL_DIR}/quantile_90.cbm")
    with open(f"{MODEL_DIR}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return mean_model, low_model, high_model, meta


@st.cache_data
def load_sample():
    return pd.read_csv("sample_test.csv", encoding="utf-8-sig")


def resolve_col(features, rename, candidates):
    """meta['features'] 내 실제 컬럼명을 한글 라벨 또는 별칭으로 자동 탐색"""
    for c in candidates:
        if c in features:
            return c
    rev = {v: k for k, v in rename.items()}
    for c in candidates:
        if c in rev and rev[c] in features:
            return rev[c]
    return None


def align_catboost_df(df: pd.DataFrame, features, cat_cols, categories, default_row) -> pd.DataFrame:
    """CatBoost 입력 타입 및 컬럼 순서 일치"""
    out = pd.DataFrame(index=df.index)
    for col in features:
        if col in df.columns:
            s = df[col]
        else:
            s = pd.Series([default_row.get(col, 0)] * len(df), index=df.index)

        if col in cat_cols:
            choices = categories.get(col, [])
            if choices and isinstance(choices[0], (int, np.integer)) and not isinstance(choices[0], bool):
                s = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
            else:
                s = s.apply(
                    lambda x: str(int(x))
                    if isinstance(x, (float, np.floating)) and pd.notna(x) and float(x).is_integer()
                    else str(x)
                )
        else:
            s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
        out[col] = s
    return out[features]


def predict_hourly_batch(mean_model, low_model, high_model, df_aligned):
    mean_p = np.maximum(0.0, mean_model.predict(df_aligned, prediction_type="Exponent"))
    low_p = np.maximum(0.0, low_model.predict(df_aligned))
    high_p = np.maximum(0.0, high_model.predict(df_aligned))
    low_p = np.minimum(low_p, mean_p)
    high_p = np.maximum(high_p, mean_p)
    return mean_p, low_p, high_p


@st.cache_data(ttl=1800)
def cached_weather(lat: float, lon: float, kma_key: str, hours: int, weekday_base: int):
    return get_future_weather_calendar(
        lat=lat, lon=lon, kma_api_key=kma_key, hours=hours, weekday_base=weekday_base
    )


# ── 1. 모델 및 메타데이터 로드 ──
mean_model, low_model, high_model, meta = load_models()
sample = load_sample()
features = meta["features"]
cat_cols = meta.get("cat_cols", [])
categories = meta.get("categories", {})
rename = meta.get("rename", {})
default_row = sample.iloc[0].to_dict()

COL = {
    "city": resolve_col(features, rename, ["지점명", "branch", "city", "region"]),
    "month": resolve_col(features, rename, ["월", "month"]),
    "day": resolve_col(features, rename, ["일", "day"]),
    "hour": resolve_col(features, rename, ["시", "hour"]),
    "temp": resolve_col(features, rename, ["기온", "temp", "temperature"]),
    "rain": resolve_col(features, rename, ["강수량", "rain", "precipitation"]),
    "hum": resolve_col(features, rename, ["습도", "humidity"]),
    "wind": resolve_col(features, rename, ["풍속", "wind", "wind_speed"]),
    "di": resolve_col(features, rename, ["불쾌지수", "di", "discomfort_index"]),
    "weekend": resolve_col(features, rename, ["주말", "weekend", "is_weekend"]),
    "holiday": resolve_col(features, rename, ["공휴일", "holiday", "is_holiday"]),
    "long_hol": resolve_col(features, rename, ["연휴", "long_holiday"]),
    "vac_s": resolve_col(features, rename, ["초중고방학", "school_vacation"]),
    "vac_u": resolve_col(features, rename, ["대학방학", "univ_vacation"]),
    "store_id": resolve_col(features, rename, ["매장ID", "store_id"]),
    "store_type": resolve_col(features, rename, ["매장유형", "store_type"]),
    "store_size": resolve_col(features, rename, ["매장규모", "store_size"]),
    "res_pop": resolve_col(features, rename, ["상주인구", "resident_pop"]),
    "flt_pop": resolve_col(features, rename, ["유동인구", "floating_pop"]),
    "schools": resolve_col(features, rename, ["학교수", "school_cnt", "schools"]),
    "weekday": resolve_col(features, rename, ["요일", "weekday", "dayofweek"]),
    "category": resolve_col(features, rename, ["상품군", "category"]),
    "item": resolve_col(features, rename, ["품목명", "item_name"]),
    "price": resolve_col(features, rename, ["기준가격", "price"]),
}

# 요일 인코딩(0~6 vs 1~7) 자동 판별
weekday_base = 0
if COL["weekday"]:
    w_vals = categories.get(COL["weekday"], []) or sample[COL["weekday"]].dropna().tolist()
    try:
        w_nums = [int(float(x)) for x in w_vals]
        if w_nums and min(w_nums) >= 1 and max(w_nums) >= 7:
            weekday_base = 1
    except Exception:
        pass

# ── 2. 매장·품목 마스터 구축 ──
store_cols = [
    c
    for c in [
        COL["store_id"],
        COL["city"],
        COL["store_type"],
        COL["store_size"],
        COL["res_pop"],
        COL["flt_pop"],
        COL["schools"],
    ]
    if c
]
if COL["store_id"] and COL["store_id"] in sample.columns:
    stores_df = sample[store_cols].drop_duplicates(subset=[COL["store_id"]]).copy()
    store_ids = stores_df[COL["store_id"]].astype(str).tolist()
    for s in categories.get(COL["store_id"], []):
        if str(s) not in store_ids:
            store_ids.append(str(s))
else:
    stores_df = pd.DataFrame([default_row])
    store_ids = [str(default_row.get(COL["store_id"], "S001"))]

item_cols = [c for c in [COL["item"], COL["category"], COL["price"]] if c]
if COL["item"] and COL["item"] in sample.columns:
    items_df = sample[item_cols].drop_duplicates(subset=[COL["item"]]).copy()
    items_df[COL["item"]] = items_df[COL["item"]].astype(str)
    all_items = items_df[COL["item"]].tolist()
    for it in categories.get(COL["item"], []):
        if str(it) not in all_items:
            all_items.append(str(it))
            inferred_cat = next(
                (k for k in DEFAULT_PACK if str(it).startswith(k)),
                str(default_row.get(COL["category"], "젤리")),
            )
            items_df = pd.concat(
                [
                    items_df,
                    pd.DataFrame(
                        [
                            {
                                COL["item"]: str(it),
                                COL["category"]: inferred_cat,
                                COL["price"]: default_row.get(COL["price"], 2000),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
else:
    items_df = pd.DataFrame([default_row])
    all_items = [str(default_row.get(COL["item"], "품목1"))]

# ── 3. 사이드바 (최소 설정 + GPS 위치 옵션) ──
st.sidebar.header("🏪 매장 및 발주 설정")
selected_store = st.sidebar.selectbox("매장 선택", store_ids, index=0)

if (
    COL["store_id"]
    and COL["store_id"] in stores_df.columns
    and selected_store in stores_df[COL["store_id"]].astype(str).values
):
    store_info = stores_df[stores_df[COL["store_id"]].astype(str) == selected_store].iloc[0].to_dict()
else:
    store_info = default_row.copy()
    if COL["store_id"]:
        store_info[COL["store_id"]] = selected_store

city_name = str(store_info.get(COL["city"], "대전"))
store_type = str(store_info.get(COL["store_type"], "주거형"))

all_branches = (
    [str(x) for x in categories.get(COL["city"], [])]
    if COL["city"] and categories.get(COL["city"])
    else ([city_name] if city_name else ["대전"])
)

use_gps = st.sidebar.toggle("📍 내 현재 GPS 위치로 날씨·지점 자동 맞추기", value=False)
target_lat, target_lon = resolve_branch_coords(city_name)
location_source = f"매장 지점({city_name})"

if use_gps:
    if not HAS_GEO:
        st.sidebar.warning("`pip install streamlit-geolocation` 설치 시 GPS 사용 가능")
    else:
        with st.sidebar:
            st.caption("아래 버튼을 눌러 브라우저 위치 권한을 허용하세요.")
            loc = streamlit_geolocation()
        if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
            target_lat = float(loc["latitude"])
            target_lon = float(loc["longitude"])
            city_name = find_nearest_branch(target_lat, target_lon, all_branches)
            location_source = f"GPS({target_lat:.3f}, {target_lon:.3f}) → {city_name}"
            st.sidebar.success(f"GPS 수신 완료 → 최근접 지점: **{city_name}**")
        else:
            st.sidebar.info(f"GPS 대기 중 (현재 `{city_name}` 기준 사용 중)")

lead_days = st.sidebar.number_input("다음 입고까지 남은 일수", min_value=1, max_value=7, value=3)

if COL["store_id"] and COL["item"] and COL["store_id"] in sample.columns:
    store_sample_items = (
        sample[sample[COL["store_id"]].astype(str) == selected_store][COL["item"]]
        .astype(str)
        .unique()
        .tolist()
    )
else:
    store_sample_items = []
default_selected_items = (store_sample_items if len(store_sample_items) >= 3 else all_items)[:6]

selected_items = st.sidebar.multiselect(
    "발주 관리 품목",
    options=all_items,
    default=default_selected_items,
    help="발주표에 표시할 품목을 선택하세요.",
)
if not selected_items:
    st.warning("사이드바에서 최소 1개 이상의 품목을 선택해 주세요.")
    st.stop()

with st.sidebar.expander("🌤️ 기상 연동 및 시뮬레이션", expanded=False):
    kma_key = st.text_input(
        "기상청 API 키 (선택)",
        value=get_secret("KMA_API_KEY", ""),
        type="password",
        help="비워두면 Open-Meteo 무료 예보가 자동 사용됩니다.",
    )
    if st.button("기상 데이터 새로고침", use_container_width=True):
        cached_weather.clear()
    override_weather = st.checkbox("가상 기상 테스트 (시연용)")
    sim_temp_offset = st.slider("기온 보정 (°C)", -15.0, 15.0, 0.0, 0.5, disabled=not override_weather)
    sim_rain = st.number_input("강수량 고정 (mm, 0=미적용)", min_value=0.0, value=0.0, disabled=not override_weather)

# ── 4. 현재 시각 기준 미래 7일(168시간) 기상·달력 자동 생성 ──
weather_df = cached_weather(target_lat, target_lon, kma_key, 168, weekday_base).copy()
if override_weather:
    weather_df["기온"] = (weather_df["기온"] + sim_temp_offset).round(1)
    if sim_rain > 0:
        weather_df["강수량"] = sim_rain
    weather_df["불쾌지수"] = [
        calc_discomfort_index(t, h) for t, h in zip(weather_df["기온"], weather_df["습도"])
    ]

now_str = now_kst_naive().strftime("%Y-%m-%d %H:%M KST")
lead_hours = lead_days * 24
w_lead = weather_df.iloc[:lead_hours]
avg_t = float(w_lead["기온"].mean())
max_t = float(w_lead["기온"].max())
avg_h = float(w_lead["습도"].mean())
avg_di = float(w_lead["불쾌지수"].mean())
tot_r = float(w_lead["강수량"].sum())
di_label = "매우높음" if avg_di >= 80 else ("높음" if avg_di >= 75 else ("보통" if avg_di >= 68 else "쾌적"))

st.title("🧊 무잉 발주 도우미")
st.caption(f"{selected_store} 매장 · {now_str} · 다음 입고까지 {lead_days}일")
st.markdown(
    f"🌡️ 앞으로 {lead_days}일 날씨: 평균 **{avg_t:.0f}°C** (최고 {max_t:.0f}°C) · "
    f"비 **{tot_r:.0f}mm** · 습도 {avg_h:.0f}%"
)

# 화면 순서: AI 도우미 → 이번 발주 → 그림 → 직접 고치기 → 분석가용 상세
ai_box = st.container()
order_box = st.container()
viz_box = st.container()
edit_box = st.container()
detail_box = st.container()

# ── 6. 전 품목 × 향후 168시간 일괄 예측 ──
batch_rows = []
item_meta_map = {}

for it_name in selected_items:
    it_info = items_df[items_df[COL["item"]].astype(str) == it_name].iloc[0].to_dict()
    cat_val = str(it_info.get(COL["category"], "젤리"))
    price_val = float(it_info.get(COL["price"], 2000))
    item_meta_map[it_name] = {"category": cat_val, "price": price_val}

    df_it = weather_df.copy()
    mapping = {
        COL["city"]: city_name,
        COL["month"]: df_it["월"],
        COL["day"]: df_it["일"],
        COL["hour"]: df_it["시"],
        COL["temp"]: df_it["기온"],
        COL["rain"]: df_it["강수량"],
        COL["hum"]: df_it["습도"],
        COL["wind"]: df_it["풍속"],
        COL["di"]: df_it["불쾌지수"],
        COL["weekend"]: df_it["주말"],
        COL["holiday"]: df_it["공휴일"],
        COL["long_hol"]: df_it["연휴"],
        COL["vac_s"]: df_it["초중고방학"],
        COL["vac_u"]: df_it["대학방학"],
        COL["weekday"]: df_it["요일"],
        COL["store_id"]: selected_store,
        COL["store_type"]: store_type,
        COL["store_size"]: store_info.get(COL["store_size"], 15.0),
        COL["res_pop"]: store_info.get(COL["res_pop"], 15000),
        COL["flt_pop"]: store_info.get(COL["flt_pop"], 5000),
        COL["schools"]: store_info.get(COL["schools"], 0),
        COL["category"]: cat_val,
        COL["item"]: it_name,
        COL["price"]: price_val,
    }
    for feat_col, val in mapping.items():
        if feat_col:
            df_it[feat_col] = val
    df_it["_item"] = it_name
    df_it["_cat"] = cat_val
    batch_rows.append(df_it)

full_future_df = pd.concat(batch_rows, ignore_index=True)
aligned_X = align_catboost_df(full_future_df, features, cat_cols, categories, default_row)
pred_m, pred_l, pred_h = predict_hourly_batch(mean_model, low_model, high_model, aligned_X)

full_future_df["pred_mean"] = pred_m
full_future_df["pred_low"] = pred_l
full_future_df["pred_high"] = pred_h
full_future_df["hi_sq"] = (full_future_df["pred_high"] - full_future_df["pred_mean"]) ** 2
full_future_df["lo_sq"] = (full_future_df["pred_mean"] - full_future_df["pred_low"]) ** 2
full_future_df["date"] = full_future_df["datetime"].dt.date

# ── 7. 품목별 입고 기간(lead_days) 수요 집계 ──
summary_rows = []
for it_name in selected_items:
    sub = full_future_df[full_future_df["_item"] == it_name].iloc[:lead_hours]
    m_sum = float(sub["pred_mean"].sum())
    hi_sum = m_sum + float(np.sqrt(sub["hi_sq"].sum()))
    lo_sum = max(0.0, m_sum - float(np.sqrt(sub["lo_sq"].sum())))
    summary_rows.append({
        "품목": it_name,
        "상품군": item_meta_map[it_name]["category"],
        "예측(평균)": round(m_sum, 1),
        "하위10%": round(lo_sum, 1),
        "상위90%": round(hi_sum, 1),
    })
pred_summary_df = pd.DataFrame(summary_rows)
# ── 8. 세션 상태 초기화 (매장별 재고, 입고예정, 입수, 점주수정, 로그) ──
if "inventory_state" not in st.session_state:
    st.session_state["inventory_state"] = {}
if "stock_logs" not in st.session_state:
    st.session_state["stock_logs"] = []

for _, r in pred_summary_df.iterrows():
    it_name = r["품목"]
    state_key = f"{selected_store}::{it_name}"
    if state_key not in st.session_state["inventory_state"]:
        init_stock = max(2, int(round(r["예측(평균)"] * 0.6)))
        pack_def = DEFAULT_PACK.get(r["상품군"], 12)
        st.session_state["inventory_state"][state_key] = {
            "재고": init_stock,
            "입고예정": 0,
            "입수": pack_def,
            "점주수정": np.nan,
        }

with order_box:
    st.subheader("① 이번 발주")
    mode_label = st.radio(
        "발주 스타일",
        ["넉넉하게 (품절 걱정 없이)", "보통", "알뜰하게 (남는 재고 적게)"],
        index=1,
        horizontal=True,
    )
mode_col = {
    "넉넉하게 (품절 걱정 없이)": "상위90%",
    "보통": "예측(평균)",
    "알뜰하게 (남는 재고 적게)": "하위10%",
}[mode_label]

table_rows = []
for _, r in pred_summary_df.iterrows():
    it_name = r["품목"]
    st_info = st.session_state["inventory_state"][f"{selected_store}::{it_name}"]
    target = float(r[mode_col])
    stock = int(st_info["재고"])
    inb = int(st_info["입고예정"])
    pack = max(1, int(st_info["입수"]))
    edit_val = st_info["점주수정"]

    needed = max(0.0, math.ceil(target - stock - inb))
    rec_boxes = int(math.ceil(needed / pack))
    final_boxes = rec_boxes if pd.isna(edit_val) else int(max(0, edit_val))

    table_rows.append({
        "품목": it_name,
        "예측(평균)": r["예측(평균)"],
        "하위10%": r["하위10%"],
        "상위90%": r["상위90%"],
        "재고": stock,
        "입고예정": inb,
        "입수": pack,
        "추천(박스)": rec_boxes,
        "점주 수정(박스)": np.nan if pd.isna(edit_val) else float(edit_val),
        "최종": final_boxes,
    })

order_df = pd.DataFrame(table_rows)

with edit_box:
    st.divider()
    st.subheader("③ 재고·주문 직접 고치기")
    st.caption("숫자가 실제와 다르면 칸을 눌러 고쳐 주세요. 위의 추천이 바로 다시 계산돼요.")
    reason = st.selectbox("재고를 고치는 이유", ["실제와 달라서", "파손", "분실", "폐기", "미도착", "원인 미상"])
    edited_df = st.data_editor(
        order_df,
        key=f"order_editor_{selected_store}",
        use_container_width=True,
        hide_index=True,
        column_order=["품목", "재고", "입고예정", "입수", "추천(박스)", "점주 수정(박스)"],
        disabled=["품목", "예측(평균)", "하위10%", "상위90%", "추천(박스)", "최종"],
        column_config={
            "재고": st.column_config.NumberColumn("지금 재고(추정)", min_value=0, step=1, format="%d"),
            "입고예정": st.column_config.NumberColumn("들어올 예정 수량", min_value=0, step=1, format="%d"),
            "입수": st.column_config.NumberColumn("한 박스 개수", min_value=1, step=1, format="%d"),
            "추천(박스)": st.column_config.NumberColumn("추천(박스)", format="%d 박스"),
            "점주 수정(박스)": st.column_config.NumberColumn("내가 정할 박스 수 (비우면 추천대로)", min_value=0, step=1, format="%d"),
        },
    )

    changed = False
    for _, row in edited_df.iterrows():
        it_name = row["품목"]
        prev = st.session_state["inventory_state"][f"{selected_store}::{it_name}"]
        new_stock = int(max(0, 0 if pd.isna(row["재고"]) else row["재고"]))
        new_inb = int(max(0, 0 if pd.isna(row["입고예정"]) else row["입고예정"]))
        new_pack = int(max(1, 1 if pd.isna(row["입수"]) else row["입수"]))
        new_edit = row["점주 수정(박스)"]
        prev_edit = prev["점주수정"]
        edit_changed = (pd.isna(new_edit) != pd.isna(prev_edit)) or (
            not pd.isna(new_edit) and int(new_edit) != int(prev_edit)
        )

        if new_stock != prev["재고"]:
            st.session_state["stock_logs"].insert(
                0,
                f"[{now_kst_naive().strftime('%H:%M')}] {selected_store} · {it_name}: 재고 {prev['재고']}개 → {new_stock}개 ({reason})",
            )
            prev["재고"] = new_stock
            changed = True
        if new_inb != prev["입고예정"] or new_pack != prev["입수"] or edit_changed:
            prev["입고예정"] = new_inb
            prev["입수"] = new_pack
            prev["점주수정"] = np.nan if pd.isna(new_edit) else int(max(0, new_edit))
            changed = True

    if changed:
        st.rerun()

    with st.expander("재고 고친 기록"):
        if st.session_state["stock_logs"]:
            for log in st.session_state["stock_logs"]:
                st.write(f"- {log}")
        else:
            st.caption("아직 고친 기록이 없어요.")


SYSTEM_PROMPT = (
    "당신은 무인 아이스크림·과자 매장 점주를 돕는 발주 도우미입니다. "
    "통계나 기계학습 용어(분위수, 모델, 신뢰구간 등)는 쓰지 말고 누구나 이해할 쉬운 말로 3~5문장 이내로 답하세요. "
    "숫자는 아래 [현재 상황]에 있는 것만 쓰고 새로 계산하거나 지어내지 마세요. "
    "예측은 추정치라서 실제와 다를 수 있다는 점을 부드럽게 알려 주세요. 모르는 것은 모른다고 말하세요."
)

cover_days = (order_df["재고"] + order_df["입고예정"]) / np.maximum(0.1, order_df["예측(평균)"] / lead_days)
risk_mask = cover_days < lead_days
risk_items = order_df.loc[risk_mask, "품목"].tolist()


def build_context() -> str:
    rows = "\n".join(
        f"- {r['품목']}: {lead_days}일 예상 판매 보통 {r['예측(평균)']:.0f}개 "
        f"(적게 {r['하위10%']:.0f}~많이 {r['상위90%']:.0f}개), 지금 재고(추정) {int(r['재고'])}개, "
        f"추천 {int(r['추천(박스)'])}박스, 최종 {int(r['최종'])}박스(한 박스 {int(r['입수'])}개)"
        for _, r in order_df.iterrows()
    )
    return (
        f"매장 {selected_store}({store_type}), 기준시각 {now_str}\n"
        f"앞으로 {lead_days}일 날씨: 평균 {avg_t:.1f}°C(최고 {max_t:.1f}°C), 습도 {avg_h:.0f}%, "
        f"무더위 정도 {di_label}, 비 {tot_r:.1f}mm\n"
        f"발주 스타일: {mode_label}\n"
        f"다음 입고 전에 떨어질 수 있는 품목: {risk_items if risk_items else '없음'}\n{rows}"
    )


def run_chat(question: str):
    api_key = get_secret("GROQ_API_KEY")
    history = st.session_state["chat"][-8:]
    st.session_state["chat"].append({"role": "user", "content": question})
    if not api_key:
        answer = "Streamlit Secrets에 GROQ_API_KEY를 등록하면 AI 도우미를 쓸 수 있어요."
    else:
        try:
            from groq import Groq

            msgs = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n[현재 상황]\n" + build_context()}]
            msgs += history + [{"role": "user", "content": question}]
            res = Groq(api_key=api_key).chat.completions.create(
                model="openai/gpt-oss-120b", messages=msgs, temperature=0.3
            )
            answer = res.choices[0].message.content
        except Exception as e:
            answer = f"AI 도우미 연결에 문제가 생겼어요: {e}"
    st.session_state["chat"].append({"role": "assistant", "content": answer})


with ai_box:
    st.subheader("🤖 AI 도우미")
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q_btn = st.button("오늘 뭘 주문하면 좋을지 알려 주세요", use_container_width=True)
    with st.form("ask_form", clear_on_submit=True):
        q_text = st.text_input("궁금한 점을 적어 주세요", placeholder="예: 비 오는 날은 어떤 게 덜 팔려요?")
        sent = st.form_submit_button("물어보기")
    question = "오늘 발주를 어떻게 하면 좋을지 쉽게 알려 주세요." if q_btn else (q_text.strip() if sent else "")
    if question:
        with st.spinner("생각하는 중이에요..."):
            run_chat(question)
        st.rerun()

# ── 이번 발주 요약 ──
with order_box:
    if risk_items:
        st.warning(f"다음 입고 전에 떨어질 수 있는 품목이 {len(risk_items)}개 있어요: " + ", ".join(risk_items))
    else:
        st.success("지금 재고로 다음 입고까지 버틸 수 있어요.")
    show_df = pd.DataFrame({
        "품목": order_df["품목"],
        "상태": np.where(risk_mask, "🔴 서둘러 주문", "🟢 여유 있음"),
        "지금 재고(추정)": order_df["재고"],
        f"앞으로 {lead_days}일 예상 판매": [f"{a:.0f} ~ {b:.0f}개" for a, b in zip(order_df["하위10%"], order_df["상위90%"])],
        "추천 주문": [f"{int(x)}박스" for x in order_df["추천(박스)"]],
        "최종 주문": [f"{int(x)}박스" for x in order_df["최종"]],
    })
    st.dataframe(show_df, hide_index=True, use_container_width=True)
    if st.button("✅ 이 내용으로 주문 확정", type="primary"):
        confirmed = [f"{r['품목']} {int(r['최종'])}박스" for _, r in order_df.iterrows() if r["최종"] > 0]
        st.session_state["confirmed_msg"] = (
            "주문 확정 완료: " + ", ".join(confirmed) if confirmed else "이번에는 추가로 주문할 품목이 없어요."
        )
    if "confirmed_msg" in st.session_state:
        st.success(st.session_state["confirmed_msg"])


with viz_box:
    st.subheader("② 그림으로 보기")
    viz_item = st.selectbox("품목 고르기", selected_items)
    viz_sub = full_future_df[full_future_df["_item"] == viz_item].copy()
    viz_row = order_df[order_df["품목"] == viz_item].iloc[0]
    tab_a, tab_b, tab_c = st.tabs(["📈 앞으로 판매 예상", "📦 재고로 버티는 날", "🌡️ 날씨·시간대"])

    base_one = full_future_df[full_future_df["_item"] == viz_item].iloc[[0]].copy()
    if COL["hour"]:
        base_one[COL["hour"]] = 18
    with tab_a:
        col_left, col_right = st.columns(2)

        # 차트 A: 향후 7일 일별 예측 구간
        with col_left:
            daily_g = viz_sub.groupby("date", as_index=False).agg(
                mean=("pred_mean", "sum"),
                hi_sq=("hi_sq", "sum"),
                lo_sq=("lo_sq", "sum"),
            )
            daily_g["hi"] = daily_g["mean"] + np.sqrt(daily_g["hi_sq"])
            daily_g["lo"] = np.maximum(0.0, daily_g["mean"] - np.sqrt(daily_g["lo_sq"]))
            daily_g["날짜"] = daily_g["date"].astype(str)

            fig_band = go.Figure()
            if len(daily_g) >= lead_days:
                fig_band.add_vrect(
                    x0=daily_g["날짜"].iloc[0],
                    x1=daily_g["날짜"].iloc[lead_days - 1],
                    fillcolor="rgba(255, 224, 102, 0.28)",
                    layer="below",
                    line_width=0,
                    annotation_text=f"입고 대기 {lead_days}일",
                    annotation_position="top left",
                )
            fig_band.add_trace(
                go.Scatter(
                    x=list(daily_g["날짜"]) + list(daily_g["날짜"][::-1]),
                    y=list(daily_g["hi"]) + list(daily_g["lo"][::-1]),
                    fill="toself",
                    fillcolor="rgba(47, 111, 237, 0.20)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="적게~많이 팔릴 범위",
                )
            )
            fig_band.add_trace(
                go.Scatter(
                    x=daily_g["날짜"],
                    y=daily_g["mean"],
                    mode="lines+markers",
                    line=dict(color="#2f6fed", width=3),
                    name="보통 예상",
                )
            )
            fig_band.update_layout(
                title=f"[{viz_item}] 앞으로 7일, 하루에 몇 개 팔릴까요?",
                yaxis_title="하루 판매 예상(개)",
                height=330,
                margin=dict(t=45, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_band, use_container_width=True)

        # 차트 B: 입고 전 재고 소진 시뮬레이션
        with col_right:
            dep_sub = viz_sub.iloc[:lead_hours].copy()
            cur_stock = float(viz_row["재고"])
            cur_inb = float(viz_row["입고예정"])
            dep_sub["잔여(평균)"] = cur_stock - dep_sub["pred_mean"].cumsum()
            dep_sub["잔여(상위90%)"] = cur_stock - dep_sub["pred_high"].cumsum()

            fig_dep = go.Figure()
            fig_dep.add_hline(y=0, line_color="#d33", line_width=1.5, annotation_text="0개 = 품절")
            fig_dep.add_trace(
                go.Scatter(
                    x=dep_sub["datetime"],
                    y=dep_sub["잔여(평균)"],
                    mode="lines",
                    line=dict(color="#2f6fed", width=2.5),
                    name="보통 팔릴 때",
                )
            )
            fig_dep.add_trace(
                go.Scatter(
                    x=dep_sub["datetime"],
                    y=dep_sub["잔여(상위90%)"],
                    mode="lines",
                    line=dict(color="#d33", width=2, dash="dash"),
                    name="많이 팔릴 때",
                )
            )
            fig_dep.update_layout(
                title=f"[{viz_item}] 다음 입고 전까지 재고가 얼마나 남을까요?",
                yaxis_title="남은 재고(개)",
                height=330,
                margin=dict(t=45, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_dep, use_container_width=True)
    with tab_b:
        c_dys = st.container()
        with c_dys:
            dys_data = []
            for _, r in order_df.iterrows():
                daily_avg = max(0.1, float(r["예측(평균)"]) / lead_days)
                cov_days = (r["재고"] + r["입고예정"]) / daily_avg
                dys_data.append({
                    "품목": r["품목"],
                    "가능일수": round(cov_days, 1),
                    "상태": "입고 전에 떨어질 수 있음" if cov_days < lead_days else "여유 있음",
                })
            dys_df = pd.DataFrame(dys_data)
            fig_dys = px.bar(
                dys_df,
                x="가능일수",
                y="품목",
                orientation="h",
                color="상태",
                text="가능일수",
                color_discrete_map={"입고 전에 떨어질 수 있음": "#d33", "여유 있음": "#2f6fed"},
                title="지금 재고로 며칠 버틸 수 있을까요? (빨강 = 다음 입고 전에 떨어질 수 있어요)",
            )
            fig_dys.add_vline(x=lead_days, line_dash="dash", line_color="#d33")
            fig_dys.update_layout(height=320, margin=dict(t=45, b=20, l=20, r=20))
            st.plotly_chart(fig_dys, use_container_width=True)
    with tab_c:
        c_tp, c_hm = st.columns(2)
        with c_tp:
            temps = list(range(-10, 39, 2))
            tp_df = pd.concat([base_one] * len(temps), ignore_index=True)
            if COL["temp"]:
                tp_df[COL["temp"]] = temps
            if COL["di"]:
                tp_df[COL["di"]] = [calc_discomfort_index(t, 60.0) for t in temps]
            tp_aligned = align_catboost_df(tp_df, features, cat_cols, categories, default_row)
            tp_preds = np.maximum(0.0, mean_model.predict(tp_aligned, prediction_type="Exponent"))
            base_20 = tp_preds[temps.index(20)] if 20 in temps else tp_preds.mean()
            tp_idx = tp_preds / max(1e-6, base_20) * 100

            fig_tp = go.Figure()
            fig_tp.add_hline(y=100, line_dash="dot", line_color="gray", annotation_text="20°C 기준(100)")
            fig_tp.add_trace(go.Scatter(x=temps, y=tp_idx, mode="lines+markers", line=dict(color="#2f6fed", width=2.5), name="기온별 지수"))
            fig_tp.update_layout(
                title="기온에 따라 얼마나 팔릴까요? (20°C일 때 = 100)",
                xaxis_title="기온 (°C)",
                yaxis_title="판매 정도",
                height=300,
                margin=dict(t=45, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_tp, use_container_width=True)
        with c_hm:
            hm_sub = viz_sub.copy()
            hm_sub["요일명"] = hm_sub["요일_idx"].map(lambda i: WEEKDAY_KR[i])
            hm_piv = (
                hm_sub.pivot_table(index="요일명", columns="시", values="pred_mean", aggfunc="mean")
                .reindex(index=WEEKDAY_KR, columns=list(range(24)))
            )
            fig_hm = px.imshow(
                hm_piv,
                aspect="auto",
                color_continuous_scale="Blues",
                labels=dict(x="시간(시)", y="요일", color="판매 정도"),
                title=f"[{viz_item}] 언제 많이 팔릴까요? (진할수록 많이)",
            )
            fig_hm.update_layout(height=320, margin=dict(t=45, b=20, l=20, r=20))
            st.plotly_chart(fig_hm, use_container_width=True)

with detail_box:
    with st.expander("🔬 분석가용 상세 (모델·기상 데이터)"):
        preview_cols = [
            "datetime", "월", "일", "시", "기온", "강수량", "습도", "풍속",
            "불쾌지수", "주말", "공휴일", "연휴", "기상출처",
        ]
        st.caption("모델에 자동으로 들어가는 미래 24시간 기상·달력 데이터")
        st.dataframe(weather_df[preview_cols].head(24), use_container_width=True)
        c_mo = st.container()

        with c_mo:
            tm_list = [-1, 1, 6, 13, 18, 22, 26, 26, 21, 14, 7, 1]
            mo_df = pd.concat([base_one] * 12, ignore_index=True)
            if COL["month"]:
                mo_df[COL["month"]] = list(range(1, 13))
            if COL["temp"]:
                mo_df[COL["temp"]] = tm_list
            if COL["di"]:
                mo_df[COL["di"]] = [calc_discomfort_index(t, 60.0) for t in tm_list]
            mo_aligned = align_catboost_df(mo_df, features, cat_cols, categories, default_row)
            mo_preds = np.maximum(0.0, mean_model.predict(mo_aligned, prediction_type="Exponent"))
            mo_idx = mo_preds / max(1e-6, mo_preds.mean()) * 100

            fig_mo = go.Figure()
            fig_mo.add_trace(
                go.Bar(x=[f"{m}월" for m in range(1, 13)], y=mo_idx, name="판매 지수(평균=100)", marker_color="#2f6fed", opacity=0.75)
            )
            fig_mo.add_trace(
                go.Scatter(x=[f"{m}월" for m in range(1, 13)], y=tm_list, name="월평균기온(°C)", yaxis="y2", line=dict(color="#1a7f4b", width=2.5))
            )
            fig_mo.update_layout(
                title="월별 판매 정도와 기온",
                yaxis=dict(title="판매 지수"),
                yaxis2=dict(title="기온(°C)", overlaying="y", side="right"),
                height=300,
                margin=dict(t=45, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_mo, use_container_width=True)
        c_rk = st.container()
        with c_rk:
            risk_rows = []
            for label, col_k in [("품절 방지(90%)", "상위90%"), ("기본(평균)", "예측(평균)"), ("폐기 최소(10%)", "하위10%")]:
                left_tot, short_tot = 0.0, 0.0
                for _, r in order_df.iterrows():
                    need_q = max(0.0, math.ceil(r[col_k] - r["재고"] - r["입고예정"]))
                    ord_u = math.ceil(need_q / max(1, r["입수"])) * max(1, r["입수"])
                    avail = r["재고"] + r["입고예정"] + ord_u
                    left_tot += max(0.0, avail - r["예측(평균)"])
                    short_tot += max(0.0, r["상위90%"] - avail)
                risk_rows.append({"기준": label, "구분": "평균 수요 시 남는 재고", "수량(개)": round(left_tot, 1)})
                risk_rows.append({"기준": label, "구분": "상위90% 수요 시 부족분", "수량(개)": round(short_tot, 1)})
            fig_rk = px.bar(
                pd.DataFrame(risk_rows),
                x="수량(개)",
                y="기준",
                color="구분",
                barmode="group",
                orientation="h",
                color_discrete_map={"평균 수요 시 남는 재고": "#2f6fed", "상위90% 수요 시 부족분": "#d33"},
                title="발주 기준별 위험 비교 (전 품목 합계)",
            )
            fig_rk.update_layout(height=320, margin=dict(t=45, b=20, l=20, r=20))
            st.plotly_chart(fig_rk, use_container_width=True)
        c_imp = st.container()
        with c_imp:
            imps = mean_model.get_feature_importance()
            imp_df = (
                pd.DataFrame({"변수": [rename.get(f, f) for f in features], "중요도": imps})
                .sort_values("중요도", ascending=True)
                .tail(10)
            )
            fig_imp = px.bar(
                imp_df,
                x="중요도",
                y="변수",
                orientation="h",
                title="모델이 중요하게 본 변수 상위 10개",
            )
            fig_imp.update_traces(marker_color="#2f6fed")
            fig_imp.update_layout(height=320, margin=dict(t=45, b=20, l=20, r=20))
            st.plotly_chart(fig_imp, use_container_width=True)

st.caption("시연용 프로토타입입니다. 예상은 실제 판매를 보장하지 않으니, 주문 전에 실제 재고와 매장 상황을 확인하세요.")
