# weather_auto.py
import math
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import holidays
import pandas as pd
import requests
import streamlit as st

try:
    from zoneinfo import ZoneInfo
    KST_TZ = ZoneInfo("Asia/Seoul")
except Exception:
    KST_TZ = timezone(timedelta(hours=9))

# 기상청 ASOS 주요/특수 지점 공식 위·경도 (지오코딩으로 찾기 어려운 관측소명 포함)
KMA_ASOS_COORDS = {
    "속초": (38.25085, 128.56473), "북춘천": (37.94738, 127.75443), "철원": (38.14787, 127.30420),
    "동두천": (37.90188, 127.06070), "파주": (37.88589, 126.76648), "대관령": (37.67713, 128.71834),
    "춘천": (37.90262, 127.73570), "백령도": (37.97396, 124.71237), "북강릉": (37.80456, 128.85535),
    "강릉": (37.75147, 128.89099), "동해": (37.50709, 129.12433), "서울": (37.57142, 126.96580),
    "인천": (37.47772, 126.62490), "원주": (37.33749, 127.94659), "울릉도": (37.48129, 130.89863),
    "수원": (37.25746, 126.98300), "영월": (37.18126, 128.45743), "충주": (36.97045, 127.95250),
    "서산": (36.77658, 126.49390), "울진": (36.99176, 129.41278), "청주": (36.63924, 127.44066),
    "대전": (36.37199, 127.37210), "추풍령": (36.22025, 127.99458), "안동": (36.57293, 128.70733),
    "상주": (36.40837, 128.15741), "포항": (36.03201, 129.38002), "군산": (36.00530, 126.76135),
    "대구": (35.87797, 128.65296), "전주": (35.84092, 127.11718), "울산": (35.58237, 129.33469),
    "창원": (35.17019, 128.57282), "광주": (35.17294, 126.89156), "부산": (35.10468, 129.03203),
    "통영": (34.84541, 128.43561), "목포": (34.81732, 126.38151), "여수": (34.73929, 127.74063),
    "흑산도": (34.68719, 125.45105), "완도": (34.39590, 126.70182), "고창": (35.34824, 126.59900),
    "순천": (35.02040, 127.36940), "홍성": (36.65759, 126.68772), "서청주": (36.63972, 127.39694),
    "제주": (33.51411, 126.52969), "고산": (33.29382, 126.16283), "성산": (33.38677, 126.88020),
    "서귀포": (33.24616, 126.56530), "진주": (35.16378, 128.04004), "강화": (37.70739, 126.44634),
    "양평": (37.48863, 127.49446), "이천": (37.26399, 127.48421), "인제": (38.05986, 128.16714),
    "홍천": (37.68360, 127.88043), "태백": (37.17038, 128.98929), "장흥": (34.68886, 126.91951),
    "해남": (34.55375, 126.56907), "고흥": (34.61826, 127.27572), "의령군": (35.32258, 128.28812),
    "북창원": (35.22655, 128.67260), "양산시": (35.30737, 129.02009), "보성군": (34.76335, 127.21226),
    "강진군": (34.64457, 126.78408), "독도": (37.23950, 131.86980), "세종": (36.48000, 127.28900),
}


def now_kst_naive() -> datetime:
    return datetime.now(KST_TZ).replace(tzinfo=None, minute=0, second=0, microsecond=0)


def calc_discomfort_index(temp_c: float, humidity_pct: float) -> float:
    return round(0.81 * temp_c + 0.01 * humidity_pct * (0.99 * temp_c - 14.3) + 46.3, 1)


def latlon_to_kma_grid(lat: float, lon: float) -> tuple[int, int]:
    """위도·경도 -> 기상청 단기예보 격자(nx, ny) 공식 변환 (전국 어디든 자동 계산)"""
    RE, GRID = 6371.00877, 5.0
    SLAT1, SLAT2 = 30.0, 60.0
    OLON, OLAT = 126.0, 38.0
    XO, YO = 43, 136

    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1, slat2 = SLAT1 * DEGRAD, SLAT2 * DEGRAD
    olon, olat = OLON * DEGRAD, OLAT * DEGRAD

    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = (math.tan(math.pi * 0.25 + slat1 * 0.5) ** sn) * math.cos(slat1) / sn
    ro = re * sf / (math.tan(math.pi * 0.25 + olat * 0.5) ** sn)

    ra = re * sf / (math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5) ** sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn

    nx = int(math.floor(ra * math.sin(theta) + XO + 0.5))
    ny = int(math.floor(ro - ra * math.cos(theta) + YO + 0.5))
    return nx, ny


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@st.cache_data(ttl=86400)
def resolve_branch_coords(branch_name: str) -> tuple[float, float]:
    """91개 지점명 어떤 이름이 들어와도 위·경도 자동 탐색"""
    name = str(branch_name).strip()
    if name in KMA_ASOS_COORDS:
        return KMA_ASOS_COORDS[name]

    # 괄호/접미사 정리 (예: '대구(기)' -> '대구', '경주시' -> '경주')
    clean = re.sub(r"\(.*?\)", "", name).strip()
    if clean in KMA_ASOS_COORDS:
        return KMA_ASOS_COORDS[clean]

    for query in (clean, re.sub(r"(시|군)$", "", clean)):
        if not query:
            continue
        if query in KMA_ASOS_COORDS:
            return KMA_ASOS_COORDS[query]
        try:
            res = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": query, "count": 5, "language": "ko", "format": "json"},
                timeout=4,
            ).json()
            for item in res.get("results", []):
                if item.get("country_code") == "KR":
                    return float(item["latitude"]), float(item["longitude"])
        except Exception:
            pass

    return KMA_ASOS_COORDS["대전"]


def find_nearest_branch(lat: float, lon: float, branch_list: list[str]) -> str:
    """GPS 좌표(lat, lon)와 가장 가까운 지점명(91개 중 1개) 자동 매칭"""
    best_name, best_dist = branch_list[0], float("inf")
    for b in branch_list:
        b_lat, b_lon = resolve_branch_coords(b)
        d = _haversine_km(lat, lon, b_lat, b_lon)
        if d < best_dist:
            best_dist = d
            best_name = b
    return best_name


def _parse_kma_pcp(val: str) -> float:
    if not val or "없음" in val:
        return 0.0
    if "미만" in val:
        return 0.5
    nums = re.findall(r"[\d.]+", val)
    if not nums:
        return 0.0
    if "~" in val and len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2.0
    return float(nums[0])


def _get_latest_kma_base(now_dt: datetime):
    ref = now_dt - timedelta(minutes=15)
    base_hours = [2, 5, 8, 11, 14, 17, 20, 23]
    valid = [h for h in base_hours if h <= ref.hour]
    if valid:
        return ref.strftime("%Y%m%d"), f"{valid[-1]:02d}00"
    prev = ref - timedelta(days=1)
    return prev.strftime("%Y%m%d"), "2300"


def fetch_kma_short_forecast(lat: float, lon: float, service_key: str) -> pd.DataFrame:
    if not service_key:
        return pd.DataFrame()
    nx, ny = latlon_to_kma_grid(lat, lon)
    base_date, base_time = _get_latest_kma_base(now_kst_naive())
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {
        "serviceKey": unquote(service_key),
        "pageNo": 1,
        "numOfRows": 1000,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }
    try:
        res = requests.get(url, params=params, timeout=5)
        items = res.json()["response"]["body"]["items"]["item"]
    except Exception:
        return pd.DataFrame()

    rows = {}
    for it in items:
        cat = it["category"]
        if cat not in ("TMP", "REH", "WSD", "PCP"):
            continue
        dt_str = f"{it['fcstDate']}{it['fcstTime']}"
        rows.setdefault(dt_str, {})[cat] = it["fcstValue"]

    records = []
    for dt_str, vals in rows.items():
        if "TMP" not in vals or "REH" not in vals:
            continue
        records.append({
            "datetime": datetime.strptime(dt_str, "%Y%m%d%H%M"),
            "기온": float(vals["TMP"]),
            "습도": float(vals["REH"]),
            "풍속": float(vals.get("WSD", 1.0)),
            "강수량": _parse_kma_pcp(vals.get("PCP", "강수없음")),
            "기상출처": f"기상청(격자 {nx},{ny})",
        })
    return pd.DataFrame(records).sort_values("datetime") if records else pd.DataFrame()


def fetch_open_meteo_forecast(lat: float, lon: float, days: int = 8) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "wind_speed_unit": "ms",
        "timezone": "Asia/Seoul",
        "forecast_days": min(16, days + 1),
    }
    try:
        res = requests.get(url, params=params, timeout=6).json()
        h = res["hourly"]
        return pd.DataFrame({
            "datetime": pd.to_datetime(h["time"]),
            "기온": h["temperature_2m"],
            "습도": h["relative_humidity_2m"],
            "강수량": h["precipitation"],
            "풍속": h["wind_speed_10m"],
            "기상출처": f"Open-Meteo({lat:.2f},{lon:.2f})",
        })
    except Exception:
        return pd.DataFrame()


def get_future_weather_calendar(
    lat: float,
    lon: float,
    kma_api_key: str = "",
    hours: int = 168,
    weekday_base: int = 0,
) -> pd.DataFrame:
    """위·경도 기준으로 미래 N시간 기상 + 불쾌지수 + 달력/방학 피처 자동 생성"""
    start = now_kst_naive()
    future_times = [start + timedelta(hours=i) for i in range(1, hours + 1)]
    base_df = pd.DataFrame({"datetime": future_times})

    kma_df = fetch_kma_short_forecast(lat, lon, kma_api_key)
    om_df = fetch_open_meteo_forecast(lat, lon, days=max(8, hours // 24 + 2))

    if not kma_df.empty and not om_df.empty:
        merged = base_df.merge(kma_df, on="datetime", how="left")
        merged = merged.set_index("datetime").combine_first(om_df.set_index("datetime")).reset_index()
        merged = base_df.merge(merged, on="datetime", how="left")
    elif not kma_df.empty:
        merged = base_df.merge(kma_df, on="datetime", how="left")
    elif not om_df.empty:
        merged = base_df.merge(om_df, on="datetime", how="left")
    else:
        merged = base_df.copy()
        merged["기온"], merged["습도"], merged["강수량"], merged["풍속"], merged["기상출처"] = 20.0, 60.0, 0.0, 1.5, "기본값"

    merged["기온"] = merged["기온"].ffill().bfill().fillna(20.0).round(1)
    merged["습도"] = merged["습도"].ffill().bfill().fillna(60.0).round(0)
    merged["풍속"] = merged["풍속"].ffill().bfill().fillna(1.5).round(1)
    merged["강수량"] = merged["강수량"].fillna(0.0).round(1)
    merged["기상출처"] = merged["기상출처"].fillna("자동보간")
    merged["불쾌지수"] = [calc_discomfort_index(t, h) for t, h in zip(merged["기온"], merged["습도"])]

    kr_hol = holidays.KR()
    dates = merged["datetime"].dt.date

    def is_rest(d):
        return (d in kr_hol) or (d.weekday() >= 5)

    merged["월"] = merged["datetime"].dt.month
    merged["일"] = merged["datetime"].dt.day
    merged["시"] = merged["datetime"].dt.hour
    merged["요일_idx"] = merged["datetime"].dt.dayofweek
    merged["요일"] = merged["요일_idx"] + weekday_base
    merged["주말"] = (merged["요일_idx"] >= 5).astype(int)
    merged["공휴일"] = [1 if d in kr_hol else 0 for d in dates]

    long_hol = []
    for d in dates:
        if d in kr_hol and (is_rest(d - timedelta(days=1)) or is_rest(d + timedelta(days=1))):
            long_hol.append(1)
        elif d.weekday() >= 5 and any((d + timedelta(days=k)) in kr_hol for k in (-2, -1, 1, 2)):
            long_hol.append(1)
        else:
            long_hol.append(0)
    merged["연휴"] = long_hol

    # 기존 학습 모델(.cbm) 호환을 위해 백그라운드에서만 계산 (UI에는 비노출)
    merged["초중고방학"] = [
        1 if ((dt.month == 7 and dt.day >= 20) or (dt.month == 8 and dt.day <= 20) or (dt.month in (1, 2)) or (dt.month == 12 and dt.day >= 26)) else 0
        for dt in merged["datetime"]
    ]
    merged["대학방학"] = [
        1 if ((dt.month == 6 and dt.day >= 22) or (dt.month in (7, 8, 1, 2)) or (dt.month == 12 and dt.day >= 20)) else 0
        for dt in merged["datetime"]
    ]
    return merged
