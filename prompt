import json
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

st.set_page_config(page_title="무잉 발주 추천", page_icon="🧊", layout="wide")
st.title("무잉 발주 추천 프로토타입")
st.caption("판매량 예측 모델 기반 발주 의사결정 지원")

MODEL_DIR = "."


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


mean_model, low_model, high_model, meta = load_models()
sample = load_sample()
features = meta["features"]
cat_cols = meta.get("cat_cols", [])
categories = meta.get("categories", {})
rename = meta.get("rename", {})

# Use a test row only to provide realistic initial values; user can change them.
default_row = sample.iloc[0].to_dict()
st.sidebar.header("예측 입력")
input_data = {}

for col in features:
    label = rename.get(col, col)
    default = default_row.get(col)
    if col in cat_cols:
        choices = categories.get(col, [])
        if not choices:
            choices = sorted(sample[col].dropna().astype(str).unique().tolist())
        default_value = default if pd.notna(default) else (choices[0] if choices else "")
        if default_value not in choices and choices:
            default_value = choices[0]
        input_data[col] = st.sidebar.selectbox(
            label, choices, index=choices.index(default_value) if choices else 0
        )
    else:
        try:
            default_num = float(default)
        except (TypeError, ValueError):
            default_num = 0.0
        input_data[col] = st.sidebar.number_input(
            label, value=default_num, format="%.4f"
        )

input_df = pd.DataFrame([input_data], columns=features)

st.subheader("입력 데이터")
st.dataframe(
    pd.DataFrame([{rename.get(k, k): v for k, v in input_data.items()}]),
    use_container_width=True,
)

if st.button("판매량 예측", type="primary", use_container_width=True):
    mean = float(mean_model.predict(input_df, prediction_type="Exponent")[0])
    low = float(low_model.predict(input_df)[0])
    high = float(high_model.predict(input_df)[0])

    st.session_state["prediction"] = {
        "mean": max(0, mean),
        "low": max(0, low),
        "high": max(0, high),
        "input": input_data,
    }

if "prediction" in st.session_state:
    pred = st.session_state["prediction"]
    st.subheader("예측 결과")
    c1, c2, c3 = st.columns(3)
    c1.metric("평균 예상 판매량", f'{pred["mean"]:.2f}개')
    c2.metric("하위 10% 예측", f'{pred["low"]:.2f}개')
    c3.metric("상위 90% 예측", f'{pred["high"]:.2f}개')

    st.markdown("### 발주 추천")
    days = st.number_input("다음 입고까지 남은 일수", min_value=1, max_value=30, value=3)
    stock = st.number_input("현재 재고(개)", min_value=0, value=10)
    incoming = st.number_input("입고 예정 수량(개)", min_value=0, value=0)
    pack = st.number_input("박스당 입수(개)", min_value=1, value=12)
    mode = st.radio(
        "발주 기준",
        ["기본(평균)", "품절 방지(상위 90%)", "폐기 최소(하위 10%)"],
        horizontal=True,
    )

    selected = {
        "기본(평균)": pred["mean"],
        "품절 방지(상위 90%)": pred["high"],
        "폐기 최소(하위 10%)": pred["low"],
    }[mode]
    expected_demand = selected * days
    needed = max(0, expected_demand - stock - incoming)
    boxes = int((needed + pack - 1) // pack)
    order_qty = boxes * pack

    a, b, c = st.columns(3)
    a.metric(f"{days}일 예상 수요", f"{expected_demand:.1f}개")
    b.metric("추천 발주", f"{boxes}박스")
    c.metric(
        "발주 후 예상 잔여", f"{stock + incoming + order_qty - expected_demand:.1f}개"
    )

    st.write(
        f"계산: {expected_demand:.1f}개 예상 수요 - 재고 {stock}개 - 입고 예정 {incoming}개 "
        f"= 부족 {needed:.1f}개 → {pack}개입 기준 {boxes}박스"
    )

    st.markdown("### Groq AI 추천 근거")
    if st.button("추천 근거 설명 생성"):
        try:
            from groq import Groq

            api_key = st.secrets["GROQ_API_KEY"]
            client = Groq(api_key=api_key)
            prompt = f"""무인 매장 점주에게 발주 판단을 쉽게 돕는 브리핑을 한국어로 작성하세요.
계산식이나 발주 수량 도출 과정을 다시 나열하지 말고, 점주가 직관적으로 이해할 수 있게 쉽게 설명하세요.
예측값은 추정치일 뿐 실제 판매를 보장하지 않음을 자연스럽게 반영하세요.

[매장 및 품목 정보]
- 입력 조건: {pred['input']}

[수요 예측 및 발주 계산 결과]
- 1일 평균 예측: {pred['mean']:.2f}개 (낮은 시나리오 10% 분위수: {pred['low']:.2f}개, 높은 시나리오 90% 분위수: {pred['high']:.2f}개)
- 적용 기준: {mode}
- 다음 입고까지 남은 일수: {days}일
- 기간 내 예상 수요: {expected_demand:.2f}개
- 현재 재고: {stock}개 / 입고 예정: {incoming}개
- 최종 추천 발주: {boxes}박스 ({order_qty}개)

아래 형식으로 4~5줄 이내로 간결하게 작성해 주세요.
- **한 줄 요약:** 현재 재고 상태와 추천 발주 판단을 쉬운 말로 한 문장 요약
- **수요 변동성:** 평균 대비 낮게/높게 팔릴 가능성의 폭을 점주 눈높이로 설명
- **확인할 점:** 날씨·시간대·재고 오차 등 현장에서 눈여겨볼 포인트 1가지
"""

            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 무인 매장 점주를 돕는 발주 해석 도우미입니다. 계산식을 반복하지 말고 주어진 수치의 의미와 현장 리스크를 쉬운 말로 풀어주세요.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            st.write(response.choices[0].message.content)
        except KeyError:
            st.error("Streamlit Secrets에 GROQ_API_KEY를 등록하세요.")
        except Exception as e:
            st.error(f"Groq 호출 오류: {e}")

st.divider()
st.caption(
    "시연용 프로토타입입니다. 예측값은 실제 판매량을 보장하지 않으며, 발주 전 재고와 운영 상황을 확인하세요."
)
