
import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import plotly.express as px
import re

from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="경주 AI 적정 임대료 분석",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 경주 AI 주거비 분석 시스템")
st.subheader("실거래 데이터 기반 적정 임대료 예측 및 협상 지원")

st.caption(
    "경주시 실제 전월세 실거래 데이터를 분석하여 "
    "주거비 비교와 합리적인 임대료 판단을 지원합니다."
)


# ==========================================
# 2. API 인증키
# ==========================================

try:
    SH_KEY = st.secrets["api"]["SH_KEY"]
    RH_KEY = st.secrets["api"]["RH_KEY"]
    APT_KEY = st.secrets["api"]["APT_KEY"]

except Exception as e:
    st.error(f"인증키 설정 오류: {e}")
    st.stop()


# ==========================================
# 3. API 정보
# ==========================================

API_LIST = [
    {
        "name": "단독·다가구",
        "type": "단독·다가구",
        "url": "https://apis.data.go.kr/1613000/RTMSDataSvcSHRent/getRTMSDataSvcSHRent",
        "key": SH_KEY
    },
    {
        "name": "연립·다세대",
        "type": "연립·다세대",
        "url": "https://apis.data.go.kr/1613000/RTMSDataSvcRHRent/getRTMSDataSvcRHRent",
        "key": RH_KEY
    },
    {
        "name": "아파트",
        "type": "아파트",
        "url": "https://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent",
        "key": APT_KEY
    }
]


# ==========================================
# 4. 데이터 변환 함수
# ==========================================

def normalize_key(key):
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def get_field(item, aliases):
    values = {
        normalize_key(child.tag): child.text
        for child in item
    }

    for alias in aliases:
        value = values.get(normalize_key(alias))

        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def to_number(value):
    if value is None:
        return 0

    value = str(value).replace(",", "").strip()
    match = re.search(r"-?\d+(\.\d+)?", value)

    if not match:
        return 0

    try:
        return float(match.group())
    except ValueError:
        return 0


def make_month_list(count):
    now = datetime.now()
    result = []

    year = now.year
    month = now.month

    for _ in range(count):
        result.append(f"{year}{month:02d}")

        month -= 1

        if month == 0:
            month = 12
            year -= 1

    return result


# ==========================================
# 5. 실제 국토교통부 데이터 수집
# ==========================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_real_data(months, sh_key, rh_key, apt_key):

    api_keys = [sh_key, rh_key, apt_key]

    records = []
    errors = []

    for api, key in zip(API_LIST, api_keys):

        for deal_ymd in months:

            params = {
                "serviceKey": key,
                "LAWD_CD": "47130",
                "DEAL_YMD": deal_ymd,
                "pageNo": "1",
                "numOfRows": "1000",
                "_type": "xml"
            }

            try:
                response = requests.get(
                    api["url"],
                    params=params,
                    timeout=30
                )

                response.raise_for_status()
                root = ET.fromstring(response.content)

                result_code = root.findtext(
                    ".//header/resultCode",
                    default=""
                )

                result_msg = root.findtext(
                    ".//header/resultMsg",
                    default=""
                )

                if result_code not in ("00", "000"):
                    errors.append(
                        f"{api['name']} / {deal_ymd}: "
                        f"{result_code} / {result_msg}"
                    )
                    continue

                items = root.findall(".//body/items/item")

                for item in items:

                    dong = get_field(
                        item,
                        ["umdNm", "법정동", "법정동명", "dong"]
                    )

                    area = to_number(
                        get_field(
                            item,
                            [
                                "excluUseAr",
                                "exclusiveArea",
                                "area",
                                "전용면적"
                            ]
                        )
                    )

                    deposit = to_number(
                        get_field(
                            item,
                            ["deposit", "보증금액", "보증금"]
                        )
                    )

                    rent = to_number(
                        get_field(
                            item,
                            ["monthlyRent", "월세금액", "월세"]
                        )
                    )

                    year = get_field(
                        item,
                        ["dealYear", "계약년도"]
                    )

                    month = get_field(
                        item,
                        ["dealMonth", "계약월"]
                    )

                    day = get_field(
                        item,
                        ["dealDay", "계약일"]
                    )

                    if not dong or area <= 0:
                        continue

                    records.append({
                        "지역": dong,
                        "주택유형": api["type"],
                        "면적": area,
                        "보증금": deposit,
                        "월세": rent,
                        "계약년도": year,
                        "계약월": month,
                        "계약일": day,
                        "계약년월": deal_ymd,
                        "건물명": get_field(
                            item,
                            ["aptNm", "아파트", "mhouseNm", "건물명"]
                        ),
                        "층": get_field(
                            item,
                            ["floor", "층"]
                        )
                    })

            except Exception as e:
                errors.append(
                    f"{api['name']} / {deal_ymd}: {str(e)}"
                )

    columns = [
        "지역", "주택유형", "면적", "보증금", "월세",
        "계약년도", "계약월", "계약일",
        "계약년월", "건물명", "층"
    ]

    df = pd.DataFrame(records, columns=columns)

    return df, errors


# ==========================================
# 6. 데이터 조회 기간
# ==========================================

st.sidebar.header("⚙️ 분석 설정")

month_count = st.sidebar.selectbox(
    "실거래 데이터 조회 기간",
    [3, 6, 9, 12],
    index=1,
    format_func=lambda x: f"최근 {x}개월"
)

months = make_month_list(month_count)

st.sidebar.caption(
    "데이터는 국토교통부에 공개된 실거래 자료를 기준으로 합니다."
)

if st.sidebar.button("실거래 데이터 새로고침"):
    st.cache_data.clear()
    st.rerun()


with st.spinner("경주시 실제 전월세 데이터를 수집하고 있습니다..."):

    df, api_errors = load_real_data(
        tuple(months),
        SH_KEY,
        RH_KEY,
        APT_KEY
    )


# ==========================================
# 7. 데이터 확인
# ==========================================

if df.empty:

    st.error("조회된 실거래 데이터가 없습니다.")

    if api_errors:
        st.write("API 오류 내역")
        for error in api_errors:
            st.write(error)

    st.info(
        "조회 기간을 늘리거나 잠시 후 다시 시도해 주세요."
    )

    st.stop()


for col in ["면적", "보증금", "월세"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.fillna({
    "면적": 0,
    "보증금": 0,
    "월세": 0
})

# 실제 월세 거래만 예측과 시세 통계에 사용
rent_df = df[
    (df["월세"] > 0) &
    (df["면적"] > 0)
].copy()


# ==========================================
# 8. 데이터 현황
# ==========================================

st.success("경주시 실거래 데이터 수집 완료")

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 조회 거래", f"{len(df):,}건")
col2.metric("월세 거래", f"{len(rent_df):,}건")
col3.metric("조회 지역", f"{df['지역'].nunique()}개")
col4.metric("조회 기간", f"{month_count}개월")

if api_errors:
    with st.expander("일부 API 조회 오류 확인"):
        for error in api_errors:
            st.write(error)


# ==========================================
# 9. 경주 동네별 월세 시세 분석
# ==========================================

st.divider()
st.header("📍 경주 동네별 월세 시세")
st.write(
    "실제 월세 거래를 기준으로 동네별 임대료 수준을 비교합니다. "
    "거래가 많은 지역일수록 통계 참고 가치가 높습니다."
)

if rent_df.empty:

    st.warning(
        "현재 조회 기간에 월세 거래 데이터가 없어 "
        "동네별 월세 통계를 표시할 수 없습니다."
    )

else:

    # 분석 필터
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        market_type = st.selectbox(
            "비교할 주택 유형",
            ["전체 주택"] + sorted(rent_df["주택유형"].unique().tolist()),
            key="market_type"
        )

    with filter_col2:
        area_band = st.selectbox(
            "비교할 주택 면적",
            [
                "전체 면적",
                "10㎡ 미만",
                "10~20㎡",
                "20~30㎡",
                "30~40㎡",
                "40㎡ 이상"
            ],
            key="market_area"
        )

    market_df = rent_df.copy()

    if market_type != "전체 주택":
        market_df = market_df[
            market_df["주택유형"] == market_type
        ]

    if area_band == "10㎡ 미만":
        market_df = market_df[market_df["면적"] < 10]

    elif area_band == "10~20㎡":
        market_df = market_df[
            (market_df["면적"] >= 10) &
            (market_df["면적"] < 20)
        ]

    elif area_band == "20~30㎡":
        market_df = market_df[
            (market_df["면적"] >= 20) &
            (market_df["면적"] < 30)
        ]

    elif area_band == "30~40㎡":
        market_df = market_df[
            (market_df["면적"] >= 30) &
            (market_df["면적"] < 40)
        ]

    elif area_band == "40㎡ 이상":
        market_df = market_df[market_df["면적"] >= 40]

    if market_df.empty:

        st.warning(
            "선택한 주택 유형과 면적 조건에 해당하는 거래가 없습니다. "
            "필터를 변경해 주세요."
        )

    else:

        # 동네별 통계 계산
        market_summary = (
            market_df
            .groupby("지역")
            .agg(
                거래건수=("월세", "count"),
                월세중앙값=("월세", "median"),
                평균월세=("월세", "mean"),
                평균면적=("면적", "mean"),
                평균보증금=("보증금", "mean")
            )
            .reset_index()
        )

        market_summary = market_summary.sort_values(
            "월세중앙값",
            ascending=False
        )

        # 전체 동네 통계
        m1, m2, m3 = st.columns(3)

        m1.metric(
            "비교 대상 동네",
            f"{len(market_summary)}개"
        )

        m2.metric(
            "비교 월세 거래",
            f"{len(market_df):,}건"
        )

        m3.metric(
            "경주 전체 월세 중앙값",
            f"{market_df['월세'].median():,.1f}만원"
        )

        st.subheader("동네별 월세 비교")

        chart_df = market_summary.sort_values(
            "월세중앙값",
            ascending=True
        )

        fig_market = px.bar(
            chart_df,
            x="월세중앙값",
            y="지역",
            orientation="h",
            text="월세중앙값",
            hover_data={
                "거래건수": True,
                "평균월세": ":.1f",
                "평균면적": ":.1f",
                "월세중앙값": ":.1f"
            },
            labels={
                "월세중앙값": "월세 중앙값 (만원)",
                "지역": "동네",
                "거래건수": "거래 건수",
                "평균월세": "평균 월세 (만원)",
                "평균면적": "평균 면적 (㎡)"
            },
            title="경주시 동네별 월세 중앙값"
        )

        fig_market.update_traces(
            texttemplate="%{text:.1f}만원",
            textposition="outside"
        )

        st.plotly_chart(
            fig_market,
            use_container_width=True
        )

        st.caption(
            "중앙값은 거래 금액을 순서대로 나열했을 때 가운데 값입니다. "
            "일부 고액 거래의 영향을 평균보다 덜 받습니다."
        )

        # 동네별 통계 표
        st.subheader("동네별 상세 시세")

        display_summary = market_summary.copy()

        display_summary["월세중앙값"] = (
            display_summary["월세중앙값"].round(1)
        )
        display_summary["평균월세"] = (
            display_summary["평균월세"].round(1)
        )
        display_summary["평균면적"] = (
            display_summary["평균면적"].round(1)
        )
        display_summary["평균보증금"] = (
            display_summary["평균보증금"].round(1)
        )

        display_summary = display_summary.rename(columns={
            "지역": "동네",
            "거래건수": "거래 건수",
            "월세중앙값": "월세 중앙값(만원)",
            "평균월세": "평균 월세(만원)",
            "평균면적": "평균 면적(㎡)",
            "평균보증금": "평균 보증금(만원)"
        })

        st.dataframe(
            display_summary,
            use_container_width=True,
            hide_index=True
        )

        # 석장동 집중 분석
        st.divider()
        st.subheader("🎯 석장동 집중 분석")

        seokjang_df = market_df[
            market_df["지역"].astype(str).str.contains(
                "석장동",
                na=False
            )
        ].copy()

        if len(seokjang_df) > 0:

            seokjang_median = seokjang_df["월세"].median()
            seokjang_mean = seokjang_df["월세"].mean()
            seokjang_count = len(seokjang_df)

            s1, s2, s3 = st.columns(3)

            s1.metric(
                "석장동 월세 중앙값",
                f"{seokjang_median:,.1f}만원"
            )

            s2.metric(
                "석장동 평균 월세",
                f"{seokjang_mean:,.1f}만원"
            )

            s3.metric(
                "석장동 거래 건수",
                f"{seokjang_count:,}건"
            )

            # 석장동과 경주 전체 비교
            gyeongju_median = market_df["월세"].median()
            gap = seokjang_median - gyeongju_median

            st.write(
                f"석장동 월세 중앙값은 경주 전체 비교 대상의 "
                f"월세 중앙값보다 **{gap:+,.1f}만원**입니다."
            )

            if gyeongju_median > 0:
                gap_rate = gap / gyeongju_median * 100

                st.write(
                    f"경주 전체 중앙값 대비 차이: "
                    f"**{gap_rate:+.1f}%**"
                )

            seokjang_chart = pd.DataFrame({
                "지역": ["석장동", "경주 전체"],
                "월세 중앙값": [
                    seokjang_median,
                    gyeongju_median
                ]
            })

            fig_seokjang = px.bar(
                seokjang_chart,
                x="지역",
                y="월세 중앙값",
                text="월세 중앙값",
                title="석장동과 경주 전체 월세 비교",
                labels={
                    "월세 중앙값": "월세 중앙값 (만원)",
                    "지역": ""
                }
            )

            fig_seokjang.update_traces(
                texttemplate="%{text:.1f}만원",
                textposition="outside"
            )

            st.plotly_chart(
                fig_seokjang,
                use_container_width=True
            )

            st.caption(
                "석장동 비교 결과는 현재 선택한 주택 유형과 면적 필터를 "
                "적용한 거래만을 기준으로 합니다."
            )

        else:

            st.warning(
                "현재 선택한 조건에서 석장동 거래를 찾지 못했습니다. "
                "전체 주택 또는 전체 면적으로 변경해 확인해 주세요."
            )

        # 거래 건수 적은 지역 주의
        low_count = market_summary[
            market_summary["거래건수"] < 3
        ]

        if not low_count.empty:
            st.info(
                "거래 건수가 3건 미만인 동네가 있습니다. "
                "해당 지역의 월세 중앙값은 참고용으로만 활용해 주세요."
            )

        # 통계 CSV 다운로드
        st.download_button(
            "📥 동네별 월세 통계 다운로드",
            market_summary.to_csv(
                index=False,
                encoding="utf-8-sig"
            ),
            file_name="경주_동네별_월세_시세.csv",
            mime="text/csv"
        )


# ==========================================
# 10. 실제 데이터 미리보기
# ==========================================

with st.expander("📊 실제 수집 데이터 확인"):

    st.dataframe(
        df.sort_values(
            "계약년월",
            ascending=False
        ),
        use_container_width=True
    )

    csv_data = df.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        "실거래 데이터 CSV 다운로드",
        csv_data,
        file_name="gyeongju_real_rent_data.csv",
        mime="text/csv"
    )


# ==========================================
# 11. 사용자 주거 정보 입력
# ==========================================

st.divider()
st.header("🔍 나의 주거비 분석하기")

st.write(
    "현재 거주 중이거나 계약하려는 주택 정보를 입력해 주세요."
)

available_types = sorted(df["주택유형"].unique().tolist())
available_dongs = sorted(df["지역"].dropna().unique().tolist())

with st.form("housing_form"):

    col1, col2 = st.columns(2)

    with col1:

        selected_dong = st.selectbox(
            "주거 지역",
            ["전체 지역"] + available_dongs
        )

        selected_type = st.selectbox(
            "주택 유형",
            available_types
        )

        area = st.number_input(
            "전용면적 (㎡)",
            min_value=5.0,
            max_value=500.0,
            value=20.0,
            step=1.0
        )

    with col2:

        deposit = st.number_input(
            "보증금 (만원)",
            min_value=0,
            max_value=100000,
            value=500,
            step=50
        )

        rent_method = st.selectbox(
            "임대료 방식",
            ["월세", "사글세"]
        )

        if rent_method == "월세":

            entered_rent = st.number_input(
                "월세 (만원)",
                min_value=0,
                max_value=10000,
                value=40,
                step=5
            )

            contract_months = 1

        else:

            entered_rent = st.number_input(
                "계약 기간 전체 사글세 (만원)",
                min_value=0,
                max_value=100000,
                value=300,
                step=10
            )

            contract_months = st.number_input(
                "계약 기간 (개월)",
                min_value=1,
                max_value=36,
                value=6
            )

    submitted = st.form_submit_button(
        "🤖 AI 적정 임대료 분석 시작",
        type="primary",
        use_container_width=True
    )


# ==========================================
# 12. AI 분석
# ==========================================

if submitted:

    if rent_df.empty:
        st.error("AI 분석에 사용할 월세 거래 데이터가 없습니다.")
        st.stop()

    if rent_method == "사글세":
        current_rent = entered_rent / contract_months
    else:
        current_rent = entered_rent

    st.divider()
    st.header("📈 AI 분석 결과")

    st.info(
        f"입력한 주거비: 월 환산 {current_rent:,.1f}만원 "
        f"(보증금 {deposit:,}만원 별도)"
    )

    model_df = rent_df.copy()

    model_df = model_df[
        model_df["주택유형"] == selected_type
    ]

    if selected_dong != "전체 지역":

        local_df = model_df[
            model_df["지역"] == selected_dong
        ]

    else:
        local_df = model_df.copy()

    similar = local_df[
        (local_df["면적"] >= area * 0.7) &
        (local_df["면적"] <= area * 1.3)
    ].copy()

    if len(similar) < 3:

        similar = local_df[
            (local_df["면적"] >= area * 0.5) &
            (local_df["면적"] <= area * 1.5)
        ].copy()

    if len(similar) < 3:
        similar = model_df.copy()

    # ======================================
    # 13. AI 모델 학습
    # ======================================

    prediction = None

    if len(rent_df) >= 10:

        features = [
            "지역",
            "주택유형",
            "면적",
            "보증금"
        ]

        X = rent_df[features]
        y = rent_df["월세"]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "category",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["지역", "주택유형"]
                ),
                (
                    "number",
                    "passthrough",
                    ["면적", "보증금"]
                )
            ]
        )

        model = Pipeline([
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=150,
                    random_state=42,
                    min_samples_leaf=2
                )
            )
        ])

        try:

            model.fit(X, y)

            input_dong = (
                selected_dong
                if selected_dong != "전체 지역"
                else rent_df["지역"].mode().iloc[0]
            )

            input_data = pd.DataFrame([{
                "지역": input_dong,
                "주택유형": selected_type,
                "면적": area,
                "보증금": deposit
            }])

            prediction = float(
                model.predict(input_data)[0]
            )

        except Exception:
            prediction = None

    # ======================================
    # 14. 적정 월세 계산
    # ======================================

    if prediction is not None and np.isfinite(prediction):

        fair_rent = max(0, prediction)
        analysis_method = "랜덤포레스트 AI 예측"

    elif len(similar) > 0:

        fair_rent = float(similar["월세"].median())
        analysis_method = "유사 거래 중앙값"

    else:

        fair_rent = float(rent_df["월세"].median())
        analysis_method = "전체 거래 중앙값"

    if not np.isfinite(fair_rent):
        st.error("적정 임대료를 계산할 수 없습니다.")
        st.stop()

    difference = current_rent - fair_rent

    if fair_rent > 0:
        difference_rate = difference / fair_rent * 100
    else:
        difference_rate = 0

    # ======================================
    # 15. 분석 결과 출력
    # ======================================

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "현재 월 환산 임대료",
        f"{current_rent:,.1f}만원"
    )

    c2.metric(
        "AI 추정 적정 월세",
        f"{fair_rent:,.1f}만원"
    )

    c3.metric(
        "주변 시세와 차이",
        f"{difference:+,.1f}만원",
        f"{difference_rate:+.1f}%"
    )

    st.caption(
        f"분석 방법: {analysis_method} / "
        f"비교 거래: {len(similar)}건"
    )

    if difference_rate >= 10:

        st.warning(
            "현재 임대료가 AI 추정치보다 높습니다. "
            "주변 유사 거래와 비교하여 협상을 검토할 수 있습니다."
        )

    elif difference_rate <= -10:

        st.success(
            "현재 임대료가 AI 추정치보다 낮습니다. "
            "비교 거래 기준으로 상대적으로 낮은 수준입니다."
        )

    else:

        st.info(
            "현재 임대료는 AI 추정치와 큰 차이가 없습니다."
        )

    # ======================================
    # 16. 비교 차트
    # ======================================

    st.subheader("📊 임대료 비교")

    chart_data = pd.DataFrame({
        "구분": [
            "현재 임대료",
            "AI 추정 적정 월세"
        ],
        "월세": [
            current_rent,
            fair_rent
        ]
    })

    fig = px.bar(
        chart_data,
        x="구분",
        y="월세",
        text="월세",
        title="현재 임대료와 AI 추정 임대료 비교",
        labels={
            "월세": "월세 (만원)",
            "구분": ""
        }
    )

    fig.update_traces(
        texttemplate="%{text:.1f}만원",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ======================================
    # 17. 유사 거래 내역
    # ======================================

    st.subheader("🏘️ 주변 유사 주택 실거래 내역")

    if len(similar) > 0:

        display_df = similar[
            [
                "지역",
                "주택유형",
                "면적",
                "보증금",
                "월세",
                "계약년월"
            ]
        ].copy()

        display_df = display_df.sort_values(
            "계약년월",
            ascending=False
        )

        st.dataframe(
            display_df,
            use_container_width=True
        )

        st.write(
            f"유사 거래 월세 중앙값: "
            f"**{similar['월세'].median():,.1f}만원**"
        )

    else:

        st.warning(
            "조건에 맞는 유사 거래가 부족합니다."
        )

    # ======================================
    # 18. 협상 참고 자료
    # ======================================

    st.subheader("💡 임대료 협상 참고 정보")

    if difference > 0:

        st.write(
            f"""
            - 현재 월 환산 임대료는 {current_rent:,.1f}만원입니다.
            - AI 추정 월세는 {fair_rent:,.1f}만원입니다.
            - 두 금액의 차이는 월 {difference:,.1f}만원입니다.
            - 주변 유사 거래 {len(similar)}건을 함께 제시하며
              임대료 조정을 요청하는 근거로 활용할 수 있습니다.
            """
        )

        st.info(
            "협상 참고 문구: "
            "주변 유사 주택의 실거래 내역과 AI 분석 결과를 "
            "확인해 보니 현재 임대료와 차이가 있어 "
            "임대료 조정을 논의하고 싶습니다."
        )

    else:

        st.write(
            "현재 입력한 임대료는 AI 추정 적정 월세보다 "
            "높지 않은 것으로 분석되었습니다."
        )

    # ======================================
    # 19. 분석 결과 다운로드
    # ======================================

    result = pd.DataFrame([{
        "지역": selected_dong,
        "주택유형": selected_type,
        "면적": area,
        "보증금": deposit,
        "임대료방식": rent_method,
        "입력임대료": entered_rent,
        "계약개월": contract_months,
        "월환산임대료": current_rent,
        "AI추정월세": fair_rent,
        "차액": difference,
        "차이비율": difference_rate,
        "비교거래건수": len(similar),
        "분석방법": analysis_method
    }])

    st.download_button(
        "📥 AI 분석 결과 다운로드",
        result.to_csv(
            index=False,
            encoding="utf-8-sig"
        ),
        file_name="경주_AI_임대료_분석결과.csv",
        mime="text/csv"
    )


# ==========================================
# 20. 안내
# ==========================================

st.divider()

st.caption(
    """
    본 서비스는 국토교통부에 공개된 전월세 실거래 데이터를 활용한
    참고용 분석 시스템입니다.

    AI 추정치는 실제 계약 가능한 임대료나 법적 적정 임대료를
    보장하지 않습니다. 건물 상태, 관리비, 옵션, 층수, 계약 조건 등에
    따라 실제 임대료는 달라질 수 있습니다.

    사글세는 입력한 총액을 계약 개월 수로 나눈 월 환산 금액이며,
    실제 사글세 거래가 공공데이터에 모두 포함되는 것은 아닙니다.

    동네별 통계는 조회 기간과 공개된 실거래 건수에 따라 달라집니다.
    거래 건수가 적은 지역의 통계는 참고용으로 활용해 주세요.
    """
)
