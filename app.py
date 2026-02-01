import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime

# --- 1. 核心工具函式 ---

def get_star_rating(score):
    if score >= 80: star_count = 5
    elif score >= 50: star_count = 4
    elif score >= 20: star_count = 3
    elif score > 0: star_count = 2
    else: star_count = 1
    stars = "⭐" * star_count + "☆" * (5 - star_count)
    comments = ["極不推薦", "勉強可以", "普通", "良好", "極佳！"]
    return stars, comments[star_count - 1], star_count

def get_travel_tips(star_count, info):
    tips = {
        5: "❄️【極佳：粉雪天堂】今天雪質完美！建議一早出發搶頭香，享受鬆軟粉雪。",
        4: "🎿【優良：滑雪首選】雪量充足，是非常適合練習技術的一天。",
        3: "🌤️【普通：休閒滑雪】建議以休閒滑行為主，注意防曬。",
        2: "⚠️【注意：雪質偏硬】氣溫波動，雪面可能結冰，新手請務必配戴護具。",
        1: "🏠【建議：室內活動】今日雪況不佳。建議去泡溫泉或在咖啡廳放鬆。"
    }
    temp_tip = ""
    if info['tmin'] < -10: temp_tip = " 🥶 提醒：氣溫極低，注意防凍傷！"
    elif info['tmax'] > 3: temp_tip = " 💧 提醒：氣溫回升，雪質較黏。"
    return tips.get(star_count, "資訊不足") + temp_tip

# 修改後的預測函式：支援指定「啟動日期」
def get_ski_recommendation(start_date_str, end_date_str, model, scaler, df, window_size=7):
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    
    # 決定「種子資料」的終點：如果是未來則用 df 最後一天；如果是驗證則用 start_date 的前一天
    last_date_in_df = df['Date'].max()
    is_future = start_date > last_date_in_df
    
    if is_future:
        seed_end_date = last_date_in_df
    else:
        seed_end_date = start_date - pd.Timedelta(days=1)
    
    features_cols = ['tavg', 'tmax', 'tmin', 'prcp', 'snowf', 'snowdmax', 'sunhour', 'month_sin', 'month_cos']
    
    # 抓取種子資料
    seed_data = df[df['Date'] <= seed_end_date].tail(window_size)
    if len(seed_data) < window_size:
        return None, []
        
    current_batch = scaler.transform(seed_data[features_cols].fillna(0).values).reshape(1, window_size, 9)
    
    predictions_list = []
    days_to_predict = (end_date - seed_end_date).days

    for i in range(days_to_predict):
        pred = model.predict(current_batch, verbose=0)[0]
        curr_date = seed_end_date + pd.Timedelta(days=i+1)
        
        if start_date <= curr_date <= end_date:
            dummy = np.zeros((1, 9))
            dummy[0, 0:3], dummy[0, 4:6] = pred[0:3], pred[3:5]
            res = scaler.inverse_transform(dummy)[0]
            predictions_list.append({
                'date': curr_date, 'tavg': res[0], 'tmax': res[1], 'tmin': res[2],
                'snowf': res[4], 'snowdmax': res[5]
            })
            
        m_sin, m_cos = np.sin(2 * np.pi * curr_date.month / 12), np.cos(2 * np.pi * curr_date.month / 12)
        new_entry = np.array([pred[0], pred[1], pred[2], 0, pred[3], pred[4], 0.5, m_sin, m_cos]).reshape(1, 1, 9)
        current_batch = np.append(current_batch[:, 1:, :], new_entry, axis=1)

    final_scores = []
    for day in predictions_list:
        score = day['snowdmax'] * 1.0
        if day['snowf'] > 2 and day['tmax'] < 0: score += 30
        if day['tmax'] > 3: score -= 20
        stars_str, status, star_count = get_star_rating(score)
        final_scores.append({
            'date': day['date'], 'score': score, 'info': day, 'stars': stars_str, 'tips': get_travel_tips(star_count, day)
        })
    return (max(final_scores, key=lambda x: x['score']) if final_scores else None), final_scores

# --- 2. Streamlit 介面 ---

st.set_page_config(page_title="白馬村滑雪天氣預測AI", page_icon="❄️")
st.title("❄️ 白馬村滑雪天氣預測AI")

@st.cache_resource
def load_assets():
    model = load_model('my_lstm_model.h5', compile=False) 
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    df = pd.read_csv('weather_exam.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
    return model, scaler, df

try:
    model, scaler, df = load_assets()
    st.sidebar.header("功能設定")
    mode = st.sidebar.radio("選擇模式", ["未來行程規劃", "歷史預測驗證"])

    if mode == "未來行程規劃":
        st.sidebar.header("行程設定")
        start_input = st.sidebar.date_input("開始日期", datetime(2026, 2, 10))
        end_input = st.sidebar.date_input("結束日期", datetime(2026, 2, 15))
        if st.sidebar.button("執行AI分析"):
            best, results = get_ski_recommendation(str(start_input), str(end_input), model, scaler, df)
            if best:
                st.success(f"🏆 最佳推薦日：{best['date'].date()}")
                c1, c2 = st.columns(2)
                c1.metric("滑雪指數", best['stars'])
                c2.metric("預計積雪", f"{best['info']['snowdmax']:.1f} cm")
                st.info(f"💡 教練建議：{best['tips']}")
                # 表格顯示---------------------------------------------------------------
                st.divider()
                st.subheader("📅 區間詳細預報")
                display_df = pd.DataFrame([{
                    '日期': r['date'].date(),
                    '最高溫': f"{r['info']['tmax']:.1f}°C",
                    '最低溫': f"{r['info']['tmin']:.1f}°C",
                    '積雪(cm)': round(r['info']['snowdmax'], 1),
                    '指數': r['stars']
                } for r in results])
                st.table(display_df)
                #-------------------------------------------------------------
            else:
                st.warning("請選擇資料集最後一天之後的日期。")

    else:
        st.sidebar.warning("⚠️ 驗證模式：對比模型預測與真實觀測值。")
        # 修正：確保 verify_date 永遠有值
        if st.sidebar.button("隨機挑選一個歷史日期"):
            st.session_state['check_date'] = df['Date'].sample(1).iloc[0].date()
        
        default_date = st.session_state.get('check_date', df['Date'].max().date())
        verify_date = st.sidebar.date_input("選擇驗證日期", default_date)

        st.subheader(f"🔍 歷史資料驗證：{verify_date}")
        real_data = df[df['Date'] == pd.to_datetime(verify_date)]

        if not real_data.empty:
            if st.button("開始核對"):
                _, result = get_ski_recommendation(str(verify_date), str(verify_date), model, scaler, df)
                if result:
                    pred, actual = result[0]['info'], real_data.iloc[0]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("項目", "平均溫", "積雪深度")
                    col2.metric("真實觀測", f"{actual['tavg']:.1f}°C", f"{actual['snowdmax']:.1f} cm")
                    col3.metric("AI 預測", f"{pred['tavg']:.1f}°C", f"{pred['snowdmax']:.1f} cm")
                    diff = abs(actual['tavg'] - pred['tavg'])
                    st.info(f"💡 溫度誤差：{diff:.2f}°C")
                    if diff < 2.0:
                        st.success("✅ 驗證完成！系統預測相當準確。")
                    else:
                        st.warning("🧐 誤差較大。這通常是因為當天有突發氣候變化（如強烈寒流或暖流）。")
        else:
            st.error("此日期不在資料庫中。")

except Exception as e:
    st.error(f"載入失敗，錯誤細節: {e}")

