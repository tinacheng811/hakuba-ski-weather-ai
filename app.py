import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime

# --- 1. 核心工具函式 (必須放在主程式前面) ---

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

def get_ski_recommendation(start_date_str, end_date_str, model, scaler, df, window_size=7):
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    last_date = df['Date'].max()
    features_cols = ['tavg', 'tmax', 'tmin', 'prcp', 'snowf', 'snowdmax', 'sunhour', 'month_sin', 'month_cos']
    
    predictions_list = []
    
    # 簡化邏輯：統一使用預測 (適合部署演示)
    days_to_predict = (end_date - last_date).days
    last_data_raw = df[features_cols].tail(window_size).fillna(0).values
    current_batch = scaler.transform(last_data_raw).reshape(1, window_size, 9)

    for i in range(days_to_predict):
        pred = model.predict(current_batch, verbose=0)[0]
        curr_date = last_date + pd.Timedelta(days=i+1)
        if start_date <= curr_date <= end_date:
            dummy = np.zeros((1, 9))
            dummy[0, 0:3] = pred[0:3]
            dummy[0, 4:6] = pred[3:5]
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
            'date': day['date'], 'score': score, 'info': day,
            'stars': stars_str, 'tips': get_travel_tips(star_count, day)
        })
    best_day = max(final_scores, key=lambda x: x['score']) if final_scores else None
    return best_day, final_scores

# --- 2. Streamlit 介面 ---

st.set_page_config(page_title="白馬村滑雪預測", page_icon="❄️")
st.title("❄️ 白馬村滑雪 AI 特助")

@st.cache_resource
def load_assets():
    model = load_model('my_lstm_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    df = pd.read_csv('weather_exam.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return model, scaler, df

try:
    model, scaler, df = load_assets()
    
    st.sidebar.header("行程設定")
    start_input = st.sidebar.date_input("開始日期", datetime(2026, 2, 10))
    end_input = st.sidebar.date_input("結束日期", datetime(2026, 2, 15))

    if st.sidebar.button("執行 AI 分析"):
        best, results = get_ski_recommendation(str(start_input), str(end_input), model, scaler, df)
        
        if best:
            st.success(f"🏆 最佳推薦日：{best['date'].date()}")
            col1, col2 = st.columns(2)
            col1.metric("滑雪指數", best['stars'])
            col2.metric("預計積雪", f"{best['info']['snowdmax']:.1f} cm")
            st.info(f"💡 教練建議：{best['tips']}")
            
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
        else:
            st.warning("請選擇資料集日期之後的未來區間（例如 2026 年之後）。")

except Exception as e:
    st.error(f"載入失敗：請確保 GitHub 中有 model.h5, scaler.pkl 和 weather_exam.csv 三個檔案。")
    st.write(f"錯誤細節: {e}")