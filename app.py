import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle # 用來載入你的 scaler

# 1. 網頁標題與選單
st.set_page_config(page_title="白馬村滑雪天氣AI助理", page_icon="❄️")
st.title("❄️ 白馬村滑雪天氣AI助理")
st.write("透過LSTM深度學習模型預測最佳滑雪時機")

# 2. 側邊欄：使用者輸入
st.sidebar.header("請選擇您的旅遊期間:")
start_date = st.sidebar.date_input("開始日期:")
end_date = st.sidebar.date_input("結束日期:")

# 3. 載入模型與工具 (這部分建議先在 Colab 儲存好)
# @st.cache_resource 確保模型只會載入一次，節省手機開啟時間
@st.cache_resource
def load_ai_assets():
    model = load_model('my_lstm_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# 4. 核心邏輯 (將你之前的 get_ski_recommendation 放進來)
def get_ski_recommendation(start_date_str, end_date_str, model, scaler, df, window_size=7):
    # 1. 時間格式轉換
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    last_date = df['Date'].max()
    
    # 2. 定義特徵欄位 (必須是訓練時的 9 個)
    features_cols = ['tavg', 'tmax', 'tmin', 'prcp', 'snowf', 'snowdmax', 'sunhour', 'month_sin', 'month_cos']
    
    # 3. 準備儲存結果的清單
    predictions_list = []

    # 4. 判斷區間落在「歷史」還是「未來」
    # 如果區間完全在歷史內，直接從 df 抓
    if end_date <= last_date:
        relevant_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        for _, row in relevant_df.iterrows():
            predictions_list.append({
                'date': row['Date'],
                'tavg': row['tavg'], 'tmax': row['tmax'], 'tmin': row['tmin'],
                'snowf': row['snowf'], 'snowdmax': row['snowf'] # 歷史資料可能叫不同名
            })
    else:
        # 如果包含未來，執行 LSTM 遞迴預測
        days_to_predict = (end_date - last_date).days
        
        # 準備最後 window_size 天的資料作為啟動種子
        last_data_raw = df[features_cols].tail(window_size).fillna(0).values
        current_batch = scaler.transform(last_data_raw).reshape(1, window_size, 9)

        for i in range(days_to_predict):
            # 模型輸出 5 個值: [tavg, tmax, tmin, snowf, snowdmax]
            pred = model.predict(current_batch, verbose=0)[0]
            curr_date = last_date + pd.Timedelta(days=i+1)

            # 如果這天在遊客要求的區間內，則還原並記錄
            if start_date <= curr_date <= end_date:
                dummy = np.zeros((1, 9))
                dummy[0, 0:3] = pred[0:3] # tavg, tmax, tmin
                dummy[0, 4:6] = pred[3:5] # snowf, snowdmax
                res = scaler.inverse_transform(dummy)[0]
                
                predictions_list.append({
                    'date': curr_date,
                    'tavg': res[0], 'tmax': res[1], 'tmin': res[2],
                    'snowf': res[4], 'snowdmax': res[5]
                })

            # 更新下一天的輸入 (維持 9 特徵)
            m_sin = np.sin(2 * np.pi * curr_date.month / 12)
            m_cos = np.cos(2 * np.pi * curr_date.month / 12)
            # 構造 [tavg, tmax, tmin, prcp(0), snowf, snowdmax, sunhour(0.5), m_sin, m_cos]
            new_entry = np.array([pred[0], pred[1], pred[2], 0, pred[3], pred[4], 0.5, m_sin, m_cos]).reshape(1, 1, 9)
            current_batch = np.append(current_batch[:, 1:, :], new_entry, axis=1)

    # 5. 評分與星星轉化邏輯 (延用之前的邏輯)
# 5. 評分與星星轉化邏輯
    final_scores = []
    for day in predictions_list:
        score = 0
        score += day['snowdmax'] * 1.0  # 積雪深度分
        if day['snowf'] > 2 and day['tmax'] < 0: 
            score += 30 # 粉雪分
        if day['tmax'] > 3: 
            score -= 20 # 融雪扣分

        # 這裡要確保所有的縮進 (Indentation) 都對齊
        star_count = 1
        if score >= 80: star_count = 5
        elif score >= 50: star_count = 4
        elif score >= 20: star_count = 3
        elif score > 0: star_count = 2

        # --- 關鍵修正區塊 ---
        final_scores.append({
            'date': day['date'],
            'score': score,
            'info': day,
            'stars': "⭐" * star_count + "☆" * (5 - star_count),
            'tips': get_travel_tips(star_count, day) # 注意這行要跟上面的對齊
        })
    # ------------------

    if not final_scores:
        return None, []

    best_day = max(final_scores, key=lambda x: x['score'])
    return best_day, final_scores



# ------------------------------------------------------
if st.sidebar.button("開始AI預測"):
    with st.spinner('AI正在計算雪況中...'):
        # 這裡呼叫你之前寫好的函式
        best_day, all_results = get_ski_recommendation(str(start_date), str(end_date), model, scaler, df)
        
        # 顯示最佳日期 (星星與小撇步)
        st.success(f"🏆 最佳推薦日：{best_day['date'].date()}")
        st.metric("推薦指數", best_day['stars'])
        st.info(f"💡 教練建議：{best_day['tips']}")
        
        # 顯示詳細數據表
        st.subheader("📊 詳細預報數據")
        res_df = pd.DataFrame([r['info'] for r in all_results])
        st.dataframe(res_df)