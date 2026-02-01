import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from tensorflow.keras.models import load_model

# =================================================================
# 1. 系統常數定義 (Constants)
# =================================================================
MODEL_FILE  = 'my_lstm_model.h5'
SCALER_FILE = 'scaler.pkl'
DATA_FILE   = 'weather_exam.csv'
WINDOW_SIZE = 7  # 模型訓練時使用的時序窗口長度
# 嚴格對齊訓練時的 9 個特徵欄位順序
FEATURES    = ['tavg', 'tmax', 'tmin', 'prcp', 'snowf', 'snowdmax', 'sunhour', 'month_sin', 'month_cos']

# =================================================================
# 2. 邏輯封裝 (Logic Layer)
# =================================================================

def get_star_advice(score, info):
    """根據評分轉換為星星與旅遊建議"""
    if score >= 80: stars, level = "⭐⭐⭐⭐⭐", 5
    elif score >= 50: stars, level = "⭐⭐⭐⭐", 4
    elif score >= 20: stars, level = "⭐⭐⭐", 3
    elif score > 0: stars, level = "⭐⭐", 2
    else: stars, level = "⭐", 1

    tips = {
        5: "❄️【粉雪天堂】雪質極佳，強烈建議一早出發享受鬆軟粉雪。",
        4: "🎿【滑雪首選】雪量充足，非常適合精進滑雪技術。",
        3: "🌤️【休閒舒適】適合輕鬆滑行，請注意陽光反射與防曬。",
        2: "⚠️【注意冰面】氣溫波動可能導致雪面結冰，請注意安全。",
        1: "🏠【建議休息】雪況不佳，推薦前往白馬村泡溫泉放鬆。"
    }
    # 根據氣溫動態增加小提醒
    extra = " 🥶 極寒注意" if info['tmin'] < -10 else " 💧 融雪注意" if info['tmax'] > 3 else ""
    return stars, tips.get(level) + extra

def run_ai_prediction(start_date, end_date, model, scaler, df):
    """執行 LSTM 遞迴預測演算法"""
    # 決定「種子資料」切片終點：若是未來則從 DB 最後一天開始；歷史驗證則從選定日前一天開始
    last_db_date = df['Date'].max()
    seed_end = last_db_date if start_date > last_db_date else start_date - pd.Timedelta(days=1)
    
    # 提取最後 window_size 天的資料作為模型輸入起點
    seed_df = df[df['Date'] <= seed_end].tail(WINDOW_SIZE)
    if len(seed_df) < WINDOW_SIZE: return []
    
    # 正規化特徵並調整維度為 (1, WINDOW_SIZE, 9)
    current_batch = scaler.transform(seed_df[FEATURES].fillna(0).values).reshape(1, WINDOW_SIZE, 9)
    predictions = []
    days_to_run = (end_date - seed_end).days

    for i in range(days_to_run):
        # 1. 執行模型預測 (輸出 5 個值: tavg, tmax, tmin, snowf, snowdmax)
        raw_pred = model.predict(current_batch, verbose=0)[0]
        curr_date = seed_end + pd.Timedelta(days=i+1)
        
        # 2. 數值還原：建立 9 欄 dummy 矩陣以匹配 Scaler
        dummy = np.zeros((1, 9))
        dummy[0, 0:3], dummy[0, 4:6] = raw_pred[0:3], raw_pred[3:5]
        res = scaler.inverse_transform(dummy)[0]
        
        day_info = {
            'date': curr_date, 'tavg': res[0], 'tmax': res[1], 
            'tmin': res[2], 'snowf': res[4], 'snowdmax': res[5]
        }
        
        # 3. 僅記錄使用者要求的區間
        if start_date <= curr_date <= end_date:
            # 計算滑雪評分邏輯
            score = day_info['snowdmax'] * 1.0
            if day_info['snowf'] > 2 and day_info['tmax'] < 0: score += 30
            if day_info['tmax'] > 3: score -= 20
            
            stars, tips = get_star_advice(score, day_info)
            predictions.append({'info': day_info, 'stars': stars, 'tips': tips, 'score': score})
            
        # 4. 特徵工程：計算下一天的 Sin/Cos 並更新 Batch 進行遞迴
        m_sin = np.sin(2 * np.pi * curr_date.month / 12)
        m_cos = np.cos(2 * np.pi * curr_date.month / 12)
        new_row = np.array([raw_pred[0], raw_pred[1], raw_pred[2], 0, raw_pred[3], raw_pred[4], 0.5, m_sin, m_cos])
        current_batch = np.append(current_batch[:, 1:, :], new_row.reshape(1, 1, 9), axis=1)

    return predictions

# =================================================================
# 3. 資料加載與初始化 (Initialization)
# =================================================================

@st.cache_resource
def setup_environment():
    """載入 AI 模型資產並進行資料預處理"""
    try:
        model = load_model(MODEL_FILE, compile=False)
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        df = pd.read_csv(DATA_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        # 預先計算週期性特徵，提升運行效率
        df['month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
        return model, scaler, df
    except Exception as e:
        st.error(f"🚨 系統初始化失敗: {e}")
        return None, None, None

# =================================================================
# 4. Web 介面佈局 (UI Layout)
# =================================================================

st.set_page_config(page_title="白馬村滑雪天氣AI助理", layout="centered")
st.title("❄️ 白馬村滑雪天氣AI助理")

model, scaler, df = setup_environment()

if model is not None:
    # --- 側邊欄：功能選單 ---
    st.sidebar.header("🕹️ 功能選單")
    app_mode = st.sidebar.radio("選擇功能模式", ["未來行程預測", "歷史模型驗證"])

    if app_mode == "未來行程預測":
        st.sidebar.subheader("📅 旅遊日期設定")
        d_start = st.sidebar.date_input("開始日期", datetime(2026, 2, 10))
        d_end = st.sidebar.date_input("結束日期", datetime(2026, 2, 15))
        
        # 點擊執行按鈕
        if st.sidebar.button("開始預測", type="primary"):
            results = run_ai_prediction(pd.to_datetime(d_start), pd.to_datetime(d_end), model, scaler, df)
            
            if results:
                # 找出推薦指數最高的一天
                best = max(results, key=lambda x: x['score'])
                st.success(f"🎯 最佳推薦日：{best['info']['date'].date()}")
                
                # 顯示核心數據
                c1, c2 = st.columns(2)
                c1.metric("指數", best['stars'])
                c2.metric("預估積雪", f"{best['info']['snowdmax']:.1f} cm")
                st.info(f"💡 建議：{best['tips']}")
                
                # 顯示區間詳細預測表格
                st.divider()
                st.subheader("📅 區間預測資訊")
                st.table(pd.DataFrame([{
                    '日期': r['info']['date'].date(),
                    '最高溫': f"{r['info']['tmax']:.1f}°C",
                    '最低溫': f"{r['info']['tmin']:.1f}°C",
                    '積雪(cm)': f"{r['info']['snowdmax']:.1f}",
                    '指數': r['stars']
                } for r in results]))
            else:
                st.warning("請選擇資料集日期之後的未來日期。")

    else:
        # --- 歷史模型驗證模式 ---
        st.sidebar.subheader("🔍 歷史資料核對")
        
        # 日期選擇器放在側邊欄，確保永遠可見
        target_v = st.sidebar.date_input(
            "選擇驗證日期", 
            df['Date'].max().date(),
            help="選擇資料庫已存在的日期來比對AI預測與真實觀測值"
        )
        
        # 啟動驗證按鈕也移入側邊欄
        btn_verify = st.sidebar.button("啟動驗證", type="primary")

        st.subheader(f"📊 歷史模型驗證：{target_v}")

        if btn_verify:
            # 執行單日預測
            results = run_ai_prediction(pd.to_datetime(target_v), pd.to_datetime(target_v), model, scaler, df)
            # 從 CSV 中讀取該日真實值
            actual = df[df['Date'] == pd.to_datetime(target_v)]
            
            if results and not actual.empty:
                p_info = results[0]['info']
                a_info = actual.iloc[0]
                
                # 數據對比展示
                col1, col2, col3 = st.columns(3)
                col1.metric("觀測項目", "平均氣溫", "積雪深度")
                col2.metric("真實觀測", f"{a_info['tavg']:.1f}°C", f"{a_info['snowdmax']:.1f} cm")
                col3.metric("AI預測值", f"{p_info['tavg']:.1f}°C", f"{p_info['snowdmax']:.1f} cm")
                
                # 誤差分析
                diff = abs(a_info['tavg'] - p_info['tavg'])
                if diff < 2.0:
                    st.success(f"✅ 驗證完成！溫度誤差僅 {diff:.2f}°C，表現不差。")
                else:
                    st.warning(f"🧐 誤差值為 {diff:.2f}°C。這應該是氣候異常劇烈波動的日子。")
            else:
                st.error("此日期不在資料庫中，或前置資料不足 (需至少有該日前 7 天的歷史紀錄)。")

else:
    st.error("❌ 系統啟動失敗，請檢查模型檔案是否存在。")



