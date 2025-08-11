import json
import requests
import pandas as pd
import numpy as np
from mysql.connector import Error

from config.db_config import get_fmp_api_key
from services.db import get_db_connection

# --- 1. 설정 (Settings) ---
try:
    FMP_API_KEY = get_fmp_api_key()
except RuntimeError as e:
    print(f"[ERROR] {e}")
    exit()

def load_tickers_from_json(file_path='config/tickers_to_process.json'):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            tickers = data.get('tickers')
            if not tickers or not isinstance(tickers, list):
                print(f"[ERROR] Ticker list is missing or not a list in {file_path}")
                return []
            print(f"[INFO] Loaded {len(tickers)} tickers to process: {tickers}")
            return tickers
    except FileNotFoundError:
        print(f"[ERROR] Ticker file not found at: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"[ERROR] Could not decode JSON from: {file_path}")
        return []

TICKERS_TO_PROCESS = load_tickers_from_json()

# --- 2. 기술적 지표 계산 ---
def calculate_technicals(df):
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # 기본 지표
    df['ma_20d'] = df['close'].rolling(window=20).mean()
    df['ma_50d'] = df['close'].rolling(window=50).mean()
    df['ma_100d'] = df['close'].rolling(window=100).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))
    
    df['bollinger_mid'] = df['close'].rolling(window=20).mean()
    std_20d = df['close'].rolling(window=20).std()
    df['bollinger_upper'] = df['bollinger_mid'] + (std_20d * 2)
    df['bollinger_lower'] = df['bollinger_mid'] - (std_20d * 2)
    
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14d'] = tr.rolling(window=14).mean()

    # [신규] 암호화폐 전략 기반 지표 추가
    # 스토캐스틱
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stochastic_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stochastic_d'] = df['stochastic_k'].rolling(window=3).mean()

    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # 켈트너 채널
    df['keltner_mid'] = df['close'].ewm(span=20, adjust=False).mean()
    df['keltner_upper'] = df['keltner_mid'] + (df['atr_14d'] * 2)
    df['keltner_lower'] = df['keltner_mid'] - (df['atr_14d'] * 2)
    
    return df

# --- 3. 데이터 수집 및 저장 함수 ---
def save_stock_profile(conn, ticker):
    api_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_API_KEY}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        profile_data = response.json()
        if not profile_data: return

        profile = profile_data[0]
        cursor = conn.cursor()
        query = "INSERT IGNORE INTO stock_item (symbol, name) VALUES (%s, %s)"
        values = (profile.get('symbol'), profile.get('companyName'))
        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        print(f"[SUCCESS] [stock_item] Profile for '{ticker}' synchronized.")
    except Exception as e:
        print(f"[ERROR] [stock_item] Error for '{ticker}': {e}")

def get_stock_id(conn, ticker):
    try:
        cursor = conn.cursor()
        query = "SELECT id FROM stock_item WHERE symbol = %s"
        cursor.execute(query, (ticker,))
        result = cursor.fetchone()
        cursor.close()
        if result: return result[0]
        return None
    except Error as e:
        print(f"[ERROR] Could not retrieve stock_id for '{ticker}': {e}")
        return None

def process_daily_data(conn, stock_id, ticker, backfill=False):
    api_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={FMP_API_KEY}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        if not data or 'historical' not in data: return

        df = pd.DataFrame(data['historical'])
        df['date'] = pd.to_datetime(df['date']).dt.date
        df_with_technicals = calculate_technicals(df)
        
        df_to_save = df_with_technicals.dropna() if backfill else df_with_technicals.dropna().tail(1)
        
        if df_to_save.empty: return
        
        if backfill:
            print(f"[INFO] [daily_market_data] Preparing to backfill {len(df_to_save)} records for '{ticker}'.")

        cursor = conn.cursor()
        for _, row in df_to_save.iterrows():
            query = """
                INSERT INTO daily_market_data (
                    stock_item_id, date, open, high, low, close, volume, 
                    ma_20d, ma_50d, ma_100d, rsi_14d, 
                    bollinger_mid, bollinger_upper, bollinger_lower, atr_14d,
                    stochastic_k, stochastic_d, obv, 
                    keltner_mid, keltner_upper, keltner_lower
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                    %s, %s, %s, %s, %s, %s
                ) ON DUPLICATE KEY UPDATE
                    open=VALUES(open), high=VALUES(high), low=VALUES(low), close=VALUES(close), volume=VALUES(volume),
                    ma_20d=VALUES(ma_20d), ma_50d=VALUES(ma_50d), ma_100d=VALUES(ma_100d), rsi_14d=VALUES(rsi_14d),
                    bollinger_mid=VALUES(bollinger_mid), bollinger_upper=VALUES(bollinger_upper), bollinger_lower=VALUES(bollinger_lower), atr_14d=VALUES(atr_14d),
                    stochastic_k=VALUES(stochastic_k), stochastic_d=VALUES(stochastic_d), obv=VALUES(obv),
                    keltner_mid=VALUES(keltner_mid), keltner_upper=VALUES(keltner_upper), keltner_lower=VALUES(keltner_lower)
            """
            values = (
                stock_id, row['date'], row['open'], row['high'], row['low'], row['close'], row['volume'],
                row['ma_20d'], row['ma_50d'], row['ma_100d'], row['rsi_14d'],
                row['bollinger_mid'], row['bollinger_upper'], row['bollinger_lower'], row['atr_14d'],
                row['stochastic_k'], row['stochastic_d'], row['obv'],
                row['keltner_mid'], row['keltner_upper'], row['keltner_lower']
            )
            cursor.execute(query, values)
        conn.commit()
        cursor.close()
        
        if backfill:
            print(f"[SUCCESS] [daily_market_data] Backfilled historical data for '{ticker}'.")
        else:
            print(f"[SUCCESS] [daily_market_data] Data for '{ticker}' on {df_to_save['date'].iloc[0]} saved.")
    except Exception as e:
        print(f"[ERROR] [daily_market_data] Error for '{ticker}': {e}")

def save_annual_data(conn, stock_id, ticker):
    # (이 함수는 유료 플랜 필요, 코드는 기존과 동일)
    pass

# --- 4. 메인 실행 로직 ---
def main():
    print("Data Pipeline Starting...")
    if not TICKERS_TO_PROCESS: return

    conn = get_db_connection()
    if not conn: return

    for ticker in TICKERS_TO_PROCESS:
        print(f"\n--- Processing '{ticker}' ---")
        save_stock_profile(conn, ticker)
        
        stock_id = get_stock_id(conn, ticker)
        if not stock_id:
            print(f"[ERROR] Could not find stock_id for '{ticker}'. Skipping.")
            continue

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM daily_market_data WHERE stock_item_id = %s", (stock_id,))
        count = cursor.fetchone()[0]
        cursor.close()
        
        process_daily_data(conn, stock_id, ticker, backfill=(count == 0))
        # save_annual_data(conn, stock_id, ticker)
        print(f"--- Finished '{ticker}' ---")

    if conn.is_connected():
        conn.close()
    print("\nAll tasks complete. Database connection closed.")

if __name__ == "__main__":
    main()
