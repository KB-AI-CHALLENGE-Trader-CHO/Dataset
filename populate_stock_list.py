# populate_stock_list.py
import requests
from mysql.connector import Error

# 모듈화된 설정 및 DB 연결 함수 임포트
from config.db_config import get_fmp_api_key
from services.db import get_db_connection

# --- 1. 설정 (Settings) ---
try:
    FMP_API_KEY = get_fmp_api_key()
except RuntimeError as e:
    print(f"[ERROR] {e}")
    exit()

def fetch_nasdaq_stocks():
    """FMP API에서 나스닥에 상장된 모든 주식 목록을 가져옵니다."""
    print("[INFO] Fetching NASDAQ stock list from FMP API...")
    # FMP의 전체 주식 목록 API 엔드포인트
    api_url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={FMP_API_KEY}"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        all_stocks = response.json()

        # 거래소 이름이 'NASDAQ'인 주식만 필터링
        nasdaq_stocks = [
            stock for stock in all_stocks 
            if stock.get('exchangeShortName') == 'NASDAQ' and stock.get('symbol') and stock.get('name')
        ]
        
        print(f"[SUCCESS] Fetched {len(nasdaq_stocks)} NASDAQ stocks.")
        return nasdaq_stocks

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return []
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during fetch: {e}")
        return []

def save_stocks_to_db(conn, stocks):
    """주식 목록을 stock_item 테이블에 대량으로 저장합니다."""
    if not stocks:
        print("[INFO] No stocks to save.")
        return

    print(f"[INFO] Saving {len(stocks)} stocks to the database...")
    
    try:
        cursor = conn.cursor()
        
        # executemany를 사용하여 여러 데이터를 한 번의 쿼리로 효율적으로 삽입
        query = "INSERT IGNORE INTO stock_item (symbol, name) VALUES (%s, %s)"
        
        # DB에 삽입할 데이터 리스트 생성 (튜플 형태)
        values_to_insert = [
            (stock['symbol'], stock['name']) for stock in stocks
        ]
        
        cursor.executemany(query, values_to_insert)
        conn.commit()
        
        print(f"[SUCCESS] {cursor.rowcount} new stocks were inserted into the stock_item table.")
        cursor.close()

    except Error as e:
        print(f"[ERROR] DB insert error: {e}")

# --- 2. 메인 실행 로직 ---
def main():
    """전체 스크립트를 실행합니다."""
    print("Starting NASDAQ stock list population script...")
    
    nasdaq_stocks = fetch_nasdaq_stocks()
    
    if nasdaq_stocks:
        conn = get_db_connection()
        if not conn:
            print("[ERROR] Could not connect to the database. Halting.")
            return
        
        try:
            save_stocks_to_db(conn, nasdaq_stocks)
        finally:
            if conn.is_connected():
                conn.close()
                print("[INFO] Database connection closed.")
    
    print("Script finished.")

if __name__ == "__main__":
    main()
