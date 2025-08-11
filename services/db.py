# services/db.py
import mysql.connector
from mysql.connector import Error
from config.db_config import get_db_config

def get_db_connection():
    """환경 변수 기반 설정을 사용하여 MySQL 연결을 생성하고 반환합니다."""
    cfg = get_db_config()
    try:
        conn = mysql.connector.connect(
            host=cfg.host,
            port=cfg.port,
            user=cfg.user,
            password=cfg.password,
            database=cfg.database,
        )
        if conn.is_connected():
            print("[INFO] Database connection successful.")
        return conn
    except Error as e:
        print(f"[ERROR] DB Connection Error: {e}")
        return None
