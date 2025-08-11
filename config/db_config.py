# config/db_config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# .env 로드 (프로세스 시작 시 1회)
load_dotenv()

@dataclass(frozen=True)
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

def get_db_config() -> DBConfig:
    """환경 변수에서 DB 설정을 읽어와 객체로 반환합니다."""
    return DBConfig(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )

def get_fmp_api_key() -> str:
    """환경 변수에서 FMP API 키를 읽어옵니다."""
    key = os.getenv("FMP_API_KEY")
    if not key:
        raise RuntimeError("FMP_API_KEY is not set in .env file.")
    return key
