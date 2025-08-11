# evaluation/engine.py
import json
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, time
from mysql.connector import Error

# services.db 모듈에서 DB 연결 함수를 가져옵니다.
from services.db import get_db_connection

# -----------------------------
# 유틸리티 함수들 (기존과 동일)
# -----------------------------
def to_dt(date_val, time_val) -> datetime:
    d = pd.to_datetime(date_val).date()
    # time_val이 이미 time 객체일 수 있으므로 str()로 변환
    t_str = str(time_val)
    # timedelta를 time으로 변환하는 경우 'days'가 포함될 수 있음
    if 'days' in t_str:
        t_str = t_str.split('days ')[1]
    t = pd.to_datetime(t_str).time()
    return datetime.combine(d, t)

def _safe_float(x):
    try:
        return float(x) if x is not None else np.nan
    except (ValueError, TypeError):
        return np.nan

def _percentile_regime(series: pd.Series, value: float) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or pd.isna(value):
        return "unknown"
    p33, p66 = np.percentile(s, [33, 66])
    if value <= p33: return "low"
    if value >= p66: return "high"
    return "mid"

def _band_event(close: float, upper: float, lower: float) -> str:
    c, u, l = _safe_float(close), _safe_float(upper), _safe_float(lower)
    if np.isnan([c, u, l]).any(): return "unknown"
    if c > u: return "break_upper"
    if c < l: return "break_lower"
    if abs(c - u) <= max(abs(u), 1.0) * 1e-4: return "touch_upper"
    if abs(c - l) <= max(abs(l), 1.0) * 1e-4: return "touch_lower"
    return "inside"

def _ma_stack(close: float, ma20: float, ma50: float, ma100: float) -> Tuple[str, str]:
    c, m20, m50, m100 = map(_safe_float, [close, ma20, ma50, ma100])
    if np.isnan([c, m20, m50, m100]).any(): return "mixed", "sideways"
    if c > m20 > m50 > m100: return "bullish", "uptrend"
    if c < m20 < m50 < m100: return "bearish", "downtrend"
    return "mixed", "sideways"

def _rsi_flag(rsi: float) -> str:
    r = _safe_float(rsi)
    if np.isnan(r): return "unknown"
    if r >= 70: return "overbought"
    if r <= 30: return "oversold"
    return "normal"

def _stoch_flag(k: float) -> str:
    kf = _safe_float(k)
    if np.isnan(kf): return "unknown"
    if kf >= 80: return "overbought"
    if kf <= 20: return "oversold"
    return "normal"

def _volume_zscore(vol_series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(vol_series, errors="coerce").dropna()
    if len(s) < 5: return None
    last, base = s.iloc[-1], s.iloc[:-1]
    mu, sd = base.mean(), base.std(ddof=1)
    if sd == 0 or pd.isna(sd): return 0.0
    return float((last - mu) / sd)

def _find_pivots(df: pd.DataFrame, col_high="high", col_low="low", k: int = 2):
    highs = pd.to_numeric(df[col_high], errors="coerce")
    lows = pd.to_numeric(df[col_low], errors="coerce")
    idx = df.index
    piv_h, piv_l = [], []
    for i in range(k, len(df) - k):
        if highs.iloc[i] == highs.iloc[i - k:i + k + 1].max():
            if highs.iloc[i] > highs.iloc[i - k:i + k + 1].drop(index=highs.index[i]).max():
                piv_h.append(idx[i])
        if lows.iloc[i] == lows.iloc[i - k:i + k + 1].min():
            if lows.iloc[i] < lows.iloc[i - k:i + k + 1].drop(index=lows.index[i]).min():
                piv_l.append(idx[i])
    return piv_h, piv_l

def _obv_divergence(df: pd.DataFrame, lookback: int = 30) -> str:
    if df.empty or "obv" not in df or df["obv"].isna().all():
        return "none"
    sub = df.tail(max(lookback, 10)).copy()
    piv_h, piv_l = _find_pivots(sub, col_high="high", col_low="low", k=2)
    if len(piv_h) >= 2:
        h1, h2 = piv_h[-2], piv_h[-1]
        p1, p2 = sub.loc[h1, "high"], sub.loc[h2, "high"]
        o1, o2 = _safe_float(sub.loc[h1, "obv"]), _safe_float(sub.loc[h2, "obv"])
        if not np.isnan([p1, p2, o1, o2]).any() and p2 > p1 and o2 < o1:
            return "bearish"
    if len(piv_l) >= 2:
        l1, l2 = piv_l[-2], piv_l[-1]
        p1, p2 = sub.loc[l1, "low"], sub.loc[l2, "low"]
        o1, o2 = _safe_float(sub.loc[l1, "obv"]), _safe_float(sub.loc[l2, "obv"])
        if not np.isnan([p1, p2, o1, o2]).any() and p2 < p1 and o2 > o1:
            return "bullish"
    return "none"
    
# -----------------------------
# DB 조회 및 분석 함수들 (기존과 거의 동일)
# -----------------------------
def prepare_data_for_trade(conn, trade_id: int):
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM trade_history WHERE id = %s", (trade_id,))
        trade_info = cursor.fetchone()
        if not trade_info:
            raise ValueError(f"Trade with id {trade_id} not found.")

        stock_item_id = trade_info["stock_item_id"]
        trade_date = pd.to_datetime(trade_info["trade_date"]).date()
        trade_dt = to_dt(trade_info["trade_date"], trade_info["trade_time"])

        query_daily = "SELECT * FROM daily_market_data WHERE stock_item_id = %s AND date <= %s ORDER BY date ASC"
        daily_df = pd.read_sql(query_daily, conn, params=[stock_item_id, trade_date])

        day_start = datetime.combine(trade_date, time(0, 0, 0))
        # intraday_market_data의 datetime 컬럼명을 확인하고 쿼리 수정
        query_intra = "SELECT * FROM intraday_market_data WHERE stock_item_id = %s AND datetime >= %s AND datetime <= %s ORDER BY datetime ASC"
        intraday_df = pd.read_sql(query_intra, conn, params=[stock_item_id, day_start, trade_dt])

        cursor.close()
        return trade_info, daily_df, intraday_df, trade_dt
    except Error as e:
        print(f"[ERROR] DB read error in prepare_data_for_trade: {e}")
        return None, None, None, None

def analyze_daily_context(daily_df: pd.DataFrame, trade_date) -> Dict:
    if daily_df is None or daily_df.empty: return {}
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
    latest_daily = daily_df[daily_df["date"] <= pd.to_datetime(trade_date).date()].iloc[-1]

    close, ma20, ma50, ma100 = map(_safe_float, [latest_daily.get("close"), latest_daily.get("ma_20d"), latest_daily.get("ma_50d"), latest_daily.get("ma_100d")])
    rsi, bb_up, bb_lo = map(_safe_float, [latest_daily.get("rsi_14d"), latest_daily.get("bollinger_upper"), latest_daily.get("bollinger_lower")])
    atr, stoch_k = map(_safe_float, [latest_daily.get("atr_14d"), latest_daily.get("stochastic_k")])
    kel_up, kel_lo = map(_safe_float, [latest_daily.get("keltner_upper"), latest_daily.get("keltner_lower")])

    ma_stack, trend = _ma_stack(close, ma20, ma50, ma100)
    obv_signal = _obv_divergence(daily_df, lookback=30)
    
    return {
        "daily_trend": trend, "daily_ma_stack": ma_stack,
        "daily_rsi": round(rsi, 2) if pd.notna(rsi) else None,
        "daily_rsi_status": _rsi_flag(rsi),
        "daily_stoch_k": round(stoch_k, 2) if pd.notna(stoch_k) else None,
        "daily_stoch_status": _stoch_flag(stoch_k),
        "daily_bb_event": _band_event(close, bb_up, bb_lo),
        "daily_atr_regime": _percentile_regime(daily_df["atr_14d"].tail(100), atr),
        "daily_obv_signal": obv_signal,
        "daily_keltner_event": _band_event(close, kel_up, kel_lo)
    }

def analyze_intraday_timing(intraday_df: pd.DataFrame, trade_dt: datetime) -> Dict:
    if intraday_df is None or intraday_df.empty: return {}
    intraday_df = intraday_df.copy()
    intraday_df["datetime"] = pd.to_datetime(intraday_df["datetime"])
    intraday_df = intraday_df[intraday_df["datetime"] <= trade_dt]
    if intraday_df.empty: return {}
    latest = intraday_df.iloc[-1]

    close, ma12, ma20 = map(_safe_float, [latest.get("close"), latest.get("ma_12_period"), latest.get("ma_20_period")])
    rsi, stoch_k = map(_safe_float, [latest.get("rsi_14_period"), latest.get("stochastic_k")])
    bb_up, bb_lo = map(_safe_float, [latest.get("bollinger_upper"), latest.get("bollinger_lower")])
    kel_up, kel_lo = map(_safe_float, [latest.get("keltner_upper"), latest.get("keltner_lower")])
    vol_z = _volume_zscore(intraday_df["volume"])

    ma_stack, trend = ("bullish", "uptrend") if close > ma12 > ma20 else (("bearish", "downtrend") if close < ma12 < ma20 else ("mixed", "sideways"))

    return {
        "intra_trend": trend, "intra_ma_stack": ma_stack,
        "intra_rsi": round(rsi, 2) if pd.notna(rsi) else None,
        "intra_rsi_status": _rsi_flag(rsi),
        "intra_stoch_k": round(stoch_k, 2) if pd.notna(stoch_k) else None,
        "intra_stoch_status": _stoch_flag(stoch_k),
        "intra_bb_event": _band_event(close, bb_up, bb_lo),
        "intra_keltner_event": _band_event(close, kel_up, kel_lo),
        "intra_volume_z": round(vol_z, 2) if vol_z is not None else None
    }

def calculate_scores(daily: Dict, intra: Dict) -> Dict:
    context = 0
    if daily:
        if daily.get("daily_ma_stack") == "bullish": context += 12
        if daily.get("daily_rsi_status") == "normal": context += 8
        if daily.get("daily_obv_signal") == "bullish": context += 6
    timing = 0
    if intra:
        if intra.get("intra_ma_stack") == "bullish": timing += 15
        if intra.get("intra_rsi_status") == "oversold": timing += 10
        if intra.get("intra_stoch_status") == "oversold": timing += 8
        if intra.get("intra_keltner_event") == "break_upper": timing += 10
        vol_z = intra.get("intra_volume_z")
        if isinstance(vol_z, (int, float)) and vol_z >= 1.5: timing += 5
    
    total = context + timing
    return {
        "score_context": context, "score_timing": timing,
        "score_rationale": 0, "score_risk": 0, "score_total": total,
        "score_confidence": 0.8 # Placeholder
    }

# -----------------------------
# [신규] 평가 결과 DB 저장 함수
# -----------------------------
def save_evaluation_to_db(conn, trade_history_id: int, eval_data: Dict):
    """분석된 평가 결과를 trade_evaluation 테이블에 저장합니다."""
    print(f"[INFO] Saving evaluation for trade_id {trade_history_id} to DB...")
    try:
        cursor = conn.cursor()
        
        # 쿼리 컬럼명과 값을 매핑
        columns = [
            'trade_history_id', 'daily_trend', 'daily_ma_stack', 'daily_rsi', 'daily_rsi_status',
            'daily_stoch_k', 'daily_stoch_status', 'daily_bb_event', 'daily_atr_regime',
            'daily_obv_signal', 'daily_keltner_event', 'intra_trend', 'intra_ma_stack',
            'intra_rsi', 'intra_rsi_status', 'intra_stoch_k', 'intra_stoch_status',
            'intra_bb_event', 'intra_keltner_event', 'intra_volume_z', 'score_context',
            'score_timing', 'score_rationale', 'score_risk', 'score_total', 'score_confidence'
        ]
        
        # eval_data 딕셔너리에서 값을 안전하게 추출
        values = [trade_history_id]
        values.extend([eval_data.get('daily_context', {}).get(col.replace('daily_', '')) for col in columns[1:11]])
        values.extend([eval_data.get('intraday_timing', {}).get(col.replace('intra_', '')) for col in columns[11:20]])
        values.extend([eval_data.get('scores', {}).get(col) for col in columns[20:]])

        # 쿼리 생성
        cols_str = ", ".join([f"`{col}`" for col in columns])
        placeholders = ", ".join(["%s"] * len(columns))
        update_str = ", ".join([f"`{col}`=VALUES(`{col}`)" for col in columns[1:]])
        
        query = f"""
            INSERT INTO trade_evaluation ({cols_str})
            VALUES ({placeholders})
            ON DUPLICATE KEY UPDATE {update_str}
        """
        
        cursor.execute(query, tuple(values))
        conn.commit()
        cursor.close()
        print(f"[SUCCESS] Evaluation for trade_id {trade_history_id} saved successfully.")

    except Error as e:
        print(f"[ERROR] DB error while saving evaluation: {e}")

# -----------------------------
# 메인 평가 함수 (수정됨)
# -----------------------------
def evaluate_and_save_trade(conn, trade_id: int):
    """거래를 평가하고, 결과를 JSON으로 반환하며, DB에 저장합니다."""
    trade_info, daily_df, intraday_df, trade_dt = prepare_data_for_trade(conn, trade_id)
    if not trade_info:
        return None
    
    daily_context = analyze_daily_context(daily_df, trade_info["trade_date"])
    intraday_timing = analyze_intraday_timing(intraday_df, trade_dt)
    scores = calculate_scores(daily_context, intraday_timing)
    
    evaluation_result = {
        "daily_context": daily_context,
        "intraday_timing": intraday_timing,
        "scores": scores,
    }

    # [신규] DB에 저장하는 함수 호출
    save_evaluation_to_db(conn, trade_id, evaluation_result)
    
    return {
        "trade_info": {
            "id": trade_info["id"],
            "stock_item_id": trade_info["stock_item_id"],
            "trade_type": trade_info["trade_type"],
            "trade_datetime": str(to_dt(trade_info["trade_date"], trade_info["trade_time"])),
            "price": _safe_float(trade_info["price"]),
            "memo": trade_info.get("memo"),
        },
        "evaluation": evaluation_result
    }

# -----------------------------
# CLI 실행부
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trade by trade_id and save to DB.")
    parser.add_argument("--trade_id", type=int, required=True)
    args = parser.parse_args()

    conn = get_db_connection()
    if conn is None:
        print(json.dumps({"error": "db_connection_failed"}, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    try:
        result = evaluate_and_save_trade(conn, args.trade_id)
        if result is None:
            print(json.dumps({"error": "evaluation_failed"}, ensure_ascii=False, indent=2))
        else:
            print("\n--- Evaluation Result JSON ---")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("----------------------------")
    finally:
        if conn.is_connected():
            conn.close()
