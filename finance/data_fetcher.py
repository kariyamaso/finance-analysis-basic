import os
import time
import pandas as pd
import yfinance as yf

def fetch_price_data(ticker: str, start: str, end: str, cache_dir="price_cache") -> pd.DataFrame:
    """
    指定されたティッカー(ticker)・日付範囲(start～end)の株価を取得。
    ただしローカル(cache_dir)に既にキャッシュがあれば再ダウンロードを回避する。
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = f"{ticker}_{start}_{end}.csv"
    # Windowsなどで使えない文字":"対策
    cache_filepath = os.path.join(cache_dir, cache_filename.replace(":", "-"))

    # キャッシュファイルの存在チェック
    if os.path.exists(cache_filepath):
        df_cached = pd.read_csv(cache_filepath, index_col=0, parse_dates=True)
        if not df_cached.empty:
            return df_cached

    # キャッシュが無い or 空なら yfinance でダウンロード
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return pd.DataFrame()

    # マルチインデックスがある場合、フラット化
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(tuple(map(str, col))).rstrip("_")
            for col in df.columns.to_flat_index()
        ]

    rename_dict = {}
    for col in df.columns:
        
        if col.endswith(f"_{ticker}"):
            base_col = col.replace(f"_{ticker}", "")  
            rename_dict[col] = base_col

    df.rename(columns=rename_dict, inplace=True)

    # CSVに保存
    df.to_csv(cache_filepath)
    return df


def fetch_financial_data(ticker: str, cache_dir="finance_cache"):
    """
    yfinanceから財務諸表(income_stmt, balance_sheet, cashflow_stmt)を取得。
    ただしローカルにキャッシュがあれば再利用する。
    """
    import os
    import pandas as pd
    import yfinance as yf

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}.pkl")

    if os.path.exists(cache_file):
        try:
            data = pd.read_pickle(cache_file)
            if data:
                return data["income"], data["balance"], data["cashflow"]
        except:
            pass  # 読み込み失敗時は取得し直す

    # 取得
    ticker_obj = yf.Ticker(ticker)
    income_stmt = ticker_obj.financials
    balance_sheet = ticker_obj.balance_sheet
    cashflow_stmt = ticker_obj.cashflow

    # 転置
    income_stmt = income_stmt.T if not income_stmt.empty else pd.DataFrame()
    balance_sheet = balance_sheet.T if not balance_sheet.empty else pd.DataFrame()
    cashflow_stmt = cashflow_stmt.T if not cashflow_stmt.empty else pd.DataFrame()

    # ピックルにまとめて保存
    data = {
        "income":   income_stmt,
        "balance":  balance_sheet,
        "cashflow": cashflow_stmt
    }
    pd.to_pickle(data, cache_file)

    return income_stmt, balance_sheet, cashflow_stmt

def fetch_company_info(ticker: str, cache_dir="info_cache"):
    """
    yfinanceの Ticker.info から 企業名, セクター, 業種 を取得。
    キャッシュがあれば再利用し、無ければ取得→保存する。
    """
    import os
    import json

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_info.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                info_data = json.load(f)
            return (
                info_data.get("company_name", ""),
                info_data.get("sector", ""),
                info_data.get("industry", "")
            )
        except:
            pass

    # キャッシュが無い場合は yfinance で取得
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info

    company_name = info.get("longName") or info.get("shortName") or ""
    sector = info.get("sector", "")
    industry = info.get("industry", "")

    # JSONに保存
    info_data = {
        "company_name": company_name,
        "sector": sector,
        "industry": industry
    }
    with open(cache_file, "w") as f:
        json.dump(info_data, f)

    return (company_name, sector, industry)

