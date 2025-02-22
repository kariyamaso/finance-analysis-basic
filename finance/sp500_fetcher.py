import requests
import pandas as pd

def get_sp500_list() -> pd.DataFrame:
    """
    Wikipedia から S&P500 の銘柄一覧を取得してDataFrameで返す。
    例: カラム["Symbol", "Security", "Sector", "Industry", ...]
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    tables = pd.read_html(response.text)

    df_sp500 = tables[0]
    df_sp500.rename(columns={
        "GICS Sector": "Sector",
        "GICS Sub-Industry": "Industry",
    }, inplace=True)
    return df_sp500

def get_sp500_tickers() -> list:
    """
    Symbol列をリストで返す。
    """
    df_sp500 = get_sp500_list()
    return df_sp500["Symbol"].tolist()
