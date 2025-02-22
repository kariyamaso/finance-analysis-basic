import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import plotly.express as px
import streamlit as st
from finance.sp500_fetcher import get_sp500_list
from finance.data_fetcher import fetch_financial_data  
from sklearn.metrics import silhouette_score

def _fill_and_pick_latest(df: pd.DataFrame) -> pd.DataFrame:
    """
    df を時系列ソート → 前方埋め (ffill) → 数値列は残りの欠損を列平均で埋め → 最後(最新)の行のみ返す
    dfが空の場合はそのまま返す
    """
    if df.empty:
        return df
    
    # 1) インデックスをソート（古い順→新しい順）
    df_sorted = df.sort_index(ascending=True)
    
    # 2) 前方埋め (ffill) で、古いデータを最新まで引き継ぐ
    df_sorted = df_sorted.ffill()
    
    # 3) まだ NaN が残っている場合、数値列は平均値で埋める
    numeric_cols = df_sorted.select_dtypes(include="number").columns
    for col in numeric_cols:
        # 列の平均値を計算
        col_mean = df_sorted[col].mean(skipna=True)
        # NaN を平均値で置き換え
        df_sorted[col] = df_sorted[col].fillna(col_mean)
    
    # 4) 最後(最新)の行だけ取得
    latest_idx = df_sorted.index[-1]
    return df_sorted.loc[[latest_idx]]

def integrate_financial_data(
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cashflow_stmt: pd.DataFrame
) -> pd.DataFrame:
    """
    P/L, B/S, C/F それぞれ、NaNを遡って値を探して(前方埋め)→残りは平均埋め→
    最新データ(行)を抜き出して指標を計算→一つに統合
    """
    # 1) ffill + 平均埋め + 最新行抜き出し
    inc_latest = _fill_and_pick_latest(income_stmt)
    bs_latest  = _fill_and_pick_latest(balance_sheet)
    cf_latest  = _fill_and_pick_latest(cashflow_stmt)

    # 2) 必要カラムを抜き出して指標計算
    # ---------------------------------------------------
    bs_cols = [
        "Total Assets",
        "Current Assets",
        "Non Current Assets",
        "Cash And Cash Equivalents",
        "Total Liabilities Net Minority Interest",
        "Current Liabilities",
        "Long Term Debt",
        "Stockholders Equity",
        "Retained Earnings",
        "Working Capital",
        "Ordinary Shares Number",
    ]

    pl_cols = [
        "Total Revenue",
        "Cost Of Revenue",
        "Gross Profit",
        "Operating Expense",
        "Operating Income",
        "EBIT",
        "EBITDA",
        "Pretax Income",
        "Net Income",
        "Tax Provision",
        "Selling General And Administration",
        "Research And Development",
        "Basic EPS",
        "Diluted EPS",
    ]

    cf_cols = [
        "Operating Cash Flow",
        "Investing Cash Flow",
        "Financing Cash Flow",
        "Free Cash Flow",
        "Capital Expenditure",
        "Beginning Cash Position",
        "End Cash Position",
        "Changes In Cash",
    ]

    def pick(df: pd.DataFrame, cols: list):
        if df.empty:
            return pd.DataFrame()
        
        row = df.iloc[0]
        out_dict = {}
        for c in cols:
            out_dict[c] = row[c] if c in row else np.nan
        return pd.DataFrame([out_dict], index=df.index)  

    bs_part = pick(bs_latest, bs_cols)
    pl_part = pick(inc_latest, pl_cols)
    cf_part = pick(cf_latest, cf_cols)

    # 指標計算
    if not bs_part.empty:
        bs_part["Equity Ratio (%)"] = (
            bs_part["Stockholders Equity"] / bs_part["Total Assets"] * 100
        )
        bs_part["Debt Ratio (%)"] = (
            bs_part["Total Liabilities Net Minority Interest"] / bs_part["Total Assets"] * 100
        )
        bs_part["DE Ratio"] = (
            bs_part["Total Liabilities Net Minority Interest"] / bs_part["Stockholders Equity"]
        )
        bs_part["Current Ratio"] = (
            bs_part["Current Assets"] / bs_part["Current Liabilities"]
        )
        bs_part["BPS"] = (
            bs_part["Stockholders Equity"] / bs_part["Ordinary Shares Number"]
        )

    if not pl_part.empty:
        pl_part["Gross Margin (%)"] = (
            pl_part["Gross Profit"] / pl_part["Total Revenue"] * 100
            if ("Gross Profit" in pl_part.columns and "Total Revenue" in pl_part.columns and not pl_part["Total Revenue"].isna().all())
            else np.nan
        )
        pl_part["Operating Margin (%)"] = (
            pl_part["Operating Income"] / pl_part["Total Revenue"] * 100
            if ("Operating Income" in pl_part.columns and "Total Revenue" in pl_part.columns and not pl_part["Total Revenue"].isna().all())
            else np.nan
        )
        pl_part["EBIT Margin (%)"] = (
            pl_part["EBIT"] / pl_part["Total Revenue"] * 100
            if ("EBIT" in pl_part.columns and "Total Revenue" in pl_part.columns and not pl_part["Total Revenue"].isna().all())
            else np.nan
        )
        pl_part["Net Margin (%)"] = (
            pl_part["Net Income"] / pl_part["Total Revenue"] * 100
            if ("Net Income" in pl_part.columns and "Total Revenue" in pl_part.columns and not pl_part["Total Revenue"].isna().all())
            else np.nan
        )
        pl_part["SG&A Ratio (%)"] = (
            pl_part["Selling General And Administration"] / pl_part["Total Revenue"] * 100
            if ("Selling General And Administration" in pl_part.columns and "Total Revenue" in pl_part.columns and not pl_part["Total Revenue"].isna().all())
            else np.nan
        )
        pl_part["Effective Tax Rate (%)"] = (
            pl_part["Tax Provision"] / pl_part["Pretax Income"] * 100
            if ("Tax Provision" in pl_part.columns 
                and "Pretax Income" in pl_part.columns
                and not pl_part["Pretax Income"].replace({0: np.nan}).isna().all())
            else np.nan
        )

    if not cf_part.empty:
        if "Free Cash Flow" not in cf_part.columns or cf_part["Free Cash Flow"].isna().all():
            if "Operating Cash Flow" in cf_part.columns and "Capital Expenditure" in cf_part.columns:
                cf_part["Free Cash Flow"] = (
                    cf_part["Operating Cash Flow"] - cf_part["Capital Expenditure"]
                )
            else:
                cf_part["Free Cash Flow"] = np.nan

        if "Total Revenue" in pl_part.columns and not pl_part["Total Revenue"].isna().all():
            cf_part["FCF Margin (%)"] = (
                cf_part["Free Cash Flow"] / pl_part["Total Revenue"] * 100
            )
            cf_part["Operating CF Margin (%)"] = (
                cf_part["Operating Cash Flow"] / pl_part["Total Revenue"] * 100
            )
        else:
            cf_part["FCF Margin (%)"] = np.nan
            cf_part["Operating CF Margin (%)"] = np.nan

    # 統合
    bs_part = bs_part.dropna(axis=1, how="all")
    pl_part = pl_part.dropna(axis=1, how="all")
    cf_part = cf_part.dropna(axis=1, how="all")

    bs_pl = pd.merge(bs_part, pl_part, how="outer", left_index=True, right_index=True)
    all_merged = pd.merge(bs_pl, cf_part, how="outer", left_index=True, right_index=True)

    if all_merged.empty:
        return all_merged

    # 列がすべてNaNなら削除
    all_merged = all_merged.dropna(axis=1, how="all")

    return all_merged

def ml_analysis(tickers: list):
    from finance.data_fetcher import fetch_financial_data
    from finance.analyzer import integrate_financial_data
    import streamlit as st
    import pandas as np
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    import plotly.express as px

    data_list = []
    valid_tickers = []

    for ticker in tickers:
        inc, bs, cf = fetch_financial_data(ticker)
        df_merged = integrate_financial_data(inc, bs, cf)  

        
        if not df_merged.empty:
            df_merged = df_merged.reset_index(drop=True)
            df_merged.index = [ticker]
            data_list.append(df_merged)
            valid_tickers.append(ticker)

    if not data_list:
        st.warning("有効な財務データを取得できた企業がありません。")
        return

    # 複数企業を縦方向に結合
    df_all = pd.concat(data_list)
    df_numeric = df_all.select_dtypes(include="number")
    df_numeric = df_numeric.fillna(0)

    # 行数・列数をチェック
    n_samples, n_features = df_numeric.shape
    if n_samples < 2 or n_features < 2:
        st.warning(
            f"サンプル数={n_samples}, 特徴量数={n_features} なので PCA(2次元)を実行できません。"
        )
        st.dataframe(df_numeric)
        return

    # ここから PCA / KMeans / IsolationForest
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_numeric)
    df_all["pca_x"] = pca_result[:, 0]
    df_all["pca_y"] = pca_result[:, 1]

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(df_numeric)
    df_all["cluster"] = clusters.astype(str)

    iso = IsolationForest(contamination=0.1, random_state=42)
    outliers = iso.fit_predict(df_numeric)
    df_all["anomaly"] = outliers

    fig = px.scatter(
        df_all,
        x="pca_x",
        y="pca_y",
        color="cluster",
        symbol="anomaly",
        hover_name=df_all.index,
        title="PCA + KMeans + IsolationForest (Latest Data)"
    )
    st.plotly_chart(fig)
    explained_var = pca.explained_variance_ratio_
    st.write(f"PCA 寄与率: PC1={explained_var[0]*100:.2f}%, PC2={explained_var[1]*100:.2f}%")

    st.success(f"分析完了！有効データを取得できた企業: {valid_tickers}")
    st.dataframe(df_all)

@st.cache_data
def get_sp500_list_cached():
    """
    get_sp500_list() の結果をキャッシュ
    """
    return get_sp500_list()

@st.cache_data
def fetch_financial_data_cached(symbol: str):
    """
    1銘柄の財務諸表を yfinance から取得し、キャッシュ
    """
    return fetch_financial_data(symbol)

@st.cache_data
def integrate_financial_data_cached(inc, bs, cf):
    """
    取得した財務諸表を統合 (指標計算など) し、その結果をキャッシュ
    """
    return integrate_financial_data(inc, bs, cf)

def find_optimal_k(df_numeric, k_min=2, k_max=10):
    """
    シルエットスコアを用いて、最もスコアが高くなるクラスタ数(k)を返す。
    df_numeric: 数値特徴量のみのDataFrame
    """

    best_k = None
    best_score = -1

    for k in range(k_min, k_max+1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(df_numeric)

        if df_numeric.shape[0] <= k:
            # サンプル数がクラスタ数以下の場合は silhouette_score が計算できないのでスキップ
            continue

        score = silhouette_score(df_numeric, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score
from sklearn.inspection import permutation_importance

def permutation_importance_isoforest(iso, X):
    """
    IsolationForest には feature_importances_ が無いので、
    Permutation Importance を手動で計算する。
    """

    def score_func(estimator, X_test, y_test):
        
        scores = estimator.score_samples(X_test)
        return -np.mean(scores)  # 大きいほど異常傾向とみなす

    y_dummy = np.zeros(len(X))
    base_score = score_func(iso, X, y_dummy)

    results = {}
    for col in X.columns:
        # colを一時的にシャッフル
        saved = X[col].copy()
        X[col] = np.random.permutation(X[col].values)  # シャッフル
        shuffled_score = score_func(iso, X, y_dummy)
        X[col] = saved  # 戻す

        # スコア差分
        importance = shuffled_score - base_score
        results[col] = importance

    
    return pd.Series(results).sort_values(ascending=False)



def ml_analysis_sp500():
    st.write("### 1) S&P 500銘柄一覧を取得 (キャッシュ)")
    df_sp500 = get_sp500_list_cached()
    st.dataframe(df_sp500.head())

    # 業種のプルダウン
    industry_list = sorted(df_sp500["Industry"].dropna().unique())
    selected_industry = st.selectbox("業種(Industry)を選択してください", ["All"] + industry_list)

    if selected_industry != "All":
        df_sp500 = df_sp500[df_sp500["Industry"] == selected_industry]

    st.write("#### 選択後の銘柄一覧")
    st.dataframe(df_sp500.head())

    data_list = []
    valid_symbols = []

    for i, row in df_sp500.iterrows():
        symbol = row["Symbol"]
        sector = row["Sector"]
        industry = row["Industry"]

        inc, bs, cf = fetch_financial_data_cached(symbol)
        df_merged = integrate_financial_data_cached(inc, bs, cf)
        if df_merged.empty:
            continue

        if df_merged.shape[0] > 1:
            df_merged = df_merged.iloc[[-1]]

        df_merged = df_merged.reset_index(drop=True)
        df_merged.index = [symbol]

        df_merged["Sector"] = sector
        df_merged["Industry"] = industry

        data_list.append(df_merged)
        valid_symbols.append(symbol)

    if not data_list:
        st.error("有効データを取得できた銘柄がありません。")
        return

    df_all = pd.concat(data_list)

    st.write("### 2) 結合した財務データ")
    st.dataframe(df_all.head())

    # 数値列のみ抽出＆NaN埋め
    df_numeric = df_all.select_dtypes(include="number").copy()
    df_numeric = df_numeric.fillna(0)

    n_samples, n_features = df_numeric.shape
    if n_samples < 2 or n_features < 2:
        st.warning(f"サンプル数={n_samples}, 特徴量数={n_features} なのでPCA不可")
        return

    st.write("### 3) PCA + KMeans + IsolationForest")

    # ----- PCA -----
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_numeric)
    df_all["pca_x"] = pca_result[:, 0]
    df_all["pca_y"] = pca_result[:, 1]

    # ----- KMeans -----
    k_min = 2
    k_max = 8
    best_k = None
    best_score = -1
    for k in range(k_min, k_max+1):
        if n_samples <= k:
            # サンプル数 <= k だと計算できないのでスキップ
            continue
        tmp_model = KMeans(n_clusters=k, random_state=42)
        labels = tmp_model.fit_predict(df_numeric)
        sil = silhouette_score(df_numeric, labels)
        if sil > best_score:
            best_score = sil
            best_k = k

    st.write(f"最適クラスタ数 (シルエットスコア最大) = {best_k}, スコア={best_score:.4f}")
    if best_k is None:
        st.warning("クラスタ数の自動決定に失敗しました。")
        return

    kmeans = KMeans(n_clusters=best_k, random_state=42)
    clusters = kmeans.fit_predict(df_numeric)
    df_all["cluster"] = clusters.astype(str)

    # ----- IsolationForest -----
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(df_numeric)
    outliers = iso.predict(df_numeric)
    df_all["anomaly"] = outliers  # -1(異常), 1(正常)

    # 可視化
    fig_title = f"S&P 500 - PCA / KMeans / IsolationForest (業種={selected_industry})"
    fig = px.scatter(
        df_all,
        x="pca_x", y="pca_y",
        color="Industry",
        symbol="cluster",
        hover_name=df_all.index,
        title=fig_title
    )
    st.plotly_chart(fig)

    st.write("PCA 寄与率:", pca.explained_variance_ratio_)
    st.success(f"分析完了！(業種={selected_industry}) 有効データを取得できた銘柄: {len(valid_symbols)}件")

    # ----- Permutation Importance で IsolationForest の特徴量重要度を見る -----
    if st.checkbox("Permutation Importance (IsolationForest) を計算する"):
        st.info("特徴量をランダムに入れ替えたときの score_samples の変化量で重要度を推定します。")
        imp_series = permutation_importance_isoforest(iso, df_numeric)
        st.write("Permutation Importance (上位10件):")
        st.dataframe(imp_series.head(10))

    st.dataframe(df_all)

def compute_1yr_return(
    price_df_previous: pd.DataFrame,
    price_df_current: pd.DataFrame,
    lookback_days: int = 5
) -> float:
    
    if price_df_previous.empty or price_df_current.empty:
        return np.nan

    # "Close" or "Adj Close" を優先的に探す
    cols_prev = price_df_previous.columns
    cols_curr = price_df_current.columns
    if "Close" in cols_prev and "Close" in cols_curr:
        target_col = "Close"
    elif "Adj Close" in cols_prev and "Adj Close" in cols_curr:
        target_col = "Adj Close"
    else:
        return np.nan  # 使いたい列が無ければ NaN を返す

    # 前年の末尾N日、当年の末尾N日の平均価格を取得
    old_prices = price_df_previous[target_col].dropna().tail(lookback_days)
    new_prices = price_df_current[target_col].dropna().tail(lookback_days)

    if len(old_prices) == 0 or len(new_prices) == 0:
        return np.nan

    old_mean = old_prices.mean()
    new_mean = new_prices.mean()

    # old_mean が 0 だと割り算できないのでチェック
    if pd.isna(old_mean) or pd.isna(new_mean) or old_mean == 0:
        return np.nan

    return (new_mean / old_mean) - 1


def add_noise_to_numeric_features(df: pd.DataFrame, noise_level=0.01):
    """
    数値列にランダムノイズを付加。
    noise_level=0.01 の場合、平均値の約1%程度の振れ幅でノイズを加える。
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_mean = df[col].abs().mean()
        if pd.isna(col_mean) or col_mean == 0:
            continue
        noise = np.random.randn(len(df)) * noise_level * col_mean
        df[col] = df[col] + noise
    return df