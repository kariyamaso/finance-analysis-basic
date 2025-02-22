# app.py

import streamlit as st
from finance.data_fetcher import fetch_financial_data, fetch_company_info, fetch_price_data
from finance.plotter import plot_bs, plot_pl, plot_cf
from finance.analyzer import integrate_financial_data, ml_analysis, ml_analysis_sp500
from datetime import date, timedelta
from ml_models.trainer import main_training

def main():
    st.title("財務諸表分析アプリ")

    if "sp500_run" not in st.session_state:
        st.session_state["sp500_run"] = False

    mode = st.radio(
        "モードを選択してください",
        [
            "単体企業分析",
            "機械学習による複数企業分析",
            "大規模銘柄分析（S&P500 等）",
            "将来の有望株予測(1年リターン)"
        ]
    )

    # ---- 共通のスタイル設定 (CSS) ----
    st.markdown("""
    <style>
    body {
        margin: 0 2rem;
        font-family: "メイリオ", sans-serif;
    }
    .indicator-title {
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 0.3rem;
        color: #333;
    }
    .indicator-value {
        font-size: 32px;
        font-weight: 700;
        color: #0072ff;
        margin-bottom: 0.2rem;
    }
    .indicator-desc {
        font-size: 14px;
        color: #555;
        line-height: 1.4;
        margin-bottom: 1rem;
    }
    .indicator-card {
        background-color: #fafafa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # ========== 単体企業分析 ==========
    if mode == "単体企業分析":
        st.session_state["sp500_run"] = False
        st.subheader("単体企業分析")
        ticker = st.text_input("ティッカーシンボル (例: AAPL, MSFT, 7203.T)", "AAPL")

        if st.button("データ取得"):
            with st.spinner("データ取得中..."):
                # 企業情報
                company_name, sector, industry = fetch_company_info(ticker)
                # 財務諸表データ
                income_stmt, balance_sheet, cashflow_stmt = fetch_financial_data(ticker)

            st.write(f"**企業名**: {company_name if company_name else 'N/A'}")
            st.write(f"**セクター**: {sector if sector else 'N/A'}")
            st.write(f"**業種**: {industry if industry else 'N/A'}")


            # 財務データを統合
            all_data = integrate_financial_data(income_stmt, balance_sheet, cashflow_stmt)
            if all_data.empty:
                st.warning("有効な財務データが見つかりませんでした。")
                return

            # 最新行
            latest_data = all_data.iloc[-1]

            # 貸借対照表グラフ
            plot_bs(balance_sheet)
            # 損益計算書グラフ
            plot_pl(income_stmt)
            # キャッシュフロー計算書グラフ
            plot_cf(cashflow_stmt)

            # 重要指標
            st.header("主要指標 (最新期)")
            if "Equity Ratio (%)" in latest_data:
                eq_ratio_val = latest_data["Equity Ratio (%)"]
                st.markdown(f"""
                <div class="indicator-card">
                    <div class="indicator-title">自己資本比率 (Equity Ratio, %)</div>
                    <div class="indicator-value">{eq_ratio_val:.2f}%</div>
                    <div class="indicator-desc">
                        企業が保有する総資産に対して、株主資本が占める割合を示します。<br>
                        一般に数値が高いほど財務の安定性が高いとみなされます。
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if "Current Ratio" in latest_data:
                cr_val = latest_data["Current Ratio"]
                st.markdown(f"""
                <div class="indicator-card">
                    <div class="indicator-title">流動比率 (Current Ratio)</div>
                    <div class="indicator-value">{cr_val:.2f}</div>
                    <div class="indicator-desc">
                        短期的な支払能力を示し、1.0 (100%) を下回ると資金繰りに注意が必要とされます。
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if "Operating Margin (%)" in latest_data:
                opm_val = latest_data["Operating Margin (%)"]
                st.markdown(f"""
                <div class="indicator-card">
                    <div class="indicator-title">営業利益率 (Operating Margin, %)</div>
                    <div class="indicator-value">{opm_val:.2f}%</div>
                    <div class="indicator-desc">
                        本業の儲け(営業利益)が売上高に占める割合。企業の収益力を測る代表的な指標です。
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if "FCF Margin (%)" in latest_data:
                fcf_val = latest_data["FCF Margin (%)"]
                st.markdown(f"""
                <div class="indicator-card">
                    <div class="indicator-title">フリーキャッシュフローマージン (FCF Margin, %)</div>
                    <div class="indicator-value">{fcf_val:.2f}%</div>
                    <div class="indicator-desc">
                        売上高に対して、実際に手元に残るキャッシュフローの割合です。<br>
                        投資や配当の原資をどの程度確保できるかを示します。
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ========== 複数企業分析 ==========
    elif mode == "機械学習による複数企業分析":
        st.session_state["sp500_run"] = False
        st.subheader("機械学習による複数企業分析")
        st.write("下記に複数のティッカーをカンマ区切りで入力してください。 (例: AAPL, MSFT, GOOG, AMZN)")
        multi_tickers = st.text_input("複数ティッカーを入力", "")

        if st.button("機械学習 分析実行"):
            if not multi_tickers.strip():
                st.warning("企業のティッカーを入力してください。")
                return
            tickers_list = [t.strip() for t in multi_tickers.split(",") if t.strip()]
            with st.spinner("複数企業データを取得・分析中..."):
                ml_analysis(tickers_list)

    # ========== S&P 500分析 ==========
    elif mode == "大規模銘柄分析（S&P500 等）":
        st.subheader("大規模銘柄分析（S&P 500 等）")

        if st.button("S&P 500 分析実行"):
            st.session_state["sp500_run"] = True

        if st.session_state["sp500_run"]:
            with st.spinner("S&P 500 の大量銘柄データをダウンロード or キャッシュから取得中..."):
                ml_analysis_sp500()
        else:
            st.write("S&P 500 分析を行うには、上のボタンを押してください。")

    # ========== 新規: 将来の有望株予測 (1年リターン) ==========
    else:
        st.subheader("将来の有望株予測(1年リターン)")

        st.write("""
        2021→2022, 2022→2023 の期間で学習したモデルを用いて  
        2023→2024 の1年リターンを予測し、上位銘柄を抽出します。
        """)

        if st.button("学習＆予測実行"):
            with st.spinner("学習中＆予測中..."):
                df_results, rmse, mae, diff_series, feature_cols = main_training()

            # df_results が None の場合は何らかの理由でデータが無かった
            if df_results is None:
                st.error("データ不足等により学習が実行できませんでした。")
                return

            # 全銘柄の予測結果の中から上位10件を表示
            df_sorted = df_results.sort_values("PredictedReturn", ascending=False).head(10)

            st.success("学習完了！ 以下が予測リターン上位の銘柄です。")
            st.dataframe(df_sorted[["Ticker", "Return_1yr", "PredictedReturn", "Diff"]])

            # ------ RMSE, MAE を表示 ------
            st.markdown(f"**Test RMSE**: {rmse:.4f}")
            st.markdown(f"**Test MAE** : {mae:.4f}")

            # ------ Diff(実際 - 予測) の分布を表示 ------
            st.markdown("#### 差分 (Return_1yr - PredictedReturn) の統計")
            st.write(diff_series.describe())

            # ヒストグラムを簡易表示 (matplotlib)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            diff_series.hist(ax=ax, bins=20)
            ax.set_title("Histogram of (Return_1yr - PredictedReturn)")
            ax.set_xlabel("Difference")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # ------ 使用している特徴量一覧 ------
            st.markdown("#### 使用している特徴量カラム一覧")
            st.write(list(feature_cols))

if __name__ == "__main__":
    main()
