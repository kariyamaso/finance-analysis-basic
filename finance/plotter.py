import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def plot_bs(balance_sheet: pd.DataFrame):
    needed_cols = {
        "Total Assets",
        "Total Liabilities Net Minority Interest",
        "Stockholders Equity",
    }

    if balance_sheet.empty:
        st.warning("貸借対照表データが空です。")
        return

    if not needed_cols.issubset(balance_sheet.columns):
        st.warning(
            f"必要な列 {needed_cols} のいずれかが存在しません。\n"
            f"カラム一覧: {balance_sheet.columns.tolist()}"
        )
        return

    df = balance_sheet.rename(columns={
        "Total Assets": "Assets",
        "Total Liabilities Net Minority Interest": "Liabilities",
        "Stockholders Equity": "Equity"
    })

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="総資産",
            x=df.index,
            y=df["Assets"],
            offsetgroup=0,
        )
    )
    fig.add_trace(
        go.Bar(
            name="負債",
            x=df.index,
            y=df["Liabilities"],
            offsetgroup=1,
            base=df["Equity"],
        )
    )
    fig.add_trace(
        go.Bar(
            name="純資産",
            x=df.index,
            y=df["Equity"],
            offsetgroup=1,
        )
    )

    fig.update_layout(
        title="貸借対照表",
        xaxis_title="決算期",
        yaxis_title="金額",
        barmode="group",
    )

    st.plotly_chart(fig)

def plot_pl(income_stmt: pd.DataFrame):
    needed_cols = {
        "Total Revenue",    
        "Gross Profit",     
        "Operating Income",
        "EBIT",
        "Pretax Income",
        "Net Income",
    }

    if income_stmt.empty:
        st.warning("損益計算書データが空です。")
        return

    if not needed_cols.issubset(income_stmt.columns):
        st.warning(
            f"必要な列 {needed_cols} のいずれかが存在しません。\n"
            f"現在のカラム: {income_stmt.columns.tolist()}"
        )
        return

    df = income_stmt.rename(columns={
        "Total Revenue":    "Revenue",
        "Gross Profit":     "GrossProfit",
        "Operating Income": "OpIncome",
        "EBIT":             "OrdIncome",
        "Pretax Income":    "Pretax",
        "Net Income":       "NetIncome",
    })

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="売上高",
            x=df.index,
            y=df["Revenue"],
            offsetgroup=0
        )
    )
    fig.add_trace(
        go.Bar(
            name="売上総利益",
            x=df.index,
            y=df["GrossProfit"],
            offsetgroup=1
        )
    )
    fig.add_trace(
        go.Bar(
            name="営業利益",
            x=df.index,
            y=df["OpIncome"],
            offsetgroup=2
        )
    )
    fig.add_trace(
        go.Bar(
            name="経常利益(EBIT)",
            x=df.index,
            y=df["OrdIncome"],
            offsetgroup=3
        )
    )
    fig.add_trace(
        go.Bar(
            name="税引前当期純利益",
            x=df.index,
            y=df["Pretax"],
            offsetgroup=4
        )
    )
    fig.add_trace(
        go.Bar(
            name="当期純利益",
            x=df.index,
            y=df["NetIncome"],
            offsetgroup=5
        )
    )

    fig.update_layout(
        title="損益計算書 (売上高〜当期純利益)",
        xaxis_title="決算期",
        yaxis_title="金額",
        barmode="group"
    )

    st.plotly_chart(fig)

def plot_cf(cashflow_stmt: pd.DataFrame):
    needed_cols = {
        "Operating Cash Flow",
        "Investing Cash Flow",
        "Financing Cash Flow"
    }

    if cashflow_stmt.empty:
        st.warning("キャッシュフロー計算書データが空です。")
        return

    if not needed_cols.issubset(cashflow_stmt.columns):
        st.warning(
            f"必要な列 {needed_cols} のいずれかが存在しません。\n"
            f"現在のカラム: {cashflow_stmt.columns.tolist()}"
        )
        return

    row_data = cashflow_stmt.iloc[0]
    op_cf = row_data["Operating Cash Flow"]
    inv_cf = row_data["Investing Cash Flow"]
    fin_cf = row_data["Financing Cash Flow"]

    total_cf = op_cf + inv_cf + fin_cf
    if total_cf >= 0:
        total_color = "darkblue"
    else:
        total_color = "darkred"

    import plotly.graph_objects as go
    fig = go.Figure(go.Waterfall(
        x=["営業CF", "投資CF", "財務CF", "最終増減"],
        measure=["relative","relative","relative","total"],
        y=[op_cf, inv_cf, fin_cf, 0],
        textposition="outside",
        decreasing={"marker":{"color":"salmon"}},
        increasing={"marker":{"color":"lightskyblue"}},
        totals={"marker":{"color": total_color}},
        connector={"line":{"color":"gray"}}
    ))

    fig.update_layout(
        title="キャッシュフロー",
        waterfallgap=0.5
    )

    st.plotly_chart(fig)
