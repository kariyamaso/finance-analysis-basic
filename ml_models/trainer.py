import pandas as pd
import numpy as np
import os
import joblib
import optuna

import lightgbm as lgb
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from finance.sp500_fetcher import get_sp500_tickers
from finance.data_fetcher import fetch_price_data, fetch_financial_data, fetch_company_info
from finance.analyzer import compute_1yr_return, add_noise_to_numeric_features, integrate_financial_data


def build_dataset_for_years(ticker: str, year_prev: int, year_current: int) -> pd.Series:
    """
    1) 前年株価 year_prev(1/1~12/31), 当年株価 year_current(1/1~12/31) → 1年リターン (compute_1yr_return)
    2) 最新の財務データ (integrate_financial_data) → 指標を計算
    3) 1行のSeriesとして返す
    """
    start_prev = f"{year_prev}-01-01"
    end_prev   = f"{year_prev}-12-31"
    start_curr = f"{year_current}-01-01"
    end_curr   = f"{year_current}-12-31"

    # 1) 株価をダウンロード & 1年リターン算出
    df_prev = fetch_price_data(ticker, start_prev, end_prev)
    df_curr = fetch_price_data(ticker, start_curr, end_curr)
    ret_1yr = compute_1yr_return(df_prev, df_curr)

    # 2) 最新の財務データを取得 & 指標計算
    inc, bs, cf = fetch_financial_data(ticker)
    df_fin = integrate_financial_data(inc, bs, cf)
    if df_fin.empty:
        return pd.Series()

    row_fin = df_fin.iloc[0].copy()  # 最新の行(1行のみ)
    row_fin["Ticker"] = ticker
    row_fin["YearPrev"] = year_prev
    row_fin["YearCurr"] = year_current
    row_fin["Return_1yr"] = ret_1yr

    return row_fin


def objective(trial, X_train, y_train, X_valid, y_valid):
    """
    Optunaの目的関数: LightGBM Regressor のパラメータを探索し、RMSEを返す
    ※ Xは標準化済みで渡され、yはそのままのスケール。
    """
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "random_state": 42,
        "n_jobs": -1,
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[LightGBMPruningCallback(trial, "rmse")],
    )
    y_pred = gbm.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred)
    return rmse


def main_training():
    # 1) S&P500 全銘柄を取得
    tickers = get_sp500_tickers()

    # 学習ペア: (2021→2022), (2022→2023)
    # テストペア: (2023→2024)
    train_pairs = [(2021, 2022), (2022, 2023)]
    test_pairs  = [(2023, 2024)]

    # 2) 学習用データを作成
    train_rows = []
    for ticker in tickers:
        for (y_prev, y_curr) in train_pairs:
            row_s = build_dataset_for_years(ticker, y_prev, y_curr)
            if not row_s.empty:
                train_rows.append(row_s)

    # 3) テスト用データを作成
    test_rows = []
    for ticker in tickers:
        for (y_prev, y_curr) in test_pairs:
            row_s = build_dataset_for_years(ticker, y_prev, y_curr)
            if not row_s.empty:
                test_rows.append(row_s)

    df_train = pd.DataFrame(train_rows)
    df_test  = pd.DataFrame(test_rows)

    if df_train.empty:
        print("[WARN] 学習データが取得できませんでした")
        return None
    if df_test.empty:
        print("[WARN] テストデータが取得できませんでした")
        return None

    # 4) NaN処理
    df_train = df_train.dropna(subset=["Return_1yr"])
    df_test  = df_test.dropna(subset=["Return_1yr"])

    if df_train.empty:
        print("[WARN] dropna後、学習データが空です (Return_1yr が全てNaN?)")
        return None
    if df_test.empty:
        print("[WARN] dropna後、テストデータが空です (Return_1yr が全てNaN?)")
        return None

    # 5) 特徴量にノイズ付加 (学習用のみ)
    df_train = add_noise_to_numeric_features(df_train, noise_level=0.01)

    # 6) 目的変数 & 不要列の削除
    y_train = df_train["Return_1yr"]
    y_test  = df_test["Return_1yr"]
    drop_cols = ["Return_1yr", "Ticker", "YearPrev", "YearCurr"]
    X_train = df_train.drop(columns=drop_cols, errors="ignore").fillna(0)
    X_test  = df_test.drop(columns=drop_cols, errors="ignore").fillna(0)

    print("=== Feature Columns (X_train) ===")
    print(X_train.columns.tolist())

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # 7) Optunaで LightGBM のパラメータ探索
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.25, random_state=42
    )

    def optuna_objective(trial):
        return objective(trial, X_tr, y_tr, X_val, y_val)

    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=50)
    best_params = study.best_params
    print("\n==== Best Hyperparameters ====")
    print(best_params)

    # 8) 最終モデルを train全体 で学習（X_train_scaled, y_train）
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train_scaled, y_train)

    # テスト評価 (X_test_scaled)
    y_pred = final_model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"[INFO] Test RMSE: {rmse:.4f}")
    print(f"[INFO] Test MAE : {mae:.4f}")

    df_test["PredictedReturn"] = y_pred
    df_test["Diff"] = df_test["Return_1yr"] - df_test["PredictedReturn"]

    print("\n==== Difference (Return_1yr - PredictedReturn) Summary ====")
    print(df_test["Diff"].describe())

    df_sorted = df_test.sort_values("PredictedReturn", ascending=False)
    print("\n==== Top 10 by Predicted Return ====")
    print(df_sorted.head(10)[["Ticker", "Return_1yr", "PredictedReturn", "Diff"]])

    # Feature importance
    importances = final_model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': X_train.columns,  # カラム名は元の列名を使用
        'importance': importances
    }).sort_values('importance', ascending=False)
    print("\n==== Feature Importance (Top 10) ====")
    print(feature_imp_df.head(10))

    # 9) モデル保存 & CSV出力
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "lgbm_financial_return_model.pkl")
    joblib.dump(final_model, model_path)
    print(f"[INFO] Model saved to {model_path}")

    df_sorted.to_csv("test_return_pred_results.csv", index=False)
    print("[INFO] CSV saved: test_return_pred_results.csv")

    
    return df_sorted, rmse, mae, df_test["Diff"], X_train.columns


def predict_with_saved_model(df_input: pd.DataFrame, scaler_X: StandardScaler) -> pd.DataFrame:
    """
    事前に学習・保存したモデル (lgbm_financial_return_model.pkl) を読み込み、
    新たに df_input (特徴量) に対して推論し、予測結果を返す。
    ※ X のみ標準化しているため、推論前に必ず scaler_X.transform() が必要。
    """
    model_path = os.path.join("models", "lgbm_financial_return_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    loaded_model = joblib.load(model_path)

    # 特徴量を標準化
    X_infer_scaled = scaler_X.transform(df_input.fillna(0))

    # 推論
    preds = loaded_model.predict(X_infer_scaled)

    df_output = df_input.copy()
    df_output["PredictedReturn"] = preds
    return df_output


if __name__ == "__main__":
    results = main_training()
    if results is None:
        print("[INFO] No result (data was empty or dropped).")
    else:
        df_results, rmse, mae, diff_series, feature_cols = results

        print(f"\n[INFO] Done. shape={df_results.shape}")

