import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd
import yaml
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import json
import xgboost as xgb


def _load_config(project_root: Path) -> dict:
    config_path = project_root / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_labeled_df(labeled_dir: Path, ticker: str, target_column: str) -> Optional[pd.DataFrame]:
    csv_path = labeled_dir / f"{ticker}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    if target_column not in df.columns:
        return None
    return df


def compute_label_distribution_per_ticker(
    tickers: List[str],
    target_column: str,
    start_period: Optional[str] = None,
    end_period: Optional[str] = None,
) -> pd.DataFrame:
    project_root = _get_project_root()
    labeled_dir = project_root / "data" / "04_labeled"

    records = []

    for ticker in tickers:
        df = _read_labeled_df(labeled_dir, ticker, target_column)
        if df is None or df.empty:
            records.append({
                "ticker": ticker,
                "subset": "all",
                "n": 0,
                "class_-1": 0.0,
                "class_0": 0.0,
                "class_1": 0.0,
                "note": "missing labeled data"
            })
            # Ir para próximo ticker
            continue

        # No pipeline os labels são mapeados {-1:0, 0:1, 1:2}
        # Vamos remontar as classes originais para leitura humana
        series = df[target_column].dropna().astype(int)
        if series.empty:
            # Nenhum rótulo válido
            records.append({
                "ticker": ticker,
                "subset": "all",
                "n": 0,
                "class_-1": 0.0,
                "class_0": 0.0,
                "class_1": 0.0,
                "note": "no labels"
            })
        else:
            # Mapear de volta para {-1, 0, 1}
            reverse_map = {0: -1, 1: 0, 2: 1}
            classes_orig = series.map(reverse_map)

            def summarize(sub_df: pd.Series, subset_name: str) -> None:
                if sub_df is None or sub_df.empty:
                    records.append({
                        "ticker": ticker,
                        "subset": subset_name,
                        "n": 0,
                        "class_-1": 0.0,
                        "class_0": 0.0,
                        "class_1": 0.0,
                        "note": "empty subset"
                    })
                    return
                n = int(sub_df.shape[0])
                value_counts = sub_df.value_counts(normalize=True)
                records.append({
                    "ticker": ticker,
                    "subset": subset_name,
                    "n": n,
                    "class_-1": float(value_counts.get(-1, 0.0)),
                    "class_0": float(value_counts.get(0, 0.0)),
                    "class_1": float(value_counts.get(1, 0.0)),
                    "note": ""
                })

            # All
            summarize(classes_orig, "all")

            # Recorte por período (se fornecido)
            if start_period or end_period:
                classes_period = classes_orig
                if start_period:
                    classes_period = classes_period[classes_period.index >= pd.to_datetime(start_period)]
                if end_period:
                    classes_period = classes_period[classes_period.index <= pd.to_datetime(end_period)]
                summarize(classes_period, f"{start_period or ''}:{end_period or ''}")

            # 2024 por trimestre
            # Q1: 2024-01-01..2024-03-31, Q2..Q4
            for q, (qs, qe) in {
                "2024Q1": ("2024-01-01", "2024-03-31"),
                "2024Q2": ("2024-04-01", "2024-06-30"),
                "2024Q3": ("2024-07-01", "2024-09-30"),
                "2024Q4": ("2024-10-01", "2024-12-31"),
            }.items():
                sub = classes_orig[(classes_orig.index >= qs) & (classes_orig.index <= qe)]
                summarize(sub, q)

    return pd.DataFrame.from_records(records)


def main():
    project_root = _get_project_root()
    config = _load_config(project_root)

    tickers = list(config["data"]["tickers"]) if isinstance(config["data"]["tickers"], list) else []
    target_col = config["model_training"]["target_column"]

    df = compute_label_distribution_per_ticker(
        tickers=tickers,
        target_column=target_col,
        start_period=config["model_training"].get("test_start_date", None),
        end_period=config["model_training"].get("test_end_date", None),
    )

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_csv = reports_dir / f"diagnostics_label_distribution_{ts}.csv"
    df.to_csv(out_csv, index=False)

    # Print resumo compacto no console
    by_ticker = (
        df[df["subset"] == "all"][["ticker", "n", "class_-1", "class_0", "class_1"]]
        .sort_values("ticker")
    )
    print("Resumo de distribuição de rótulos (ALL):")
    print(by_ticker.to_string(index=False))
    print(f"Arquivo salvo em: {out_csv}")

    # ===== Curvas Precision-Recall para classe Up (2) no período de validação =====
    # Vamos carregar modelos e dados rotulados, e calcular PR por ticker na janela de validação
    features_dir = project_root / "data" / "03_features"
    labeled_dir = project_root / "data" / "04_labeled"
    models_dir = project_root / config["model_training"].get("model_output_path", "models/01_xgboost/")

    pr_rows = []
    val_start = pd.to_datetime(config["model_training"]["validation_start_date"]) if config["model_training"].get("validation_start_date") else None
    val_end = pd.to_datetime(config["model_training"]["validation_end_date"]) if config["model_training"].get("validation_end_date") else None

    for ticker in tickers:
        model_path = models_dir / f"{ticker}.json"
        labeled_csv = labeled_dir / f"{ticker}.csv"

        if not model_path.exists() or not labeled_csv.exists():
            pr_rows.append({
                "ticker": ticker,
                "avg_precision_up": np.nan,
                "note": "missing model or labeled data"
            })
            continue

        df_labeled = pd.read_csv(labeled_csv, index_col="Date", parse_dates=True)
        if target_col not in df_labeled.columns:
            pr_rows.append({
                "ticker": ticker,
                "avg_precision_up": np.nan,
                "note": "missing target in labeled"
            })
            continue

        # Split janela de validação
        if val_start is not None and val_end is not None:
            df_val = df_labeled[(df_labeled.index >= val_start) & (df_labeled.index <= val_end)].copy()
        else:
            df_val = df_labeled.copy()

        if df_val.empty:
            pr_rows.append({
                "ticker": ticker,
                "avg_precision_up": np.nan,
                "note": "empty validation subset"
            })
            continue

        # Preparar X e y (classe up = 2 na codificação armazenada)
        y_true = df_val[target_col].astype(int).values
        X_val = df_val.drop(columns=[target_col])

        # Carregar modelo XGBoost a partir do JSON salvo
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        dval = xgb.DMatrix(X_val)
        proba = booster.predict(dval)

        if proba.ndim == 2 and proba.shape[1] >= 3:
            p_up = proba[:, 2]
        else:
            # fallback: se binário, considerar coluna 1
            p_up = proba[:, -1] if proba.ndim == 2 else proba

        # Converter y_true para binário: up=1, demais=0
        y_up = (y_true == 2).astype(int)
        try:
            ap = average_precision_score(y_up, p_up) if np.unique(y_up).size > 1 else np.nan
            precision, recall, _ = precision_recall_curve(y_up, p_up)
            pr_rows.append({
                "ticker": ticker,
                "avg_precision_up": float(ap) if ap is not None else np.nan,
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "note": ""
            })
        except Exception as e:
            pr_rows.append({
                "ticker": ticker,
                "avg_precision_up": np.nan,
                "note": f"error: {e}"
            })

    pr_df = pd.DataFrame(pr_rows)
    pr_csv = reports_dir / f"diagnostics_pr_curves_{ts}.csv"
    pr_json = reports_dir / f"diagnostics_pr_curves_{ts}.json"
    pr_df.drop(columns=["precision", "recall"], errors="ignore").to_csv(pr_csv, index=False)
    with open(pr_json, "w", encoding="utf-8") as f:
        json.dump(pr_rows, f, ensure_ascii=False, indent=2)
    print(f"PR/AUC-PR salvos em: {pr_csv} e {pr_json}")

    # ===== Calibração de probabilidades e Brier score por ticker (validação) =====
    # Produz reliability data (bins) e Brier score por ticker. Importações locais para evitar erro se pacote ausente.
    try:
        from sklearn.metrics import brier_score_loss
    except Exception:
        brier_score_loss = None

    calib_rows = []
    calib_json_rows = []
    for ticker in tickers:
        model_path = models_dir / f"{ticker}.json"
        labeled_csv = labeled_dir / f"{ticker}.csv"
        if not model_path.exists() or not labeled_csv.exists():
            calib_rows.append({
                "ticker": ticker,
                "brier_up": np.nan,
                "note": "missing model or labeled data"
            })
            continue

        df_labeled = pd.read_csv(labeled_csv, index_col="Date", parse_dates=True)
        if target_col not in df_labeled.columns:
            calib_rows.append({
                "ticker": ticker,
                "brier_up": np.nan,
                "note": "missing target in labeled"
            })
            continue

        df_val = df_labeled
        if val_start is not None and val_end is not None:
            df_val = df_val[(df_val.index >= val_start) & (df_val.index <= val_end)].copy()
        if df_val.empty:
            calib_rows.append({
                "ticker": ticker,
                "brier_up": np.nan,
                "note": "empty validation subset"
            })
            continue

        y_true = df_val[target_col].astype(int).values
        y_up = (y_true == 2).astype(int)
        X_val = df_val.drop(columns=[target_col])

        booster = xgb.Booster()
        booster.load_model(str(model_path))
        dval = xgb.DMatrix(X_val)
        proba = booster.predict(dval)
        p_up = proba[:, 2] if proba.ndim == 2 and proba.shape[1] >= 3 else (proba[:, -1] if proba.ndim == 2 else proba)

        # Reliability bins
        bins = np.linspace(0.0, 1.0, 11)
        bin_indices = np.digitize(p_up, bins) - 1
        rel_points = []
        for b in range(len(bins) - 1):
            mask = bin_indices == b
            if mask.any():
                avg_pred = float(np.mean(p_up[mask]))
                avg_true = float(np.mean(y_up[mask]))
                count = int(np.sum(mask))
                rel_points.append({"bin_left": float(bins[b]), "bin_right": float(bins[b+1]), "avg_pred": avg_pred, "avg_true": avg_true, "count": count})
            else:
                rel_points.append({"bin_left": float(bins[b]), "bin_right": float(bins[b+1]), "avg_pred": np.nan, "avg_true": np.nan, "count": 0})

        # Brier score (se disponível)
        brier = float(brier_score_loss(y_up, p_up)) if brier_score_loss is not None and np.unique(y_up).size > 1 else np.nan
        calib_rows.append({"ticker": ticker, "brier_up": brier, "note": ""})
        calib_json_rows.append({"ticker": ticker, "reliability": rel_points, "brier_up": brier})

    calib_csv = reports_dir / f"diagnostics_calibration_brier_{ts}.csv"
    calib_json = reports_dir / f"diagnostics_calibration_brier_{ts}.json"
    pd.DataFrame(calib_rows).to_csv(calib_csv, index=False)
    with open(calib_json, "w", encoding="utf-8") as f:
        json.dump(calib_json_rows, f, ensure_ascii=False, indent=2)
    print(f"Calibração/Brier salvos em: {calib_csv} e {calib_json}")

    # ===== Sanity check de alinhamento (p_up(t) vs retornos) =====
    # Mede correlação entre p_up(t) e retorno Close_{t+1}/Close_t - 1 (esperado positivo) e também mesma barra (esperado ~0)
    align_rows = []
    for ticker in tickers:
        model_path = models_dir / f"{ticker}.json"
        labeled_csv = labeled_dir / f"{ticker}.csv"
        if not model_path.exists() or not labeled_csv.exists():
            align_rows.append({"ticker": ticker, "corr_next_return": np.nan, "corr_same_return": np.nan, "note": "missing model or labeled data"})
            continue

        df_lbl = pd.read_csv(labeled_csv, index_col="Date", parse_dates=True)
        if target_col not in df_lbl.columns:
            align_rows.append({"ticker": ticker, "corr_next_return": np.nan, "corr_same_return": np.nan, "note": "missing target in labeled"})
            continue

        # Preparar features e preços
        df_tmp = df_lbl.copy()
        # Garantir OHLC para retornos; se não houver no labeled, não calcula
        has_prices = all(col in df_tmp.columns for col in ["Open", "Close"])
        if not has_prices:
            align_rows.append({"ticker": ticker, "corr_next_return": np.nan, "corr_same_return": np.nan, "note": "missing OHLC in labeled"})
            continue

        y_true = df_tmp[target_col].astype(int)
        X_all = df_tmp.drop(columns=[target_col])

        booster = xgb.Booster(); booster.load_model(str(model_path))
        dall = xgb.DMatrix(X_all)
        proba_all = booster.predict(dall)
        p_up_all = proba_all[:, 2] if proba_all.ndim == 2 and proba_all.shape[1] >= 3 else (proba_all[:, -1] if proba_all.ndim == 2 else proba_all)

        # Retornos
        close = df_tmp["Close"]
        same_return = close.pct_change().fillna(0.0)
        next_return = close.shift(-1) / close - 1.0

        # Alinhar comprimentos
        m = min(len(p_up_all), len(next_return))
        p_cut = pd.Series(p_up_all[:m], index=close.index[:m])
        nr_cut = next_return.iloc[:m]
        sr_cut = same_return.iloc[:m]

        corr_next = float(pd.concat([p_cut, nr_cut], axis=1).corr().iloc[0, 1]) if nr_cut.notna().sum() > 5 else np.nan
        corr_same = float(pd.concat([p_cut, sr_cut], axis=1).corr().iloc[0, 1]) if sr_cut.notna().sum() > 5 else np.nan

        align_rows.append({"ticker": ticker, "corr_next_return": corr_next, "corr_same_return": corr_same, "note": ""})

    align_csv = reports_dir / f"diagnostics_alignment_{ts}.csv"
    pd.DataFrame(align_rows).to_csv(align_csv, index=False)
    print(f"Alinhamento salvo em: {align_csv}")

    # ===== Blueprint de sensibilidade do backtest (não executa) =====
    sensitivity_blueprint = {
        "thresholds": [round(x, 2) for x in np.linspace(0.4, 0.8, 21).tolist()],
        "hold_min_days": [0, 1, 2],
        "position_sizing": [
            {"method": "linear", "min_fraction": 0.3, "max_fraction": 1.0},
            {"method": "sigmoid", "min_fraction": 0.2, "max_fraction": 1.0}
        ],
        "apply_hysteresis": True,
        "sell_threshold_offset": 0.05
    }
    sensitivity_json = reports_dir / f"diagnostics_backtest_sensitivity_blueprint_{ts}.json"
    with open(sensitivity_json, "w", encoding="utf-8") as f:
        json.dump(sensitivity_blueprint, f, ensure_ascii=False, indent=2)
    print(f"Blueprint de sensibilidade salvo em: {sensitivity_json}")


if __name__ == "__main__":
    main()


