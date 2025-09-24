import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.train_models import (
    create_dynamic_triple_barrier_target,
    split_data,
    objective,
)

import xgboost as xgb
import optuna


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features_csv(features_dir: Path, ticker: str) -> pd.DataFrame:
    csv_path = features_dir / f"{ticker}.csv" if not ticker.endswith(".csv") else features_dir / ticker
    if not csv_path.exists():
        raise FileNotFoundError(f"Features CSV not found: {csv_path}")
    return pd.read_csv(csv_path, index_col=0, parse_dates=True)


def print_distribution(name: str, y: pd.Series) -> None:
    counts = y.value_counts(dropna=False).sort_index()
    ratios = (counts / counts.sum()).round(4)
    print(f"\n{name} - Distribuição do target (0=Perda, 1=Timeout, 2=Lucro):")
    print(pd.DataFrame({"count": counts, "ratio": ratios}))


def diagnose_imbalance(y_train: pd.Series) -> None:
    if y_train.empty:
        print("Aviso: y_train vazio após criação do target. Verifique parâmetros do triple barrier e colunas obrigatórias (OHLC + ATR).")
        return
    ratio_timeout = (y_train == 1).mean()
    ratio_profit = (y_train == 2).mean()
    ratio_loss = (y_train == 0).mean()
    msgs = []
    if ratio_timeout >= 0.75:
        msgs.append(
            "Classe Timeout (1) dominante (>75%). Considere aumentar holding_days e/ou reduzir multiplicadores para facilitar toques em lucro/perda."
        )
    if ratio_profit <= 0.10:
        msgs.append(
            "Classe Lucro (2) muito rara (<=10%). Mercado/parametrização pode estar gerando poucas oportunidades; tente profit_multiplier menor."
        )
    if ratio_loss <= 0.10:
        msgs.append(
            "Classe Perda (0) muito rara (<=10%). Verifique se loss_multiplier não está grande demais para o regime atual."
        )
    if msgs:
        print("\nSinais de desbalanceamento:")
        for m in msgs:
            print(f"- {m}")


def run_optuna_short(x_train: pd.DataFrame, y_train: pd.Series,
                     x_val: pd.DataFrame, y_val: pd.Series, trials: int = 10):
    # Use the same objective and direction used in the main training for consistency
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    baseline = 1.0
    study.optimize(lambda trial: objective(trial, x_train, y_train, x_val, y_val, baseline), n_trials=trials)
    return study


def fit_final_booster(best_params: dict,
                      x_train: pd.DataFrame, y_train: pd.Series,
                      x_val: pd.DataFrame, y_val: pd.Series):
    final_params = {**best_params}
    final_params['use_label_encoder'] = False
    final_params['objective'] = 'multi:softprob'
    final_params['num_class'] = 3
    final_params['eval_metric'] = 'mlogloss'

    x_train_full = pd.concat([x_train, x_val])
    y_train_full = pd.concat([y_train, y_val])

    dtrain_full = xgb.DMatrix(x_train_full, label=y_train_full)
    dval_final = xgb.DMatrix(x_val, label=y_val)

    booster = xgb.train(
        final_params,
        dtrain_full,
        num_boost_round=final_params.get('n_estimators', 500),
        evals=[(dval_final, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return booster


def evaluate_model(booster: xgb.Booster, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    dtest = xgb.DMatrix(x_test)
    proba = booster.predict(dtest)
    pred = np.argmax(proba, axis=1)

    print("\n--- Avaliação no Conjunto de Teste ---")
    print(f"Classes encontradas: {np.unique(np.concatenate([y_test.values, pred]))}")
    print(f"Distribuição y_test: {pd.Series(y_test).value_counts().to_dict()}")
    print(f"Distribuição pred: {pd.Series(pred).value_counts().to_dict()}")

    print("\nRelatório de Classificação (3 classes):")
    print(classification_report(y_test, pred, digits=4, zero_division=0))

    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
    print("Matriz de Confusão [linhas=verdade, colunas=predito] (ordem: 0,1,2):")
    print(cm)

    acc = accuracy_score(y_test, pred)
    print(f"Acurácia: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnóstico Triple Barrier e re-otimização rápida")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker para testar (ex: VIVT3.SA)")
    parser.add_argument("--trials", type=int, default=0, help="Número de trials do Optuna (0 para pular)")
    parser.add_argument("--holding-days", type=int, default=None, dest="holding_days", help="holding_days para o triple barrier")
    parser.add_argument("--profit-multiplier", type=float, default=None, dest="profit_multiplier", help="Multiplicador de lucro (ATR)")
    parser.add_argument("--loss-multiplier", type=float, default=None, dest="loss_multiplier", help="Multiplicador de perda (ATR)")
    parser.add_argument("--no-hybrid", action="store_true", help="Usa método puro ATR (sem híbrido)")
    args = parser.parse_args()

    config = load_config()

    features_dir = PROJECT_ROOT / config["data"]["features_data_path"]
    tickers = config["data"].get("tickers", [])
    ticker = args.ticker or (tickers[0] if tickers else None)
    if not ticker:
        raise ValueError("Nenhum ticker definido. Informe --ticker ou preencha data.tickers no config.yaml")

    tb_cfg = config["model_training"].get("triple_barrier", {})
    holding_days = args.holding_days if args.holding_days is not None else tb_cfg.get("holding_days", 7)
    profit_multiplier = args.profit_multiplier if args.profit_multiplier is not None else tb_cfg.get("profit_multiplier", 2.0)
    loss_multiplier = args.loss_multiplier if args.loss_multiplier is not None else tb_cfg.get("loss_multiplier", 1.5)
    use_hybrid = not args.no_hybrid

    print(f"Carregando features de {ticker}...")
    df = load_features_csv(features_dir, ticker)

    target_col = config["model_training"]["target_column"]

    print("Criando target ternário (Triple Barrier)...")
    df = create_dynamic_triple_barrier_target(
        df,
        target_col,
        holding_days=holding_days,
        profit_multiplier=profit_multiplier,
        loss_multiplier=loss_multiplier,
        use_hybrid=use_hybrid,
    )

    # Split datasets
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(
        df,
        config["model_training"]["train_final_date"],
        config["model_training"]["validation_start_date"],
        config["model_training"]["validation_end_date"],
        config["model_training"]["test_start_date"],
        config["model_training"]["test_end_date"],
        target_col,
    )

    # Diagnostics: class distributions
    print_distribution("Treino", y_train)
    print_distribution("Validação", y_val)
    print_distribution("Teste", y_test)

    diagnose_imbalance(y_train)

    # Optional quick re-optimization and evaluation
    if args.trials and args.trials > 0:
        print(f"\nExecutando re-otimização rápida com Optuna (trials={args.trials})...")
        study = run_optuna_short(x_train, y_train, x_val, y_val, trials=args.trials)
        print("\nMelhores hiperparâmetros encontrados:")
        print(study.best_params)

        booster = fit_final_booster(study.best_params, x_train, y_train, x_val, y_val)
        evaluate_model(booster, x_test, y_test)
    else:
        print("\nPulando re-otimização (use --trials N para rodar um estudo curto do Optuna).")


if __name__ == "__main__":
    main()
