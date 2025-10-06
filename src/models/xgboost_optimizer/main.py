"""UFC Fight Prediction - XGBoost training with walk-forward nested cross-validation."""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "train_test"
SAVE_DIR = PROJECT_ROOT / "saved_models" / "xgboost" / "trials"
FINAL_MODEL_DIR = PROJECT_ROOT / "saved_models" / "xgboost" / "no_women"
TRIAL_PLOTS_DIR = SAVE_DIR / "trial_plots"

ACC_THRESHOLD = 0.60
VAL_ACC_SAVE_THRESHOLD = 0.60
LOSS_GAP_THRESHOLD = 0.05

# Configuration: Set to False to drop all columns containing "odd" in their name
INCLUDE_ODDS_COLUMNS = False

SAVE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRIAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data_for_cv(train_path: str | Path | None = None,
                     val_path: str | Path | None = None,
                     include_odds: bool = True,
                     date_column: str = 'current_fight_date'):
    """
    Combine train and validation data for walk-forward nested cross-validation.
    Keeps date column for chronological sorting.
    """
    from pandas.api.types import CategoricalDtype

    train_path = Path(train_path) if train_path is not None else DATA_DIR / "train_data.csv"
    val_path = Path(val_path) if val_path is not None else DATA_DIR / "val_data.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    df = pd.concat([train_df, val_df], ignore_index=True)

    # Ensure date column exists
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in data. Available columns: {df.columns.tolist()}")

    # Convert date column to datetime and sort chronologically
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)

    drop_cols = ["winner", "fighter_a", "fighter_b", "date"]
    # Keep date_column for now, will separate it later

    # Optionally drop odds columns
    if not include_odds:
        odds_cols = [c for c in df.columns if "odd" in c.lower()]
        drop_cols.extend(odds_cols)
        print(f"Dropping {len(odds_cols)} odds columns: {odds_cols}")

    feature_cols = [c for c in df.columns if c not in drop_cols and c != date_column]

    X = df[feature_cols].copy()
    y = df["winner"].copy()
    dates = df[date_column].copy()

    obj_cols = list(X.select_dtypes(include="object").columns)
    for column in obj_cols:
        values = X[column].astype("string").fillna("<NA>")
        categories = sorted(values.unique())
        cat_dtype = CategoricalDtype(categories=categories, ordered=False)
        X[column] = values.astype(cat_dtype)

    print(f"Combined data for walk-forward nested CV: {X.shape}")
    print(f"Date range: {dates.min()} to {dates.max()}")
    if obj_cols:
        print(f"Categorical cols: {obj_cols}")

    return X, y, dates


def get_walk_forward_splits(n_samples: int, n_splits: int = 5, min_train_ratio: float = 0.5):
    """
    Generate walk-forward (expanding window) splits for time series cross-validation.

    Args:
        n_samples: Total number of samples
        n_splits: Number of folds
        min_train_ratio: Minimum ratio of data to use for first training set

    Returns:
        List of (train_indices, test_indices) tuples
    """
    min_train_size = int(n_samples * min_train_ratio)
    remaining_samples = n_samples - min_train_size
    test_size = remaining_samples // n_splits

    splits = []
    for i in range(n_splits):
        train_end = min_train_size + (i * test_size)
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)

        # Ensure we have at least some test samples
        if test_end <= test_start:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, test_idx))

    return splits


def plot_trial_metrics(train_logloss_curves, val_logloss_curves,
                       train_error_curves, val_error_curves,
                       plot_path: Path, trial_number: int, outer_fold: int) -> None:
    """Plot aggregated loss and accuracy curves for an Optuna trial."""
    if not train_logloss_curves or not val_logloss_curves:
        return

    max_len = max(len(curve) for curve in train_logloss_curves + val_logloss_curves)

    def _mean_curve(curves, transform=None):
        arr = np.full((len(curves), max_len), np.nan, dtype=float)
        for idx, curve in enumerate(curves):
            values = np.asarray(curve, dtype=float)
            if transform is not None:
                values = transform(values)
            arr[idx, :len(values)] = values
        return np.nanmean(arr, axis=0)

    mean_train_loss = _mean_curve(train_logloss_curves)
    mean_val_loss = _mean_curve(val_logloss_curves)
    mean_train_acc = _mean_curve(train_error_curves, transform=lambda x: 1.0 - x)
    mean_val_acc = _mean_curve(val_error_curves, transform=lambda x: 1.0 - x)

    iterations = np.arange(1, max_len + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Optuna Trial {trial_number} - Outer Fold {outer_fold}")

    for idx, curve in enumerate(train_logloss_curves):
        rounds = np.arange(1, len(curve) + 1)
        axes[0].plot(rounds, curve, color="tab:blue", alpha=0.3,
                     label="Train Logloss" if idx == 0 else None)
    for idx, curve in enumerate(val_logloss_curves):
        rounds = np.arange(1, len(curve) + 1)
        axes[0].plot(rounds, curve, color="tab:orange", alpha=0.5,
                     label="Validation Logloss" if idx == 0 else None)
    axes[0].plot(iterations, mean_train_loss, color="tab:blue", linewidth=2,
                 label="Train Logloss (mean)")
    axes[0].plot(iterations, mean_val_loss, color="tab:orange", linewidth=2,
                 label="Validation Logloss (mean)")
    axes[0].set_xlabel("Boosting Rounds")
    axes[0].set_ylabel("Log Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    for idx, curve in enumerate(train_error_curves):
        rounds = np.arange(1, len(curve) + 1)
        axes[1].plot(rounds, 1.0 - np.asarray(curve), color="tab:green", alpha=0.3,
                     label="Train Accuracy" if idx == 0 else None)
    for idx, curve in enumerate(val_error_curves):
        rounds = np.arange(1, len(curve) + 1)
        axes[1].plot(rounds, 1.0 - np.asarray(curve), color="tab:red", alpha=0.5,
                     label="Validation Accuracy" if idx == 0 else None)
    axes[1].plot(iterations, mean_train_acc, color="tab:green", linewidth=2,
                 label="Train Accuracy (mean)")
    axes[1].plot(iterations, mean_val_acc, color="tab:red", linewidth=2,
                 label="Validation Accuracy (mean)")
    axes[1].set_xlabel("Boosting Rounds")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def aggregate_best_params(best_params_per_fold):
    """Aggregate hyperparameters from all folds by taking median/mode."""
    aggregated = {}

    # Get all parameter names
    param_names = set()
    for params in best_params_per_fold:
        param_names.update(params.keys())

    for param_name in param_names:
        values = [params[param_name] for params in best_params_per_fold]

        # Use median for numeric parameters
        if isinstance(values[0], (int, float)):
            aggregated[param_name] = type(values[0])(np.median(values))
        else:
            # Use mode for categorical
            from collections import Counter
            aggregated[param_name] = Counter(values).most_common(1)[0][0]

    return aggregated


def walk_forward_nested_cv(X, y, dates, outer_cv: int = 5, inner_cv: int = 3,
                           optuna_trials: int = 100, save_models: bool = True,
                           run_number: int = 1, include_odds: bool = True):
    """
    Run walk-forward nested cross-validation.
    Respects chronological order - never trains on future to predict past.

    Outer loop: Expanding window for model evaluation
    Inner loop: Walk-forward for hyperparameter tuning (within each outer training set)
    """
    print("\n" + "=" * 70)
    print(f"RUN {run_number} | Walk-Forward Nested CV")
    print(f"Outer folds: {outer_cv} | Inner folds: {inner_cv} | Optuna trials: {optuna_trials}")
    print("=" * 70)

    # Get outer walk-forward splits (expanding window)
    outer_splits = get_walk_forward_splits(len(X), n_splits=outer_cv, min_train_ratio=0.5)

    outer_accs, outer_aucs = [], []
    train_accs, train_aucs = [], []
    best_params_per_fold = []
    best_n_per_fold = []

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        print(f"\n{'='*70}")
        print(f"  RUN {run_number} | OUTER FOLD {fold_idx}/{len(outer_splits)}")
        print(f"{'='*70}")

        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        dates_tr, dates_te = dates.iloc[tr_idx], dates.iloc[te_idx]

        # Print date ranges
        print(f"  Training period:   {dates_tr.min().date()} to {dates_tr.max().date()} ({len(X_tr)} samples)")
        print(f"  Test period:       {dates_te.min().date()} to {dates_te.max().date()} ({len(X_te)} samples)")

        # Get inner walk-forward splits for hyperparameter tuning
        inner_splits = get_walk_forward_splits(len(X_tr), n_splits=inner_cv, min_train_ratio=0.6)

        fold_plot_dir = TRIAL_PLOTS_DIR / f"run{run_number}_outer_fold_{fold_idx:02d}"
        fold_plot_dir.mkdir(parents=True, exist_ok=True)

        def inner_objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna hyperparameter optimization."""
            params = {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "device": "cuda",
                "enable_categorical": True,
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                "eval_metric": ["logloss", "error"],
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.01, 1.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "early_stopping_rounds": 50,
            }

            fold_aucs = []
            train_logloss_curves, val_logloss_curves = [], []
            train_error_curves, val_error_curves = [], []

            # Evaluate on inner walk-forward splits
            for in_tr_idx, in_va_idx in inner_splits:
                X_in_tr, X_in_va = X_tr.iloc[in_tr_idx], X_tr.iloc[in_va_idx]
                y_in_tr, y_in_va = y_tr.iloc[in_tr_idx], y_tr.iloc[in_va_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_in_tr,
                    y_in_tr,
                    eval_set=[(X_in_tr, y_in_tr), (X_in_va, y_in_va)],
                    verbose=False,
                )

                proba = model.predict_proba(X_in_va)[:, 1]
                fold_aucs.append(roc_auc_score(y_in_va, proba))

                evals_result = model.evals_result()
                train_logloss_curves.append(evals_result["validation_0"]["logloss"])
                val_logloss_curves.append(evals_result["validation_1"]["logloss"])
                train_error_curves.append(evals_result["validation_0"]["error"])
                val_error_curves.append(evals_result["validation_1"]["error"])

            plot_path = fold_plot_dir / f"trial_{trial.number:03d}_metrics.png"
            plot_trial_metrics(
                train_logloss_curves,
                val_logloss_curves,
                train_error_curves,
                val_error_curves,
                plot_path,
                trial.number,
                fold_idx,
            )

            return float(np.mean(fold_aucs))

        # Optimize hyperparameters using inner walk-forward CV
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random.randint(0, 100000))
        )
        study.optimize(inner_objective, n_trials=optuna_trials)

        best_params = dict(study.best_params)
        best_params_per_fold.append(best_params)
        print(f"\n  Inner CV Best AUC: {study.best_value:.4f}")
        print(f"  Best hyperparameters found:")
        for key, value in best_params.items():
            print(f"    {key}: {value}")

        # Train final model for this fold with best hyperparameters
        # Use last 15% of training data as validation for early stopping
        val_split_point = int(len(X_tr) * 0.85)
        X_tr2, X_va2 = X_tr.iloc[:val_split_point], X_tr.iloc[val_split_point:]
        y_tr2, y_va2 = y_tr.iloc[:val_split_point], y_tr.iloc[val_split_point:]

        final_params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": "cpu",
            "enable_categorical": True,
            "eval_metric": ["logloss", "error"],
            "early_stopping_rounds": 50,
            **best_params,
        }

        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(
            X_tr2,
            y_tr2,
            eval_set=[(X_tr2, y_tr2), (X_va2, y_va2)],
            verbose=False,
        )

        # Get metrics at best iteration
        evals_result = final_model.evals_result()
        train_loss_curve = evals_result["validation_0"]["logloss"]
        val_loss_curve = evals_result["validation_1"]["logloss"]
        train_error_curve = evals_result["validation_0"]["error"]
        val_error_curve = evals_result["validation_1"]["error"]

        best_iteration = getattr(final_model, "best_iteration", None)
        metric_index = int(best_iteration) if best_iteration is not None else len(val_loss_curve) - 1
        train_loss_at_best = float(train_loss_curve[metric_index])
        val_loss_at_best = float(val_loss_curve[metric_index])
        train_acc_at_best = 1.0 - float(train_error_curve[metric_index])
        val_acc_at_best = 1.0 - float(val_error_curve[metric_index])
        loss_gap = train_loss_at_best - val_loss_at_best
        loss_gap_abs = abs(loss_gap)

        best_n_estimators = (
            int(best_iteration) + 1
            if best_iteration is not None
            else final_model.get_params().get("n_estimators", 800)
        )
        best_n_per_fold.append(int(best_n_estimators))

        # Retrain on full outer training set with optimal n_estimators
        final_model = xgb.XGBClassifier(
            **{**final_params, "n_estimators": int(best_n_estimators), "early_stopping_rounds": None}
        )
        final_model.fit(X_tr, y_tr, verbose=False)

        # Evaluate on outer test set (future data)
        te_proba = final_model.predict_proba(X_te)[:, 1]
        te_acc = accuracy_score(y_te, (te_proba > 0.5).astype(int))
        te_auc = roc_auc_score(y_te, te_proba)

        # Evaluate on outer training set
        tr_proba = final_model.predict_proba(X_tr)[:, 1]
        tr_acc = accuracy_score(y_tr, (tr_proba > 0.5).astype(int))
        tr_auc = roc_auc_score(y_tr, tr_proba)

        outer_accs.append(te_acc)
        outer_aucs.append(te_auc)
        train_accs.append(tr_acc)
        train_aucs.append(tr_auc)

        # Print comprehensive fold statistics
        print(f"\n  {'─'*66}")
        print(f"  FOLD {fold_idx} RESULTS")
        print(f"  {'─'*66}")
        print(f"  Optimal n_estimators: {best_n_estimators}")
        print(f"\n  Holdout Validation (from early stopping):")
        print(f"    Train ACC:      {train_acc_at_best:.4f} | Loss: {train_loss_at_best:.4f}")
        print(f"    Val ACC:        {val_acc_at_best:.4f} | Loss: {val_loss_at_best:.4f}")
        print(f"    Loss Gap:       {loss_gap:+.4f} (abs: {loss_gap_abs:.4f})")
        print(f"\n  Outer Split Performance (Future Prediction):")
        print(f"    Train ACC:      {tr_acc:.4f} | AUC: {tr_auc:.4f}")
        print(f"    Test ACC:       {te_acc:.4f} | AUC: {te_auc:.4f}")
        print(f"  {'─'*66}")

        # Save model if it meets quality thresholds
        if (
            save_models
            and val_acc_at_best >= ACC_THRESHOLD
            and val_acc_at_best >= VAL_ACC_SAVE_THRESHOLD
            and loss_gap_abs <= LOSS_GAP_THRESHOLD
        ):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = SAVE_DIR / f"run{run_number}_walkforward_fold{fold_idx}_acc{te_acc:.3f}_{timestamp}.json"
            final_model.save_model(str(model_path))

            metadata_path = SAVE_DIR / f"run{run_number}_walkforward_fold{fold_idx}_metadata_{timestamp}.json"
            metadata = {
                "run_number": run_number,
                "fold": fold_idx,
                "validation_method": "walk_forward",
                "include_odds": include_odds,
                "date_ranges": {
                    "train_start": str(dates_tr.min().date()),
                    "train_end": str(dates_tr.max().date()),
                    "test_start": str(dates_te.min().date()),
                    "test_end": str(dates_te.max().date()),
                },
                "params": {**final_params, "n_estimators": int(best_n_estimators), "early_stopping_rounds": None},
                "metrics": {
                    "train_holdout_acc": float(train_acc_at_best),
                    "train_holdout_loss": float(train_loss_at_best),
                    "val_holdout_acc": float(val_acc_at_best),
                    "val_holdout_loss": float(val_loss_at_best),
                    "loss_gap": float(loss_gap),
                    "loss_gap_abs": float(loss_gap_abs),
                    "train_acc": float(tr_acc),
                    "train_auc": float(tr_auc),
                    "test_acc": float(te_acc),
                    "test_auc": float(te_auc),
                },
            }
            metadata_path.write_text(json.dumps(metadata, indent=2))
            print(f"\n  ✓ Saved fold model | val_acc={val_acc_at_best:.3f}, loss_gap={loss_gap_abs:.3f}")

    print("\n" + "=" * 70)
    print(f"  RUN {run_number} | WALK-FORWARD NESTED CV RESULTS")
    print(f"  (Unbiased Performance Estimate on Future Data)")
    print("=" * 70)
    print(f"  Test Accuracy:  {np.mean(outer_accs):.4f} ± {np.std(outer_accs):.4f}")
    print(f"  Test AUC:       {np.mean(outer_aucs):.4f} ± {np.std(outer_aucs):.4f}")
    print(f"  Train Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"  Train AUC:      {np.mean(train_aucs):.4f} ± {np.std(train_aucs):.4f}")
    print("=" * 70)

    return {
        "outer_accs": outer_accs,
        "outer_aucs": outer_aucs,
        "train_accs": train_accs,
        "train_aucs": train_aucs,
        "best_params_per_fold": best_params_per_fold,
        "best_n_per_fold": best_n_per_fold,
    }


def train_final_model_on_all_data(X, y, dates, aggregated_params, median_n_estimators,
                                  run_number, include_odds):
    """Train final production model on all available data."""
    print("\n" + "=" * 70)
    print(f"  RUN {run_number} | TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 70)
    print(f"  Date range: {dates.min().date()} to {dates.max().date()}")
    print(f"  Total samples: {len(X)}")

    final_params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cpu",
        "enable_categorical": True,
        "n_estimators": int(median_n_estimators),
        "eval_metric": ["logloss", "error"],
        "early_stopping_rounds": None,  # No early stopping for final model
        **aggregated_params,
    }

    print(f"\n  Training with aggregated hyperparameters:")
    for key, value in aggregated_params.items():
        print(f"    {key}: {value}")
    print(f"    n_estimators: {median_n_estimators}")

    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X, y, verbose=False)

    # Evaluate on training data (for reference only)
    train_proba = final_model.predict_proba(X)[:, 1]
    train_acc = accuracy_score(y, (train_proba > 0.5).astype(int))
    train_auc = roc_auc_score(y, train_proba)

    print(f"\n  Final Model Training Metrics:")
    print(f"    Train ACC: {train_acc:.4f}")
    print(f"    Train AUC: {train_auc:.4f}")
    print(f"    (Note: These are on training data, not test performance)")
    print(f"    (Expected test performance from walk-forward CV is the true estimate)")

    return final_model, final_params, train_acc, train_auc


def train_xgboost_walkforward(optuna_trials: int = 100, outer_cv: int = 5,
                              inner_cv: int = 3, save_models: bool = True,
                              run_number: int = 1, include_odds: bool = True) -> dict:
    """Entry point for running walk-forward nested cross-validation and training final model."""
    print("=" * 70)
    print(f"  RUN {run_number} | XGBoost Training - Walk-Forward Nested CV")
    print(f"  Include Odds Columns: {include_odds}")
    print(f"  Validation Method: WALK-FORWARD (No Data Leakage)")
    print("=" * 70)

    X, y, dates = load_data_for_cv(include_odds=include_odds)

    # Run walk-forward nested cross-validation
    results = walk_forward_nested_cv(
        X,
        y,
        dates,
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        optuna_trials=optuna_trials,
        save_models=save_models,
        run_number=run_number,
        include_odds=include_odds,
    )

    # Save nested CV summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = SAVE_DIR / f"run{run_number}_walkforward_cv_summary_{timestamp}.json"
    summary = {
        "run_number": run_number,
        "validation_method": "walk_forward_nested_cv",
        "date_range": {
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
        },
        "mean_test_acc": float(np.mean(results["outer_accs"])),
        "std_test_acc": float(np.std(results["outer_accs"])),
        "mean_test_auc": float(np.mean(results["outer_aucs"])),
        "std_test_auc": float(np.std(results["outer_aucs"])),
        "best_params_per_fold": results["best_params_per_fold"],
        "best_n_per_fold": results["best_n_per_fold"],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Run {run_number} walk-forward CV summary saved")

    # Aggregate best hyperparameters across all folds
    aggregated_params = aggregate_best_params(results["best_params_per_fold"])
    median_n_estimators = int(np.median(results["best_n_per_fold"]))

    # Train final model on all data
    final_model, final_params, train_acc, train_auc = train_final_model_on_all_data(
        X, y, dates, aggregated_params, median_n_estimators, run_number, include_odds
    )

    # Save final model
    final_model_path = FINAL_MODEL_DIR / f"run{run_number}_final_model_{timestamp}.json"
    final_model.save_model(str(final_model_path))

    # Save final model metadata
    final_metadata = {
        "run_number": run_number,
        "timestamp": timestamp,
        "validation_method": "walk_forward_nested_cv",
        "include_odds": include_odds,
        "date_range": {
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
        },
        "nested_cv_results": {
            "mean_test_acc": float(np.mean(results["outer_accs"])),
            "std_test_acc": float(np.std(results["outer_accs"])),
            "mean_test_auc": float(np.mean(results["outer_aucs"])),
            "std_test_auc": float(np.std(results["outer_aucs"])),
        },
        "aggregated_params": aggregated_params,
        "final_params": final_params,
        "median_n_estimators": median_n_estimators,
        "best_n_per_fold": results["best_n_per_fold"],
        "training_metrics": {
            "train_acc": float(train_acc),
            "train_auc": float(train_auc),
        },
        "training_samples": len(X),
    }

    final_metadata_path = FINAL_MODEL_DIR / f"run{run_number}_final_model_metadata_{timestamp}.json"
    final_metadata_path.write_text(json.dumps(final_metadata, indent=2))

    print("\n" + "=" * 70)
    print(f"  ✓ Run {run_number} final model saved to: {final_model_path}")
    print(f"  ✓ Metadata saved to: {final_metadata_path}")
    print("=" * 70)
    print(f"\n  Expected test performance (from walk-forward CV):")
    print(f"    Accuracy: {np.mean(results['outer_accs']):.4f} ± {np.std(results['outer_accs']):.4f}")
    print(f"    AUC:      {np.mean(results['outer_aucs']):.4f} ± {np.std(results['outer_aucs']):.4f}")
    print("=" * 70)

    return {
        "run_number": run_number,
        "validation_method": "walk_forward",
        "model_path": str(final_model_path),
        "metadata_path": str(final_metadata_path),
        "mean_test_acc": float(np.mean(results["outer_accs"])),
        "std_test_acc": float(np.std(results["outer_accs"])),
        "mean_test_auc": float(np.mean(results["outer_aucs"])),
        "std_test_auc": float(np.std(results["outer_aucs"])),
    }


def run_multiple_training_sessions(n_runs: int = 5, optuna_trials: int = 100,
                                   outer_cv: int = 5, inner_cv: int = 3,
                                   save_models: bool = True, include_odds: bool = True):
    """Run multiple walk-forward training sessions to generate multiple final models."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print(f"█  WALK-FORWARD NESTED CV - {n_runs} TRAINING RUNS" + " " * (68 - len(f"  WALK-FORWARD NESTED CV - {n_runs} TRAINING RUNS")) + "█")
    print(f"█  Include Odds Columns: {include_odds}" + " " * (68 - len(f"  Include Odds Columns: {include_odds}")) + "█")
    print(f"█  Validation: Time-aware (No Data Leakage)" + " " * 28 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")

    all_run_results = []

    for run_idx in range(1, n_runs + 1):
        print(f"\n\n{'█' * 70}")
        print(f"█  STARTING RUN {run_idx}/{n_runs}")
        print(f"{'█' * 70}\n")

        run_results = train_xgboost_walkforward(
            optuna_trials=optuna_trials,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            save_models=save_models,
            run_number=run_idx,
            include_odds=include_odds,
        )

        all_run_results.append(run_results)

        print(f"\n{'█' * 70}")
        print(f"█  COMPLETED RUN {run_idx}/{n_runs}")
        print(f"{'█' * 70}\n")

    # Save aggregate summary of all runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregate_summary_path = FINAL_MODEL_DIR / f"all_runs_walkforward_summary_{timestamp}.json"

    aggregate_summary = {
        "total_runs": n_runs,
        "timestamp": timestamp,
        "validation_method": "walk_forward_nested_cv",
        "include_odds": include_odds,
        "all_runs": all_run_results,
        "aggregate_statistics": {
            "mean_test_acc": float(np.mean([r["mean_test_acc"] for r in all_run_results])),
            "std_test_acc": float(np.std([r["mean_test_acc"] for r in all_run_results])),
            "mean_test_auc": float(np.mean([r["mean_test_auc"] for r in all_run_results])),
            "std_test_auc": float(np.std([r["mean_test_auc"] for r in all_run_results])),
        },
    }

    aggregate_summary_path.write_text(json.dumps(aggregate_summary, indent=2))

    # Print final summary
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█  ALL WALK-FORWARD TRAINING RUNS COMPLETED" + " " * 25 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print(f"\n  Total models trained: {n_runs}")
    print(f"  Validation method: Walk-Forward (Time-aware, No Data Leakage)")
    print(f"\n  Aggregate Performance Across All Runs:")
    print(f"    Mean Test Accuracy: {aggregate_summary['aggregate_statistics']['mean_test_acc']:.4f} ± "
          f"{aggregate_summary['aggregate_statistics']['std_test_acc']:.4f}")
    print(f"    Mean Test AUC:      {aggregate_summary['aggregate_statistics']['mean_test_auc']:.4f} ± "
          f"{aggregate_summary['aggregate_statistics']['std_test_auc']:.4f}")
    print(f"\n  Individual Run Results:")
    for run_result in all_run_results:
        print(f"    Run {run_result['run_number']}: Acc={run_result['mean_test_acc']:.4f} ± "
              f"{run_result['std_test_acc']:.4f}, AUC={run_result['mean_test_auc']:.4f} ± "
              f"{run_result['std_test_auc']:.4f}")
    print(f"\n  ✓ Aggregate summary saved to: {aggregate_summary_path}")
    print(f"  ✓ All models saved to: {FINAL_MODEL_DIR}")
    print(f"\n  NOTE: These are honest, unbiased estimates from walk-forward CV.")
    print(f"  Models never trained on future data to predict past data.")
    print("\n" + "█" * 70 + "\n")


if __name__ == "__main__":
    run_multiple_training_sessions(
        n_runs=5,
        optuna_trials=20,
        outer_cv=5,
        inner_cv=3,
        save_models=True,
        include_odds=INCLUDE_ODDS_COLUMNS
    )