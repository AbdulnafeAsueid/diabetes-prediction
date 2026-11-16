"""Train and evaluate the models.

Examples:
    python train.py --model logreg            # final model, saves models/model.joblib + models/scaler.joblib
    python train.py --model sgd_svm nystroem_svm
    python train.py --model all               # everything (linear_svm alone takes about 70 minutes)

Results (metrics.json, report.txt, confusion_matrix.png, roc.png) go to outputs/<model>/.
"""

import argparse
import time
from pathlib import Path

import joblib
from sklearn.preprocessing import StandardScaler

from diabetes import models
from diabetes.data import (DEFAULT_DATA_PATH, add_engineered_features, balance_classes,
                           load_data, split_xy, train_test)
from diabetes.evaluation import save_results

SVM_MODELS = ["linear_svm", "sgd_svm", "nystroem_svm"]
FOREST_MODELS = ["random_forest", "balanced_random_forest"]
ALL_MODELS = SVM_MODELS + ["logreg"] + FOREST_MODELS


def run_svm(name, df, n_jobs):
    X_train, X_test, y_train, y_test = train_test(*split_xy(df))
    search = getattr(models, name)(n_samples=len(X_train), n_jobs=n_jobs)
    search.fit(X_train, y_train)
    print(f"best parameters: {search.best_params_}")
    return y_test, search.predict(X_test), None  # hinge loss gives no probabilities


def run_logreg(df, model_dir):
    X_train, X_test, y_train, y_test = train_test(*split_xy(df))
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = models.logistic_regression()
    model.fit(X_train_s, y_train)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    print(f"saved {model_dir / 'model.joblib'} and {model_dir / 'scaler.joblib'}")
    return y_test, model.predict(X_test_s), model.predict_proba(X_test_s)


def run_forest(name, df, n_jobs):
    # Both forests use the engineered features and the balanced dataset.
    balanced = balance_classes(add_engineered_features(df))
    X_train, X_test, y_train, y_test = train_test(*split_xy(balanced))
    model = getattr(models, name)(n_jobs=n_jobs)
    model.fit(X_train, y_train)
    return y_test, model.predict(X_test), model.predict_proba(X_test)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", default=["logreg"], choices=ALL_MODELS + ["all"])
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="path to the Kaggle CSV")
    parser.add_argument("--out", default="outputs", help="folder for metrics and plots")
    parser.add_argument("--model-dir", default="models", help="where the logistic regression model is saved")
    parser.add_argument("--n-jobs", type=int, default=-1, help="CPU cores to use (-1 = all)")
    args = parser.parse_args()

    names = ALL_MODELS if "all" in args.model else args.model
    df = load_data(args.data)
    print(f"loaded {len(df):,} rows from {args.data}")

    for name in names:
        print(f"\n=== {name} ===")
        start = time.time()
        if name in SVM_MODELS:
            y_test, y_pred, y_score = run_svm(name, df, args.n_jobs)
        elif name == "logreg":
            y_test, y_pred, y_score = run_logreg(df, args.model_dir)
        else:
            y_test, y_pred, y_score = run_forest(name, df, args.n_jobs)
        metrics, text = save_results(args.out, name, y_test, y_pred, y_score)
        print(text)
        if "roc_auc_weighted_ovr" in metrics:
            print(f"weighted ROC-AUC (one-vs-rest): {metrics['roc_auc_weighted_ovr']:.4f}")
        print(f"done in {time.time() - start:.0f} s, results in {Path(args.out) / name}")


if __name__ == "__main__":
    main()
