"""Model definitions: SVM variants, weighted logistic regression, random forests."""

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .data import SEED

# Regularisation strengths tried for the SVMs.
SVM_C_VALUES = [0.1, 1, 5]

# Class weights of the final logistic regression model: missing a diabetes
# case costs 20 times more than missing a healthy one.
LOGREG_CLASS_WEIGHTS = {0: 1, 1: 10, 2: 20}


def recall_for_class2(y_true, y_pred):
    """Recall of the diabetes class, used to select SVM hyperparameters."""
    return recall_score(y_true, y_pred, labels=[2], average="macro")


recall2_scorer = make_scorer(recall_for_class2)


def preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),  # SVMs need scaled features
    ])


def _grid_search(pipe, param_grid, n_jobs=-1):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return GridSearchCV(pipe, param_grid, cv=cv, scoring=recall2_scorer, n_jobs=n_jobs, verbose=1)


def _alphas(n_samples):
    # SGDClassifier uses alpha instead of C: alpha = 1 / (n_samples * C)
    return [1 / (n_samples * C) for C in SVM_C_VALUES]


def linear_svm(n_samples, n_jobs=-1):
    """LinearSVC with grid search. Slow on the full data (about 70 minutes)."""
    pipe = Pipeline([
        ("prep", preprocessor()),
        ("clf", LinearSVC(max_iter=20000, random_state=SEED)),
    ])
    grid = {"clf__C": SVM_C_VALUES, "clf__class_weight": ["balanced"]}
    return _grid_search(pipe, grid, n_jobs)


def sgd_svm(n_samples, n_jobs=-1):
    """Linear SVM trained with SGD (hinge loss). Takes about 30 seconds."""
    pipe = Pipeline([
        ("prep", preprocessor()),
        ("clf", SGDClassifier(loss="hinge", max_iter=20000, random_state=SEED)),
    ])
    grid = {"clf__alpha": _alphas(n_samples), "clf__class_weight": ["balanced", None]}
    return _grid_search(pipe, grid, n_jobs)


def nystroem_svm(n_samples, n_jobs=-1):
    """Approximate RBF-kernel SVM: Nystroem features (300 components) + SGD.

    A full RBF SVC would need an N x N kernel (~250k^2 entries), which does
    not fit in memory, so the kernel is approximated instead.
    """
    pipe = Pipeline([
        ("prep", preprocessor()),
        ("nys", Nystroem(kernel="rbf", n_components=300, random_state=SEED)),
        ("clf", SGDClassifier(loss="hinge", max_iter=20000, random_state=SEED)),
    ])
    grid = {
        "nys__gamma": [0.01, 0.1],
        "clf__alpha": _alphas(n_samples),
        "clf__class_weight": ["balanced", None],
    }
    return _grid_search(pipe, grid, n_jobs)


def logistic_regression():
    """Weighted logistic regression (the final model). Expects scaled inputs."""
    return LogisticRegression(
        class_weight=LOGREG_CLASS_WEIGHTS,
        max_iter=1000,
        random_state=SEED,
        solver="lbfgs",  # multinomial for three classes
    )


def random_forest(n_jobs=-1):
    return RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                  random_state=SEED, n_jobs=n_jobs)


def balanced_random_forest(n_jobs=-1):
    """Each tree is trained on a bootstrap sample with all classes resampled to equal size."""
    return BalancedRandomForestClassifier(n_estimators=200, sampling_strategy="all",
                                          replacement=True, random_state=SEED, n_jobs=n_jobs)
