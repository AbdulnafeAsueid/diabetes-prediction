"""Loading, splitting, feature engineering and class balancing."""

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = "data/diabetes_012_health_indicators_BRFSS2015.csv"

TARGET = "Diabetes_012"

# The 21 input features, in the order the saved model and the app expect them.
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

CLASSES = [0, 1, 2]
CLASS_NAMES = ["No Diabetes", "Prediabetes", "Diabetes"]

SEED = 42


def load_data(path=DEFAULT_DATA_PATH):
    """Read the Kaggle CSV. The target becomes an integer class (0, 1, 2)."""
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df[TARGET] = df[TARGET].astype(int)
    return df


def split_xy(df):
    return df.drop(columns=TARGET), df[TARGET]


def train_test(X, y, test_size=0.2, seed=SEED):
    """Stratified split, so both sets keep the class proportions."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def add_engineered_features(df):
    """Three extra features used by the random forest models."""
    df = df.copy()
    df["Risk_Interaction"] = df["BMI"] * df["Age"]  # high BMI together with older age
    df["Heart_Risk"] = df["HighBP"] + df["HighChol"] + df["HeartDiseaseorAttack"]
    df["Health_Struggle"] = df["MentHlth"] + df["PhysHlth"]
    return df


def balance_classes(df, seed=SEED):
    """Resample every class to the size of the diabetes class (~35k rows).

    No diabetes is down-sampled, prediabetes is up-sampled with replacement,
    diabetes is kept as is.

    Note: this is applied to the whole dataset before the train/test split
    (as in the original experiment), so copies of the same prediabetes rows
    can end up in both sets. See "Known limitations" in the README.
    """
    target_count = int((df[TARGET] == 2).sum())
    class_0 = df[df[TARGET] == 0].sample(target_count, random_state=seed)
    class_1 = df[df[TARGET] == 1].sample(target_count, replace=True, random_state=seed)
    class_2 = df[df[TARGET] == 2]
    return pd.concat([class_0, class_1, class_2], axis=0)
