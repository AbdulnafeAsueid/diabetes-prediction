# Diabetes Prediction

Predicting whether a person has **no diabetes, prediabetes or diabetes** from 21
health, lifestyle and demographic indicators of the U.S. BRFSS 2015 survey.
Three families of models are compared (support vector machines, logistic
regression, random forests), and the final model, a weighted logistic
regression tuned for high recall on diabetes, is served in a small Gradio web app.

Diabetes is a common chronic disease, and catching it early matters for
treatment. The goal of this project is a data-driven screening aid that flags
at-risk people for further testing. It is a student project, **not a medical
device**.

## Repository structure

```
diabetes-prediction/
├── diabetes/                 # Python package
│   ├── data.py               # loading, train/test split, feature engineering, class balancing
│   ├── models.py             # SVMs, logistic regression, random forests
│   └── evaluation.py         # metrics, confusion matrix and ROC plots
├── train.py                  # train and evaluate one or more models
├── eda.py                    # exploratory data analysis figures
├── app/
│   ├── app.py                # Gradio web app (final model)
│   └── requirements.txt      # requirements for Hugging Face Spaces
├── notebooks/
│   └── diabetes_prediction.ipynb   # original notebook code, with outputs
├── data/                     # put the dataset here (see data/README.md)
└── requirements.txt
```

## Installation

Python 3.9 or newer.

```bash
git clone https://github.com/AbdulnafeAsueid/diabetes-prediction.git
cd diabetes-prediction
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

[Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
(Kaggle), file `diabetes_012_health_indicators_BRFSS2015.csv`. Download it and
place it in `data/` (see [data/README.md](data/README.md)).

- 253,680 survey responses, 21 features, no missing values.
- Target `Diabetes_012`: 0 = no diabetes (or only during pregnancy), 1 = prediabetes, 2 = diabetes.
- Strongly imbalanced:

| Class | Count | Share |
|---|---:|---:|
| 0 No diabetes | 213,703 | 84.2% |
| 1 Prediabetes | 4,631 | 1.8% |
| 2 Diabetes | 35,346 | 13.9% |

Features: `HighBP`, `HighChol`, `CholCheck`, `BMI`, `Smoker`, `Stroke`,
`HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`,
`AnyHealthcare`, `NoDocbcCost`, `GenHlth` (1-5), `MentHlth` and `PhysHlth`
(days out of 30), `DiffWalk`, `Sex`, `Age` (13 groups), `Education` (1-6),
`Income` (1-8). The app's input labels describe each one.

## Usage

Exploratory data analysis (class distribution, histograms, correlation matrix, summary statistics):

```bash
python eda.py
```

Train and evaluate models. Results go to `outputs/<model>/` (`metrics.json`,
`report.txt`, `confusion_matrix.png`, `roc.png`):

```bash
python train.py --model logreg                     # final model
python train.py --model sgd_svm nystroem_svm random_forest balanced_random_forest
python train.py --model all                        # includes linear_svm, which takes about 70 minutes
```

Available models: `linear_svm`, `sgd_svm`, `nystroem_svm`, `logreg`,
`random_forest`, `balanced_random_forest`. `--model logreg` also saves
`models/model.joblib` and `models/scaler.joblib`, which the web app uses.

Web app (after training `logreg`):

```bash
python app/app.py
```

Then open the local address Gradio prints (by default http://127.0.0.1:7860).

The notebook in `notebooks/` holds the original experiment code with its outputs;
run it from the `notebooks/` folder.

## Methods

All models use a stratified 80/20 train/test split (`random_state=42`).

### 1. Support vector machines

An RBF-kernel `SVC` is not feasible here: its kernel matrix would have about
250,000² entries (hundreds of GB of memory). Three alternatives were used,
each with standardised features and a 5-fold stratified grid search that
selects hyperparameters by **recall on the diabetes class**:

- **Linear SVM** (`LinearSVC`, `C` in {0.1, 1, 5}, balanced class weights). About 70 minutes.
- **Linear SVM with SGD** (`SGDClassifier`, hinge loss, `alpha = 1 / (n·C)`). About 30-50 seconds.
- **Approximate RBF SVM**: `Nystroem` features (300 components, `gamma` in {0.01, 0.1}) followed by the SGD SVM. About 2-3 minutes.

### 2. Logistic regression (final model)

Standardised features and class weights `{0: 1, 1: 10, 2: 20}`: missing a
diabetes case is penalised 20 times more than missing a healthy one. This
pushes the model to catch nearly all diabetes cases at the cost of many false
alarms, which is the preferred trade-off for screening (a false alarm means one
more test; a missed case can mean an untreated disease).

### 3. Random forests

A plain random forest fell into the accuracy paradox: it predicted "healthy"
almost always and found no prediabetes cases, even with class weights or
SMOTE. What worked better was:

- three engineered features: `Risk_Interaction = BMI × Age`,
  `Heart_Risk = HighBP + HighChol + HeartDiseaseorAttack`,
  `Health_Struggle = MentHlth + PhysHlth`;
- resampling every class to the size of the diabetes class (35,346 rows each:
  no-diabetes down-sampled, prediabetes up-sampled with replacement);
- a **random forest** (100 trees, balanced class weights) and a
  **balanced random forest** from `imbalanced-learn` (200 trees, every class
  resampled to equal size in each tree).

## Results

Test-set results from `train.py` (scikit-learn 1.7.2, imbalanced-learn 0.14.2);
Linear SVM from the original notebook run. Recall is given per class.

| Model | Accuracy | Recall: no diabetes | Recall: prediabetes | Recall: diabetes | Macro F1 | ROC-AUC (weighted, OvR) |
|---|---:|---:|---:|---:|---:|---:|
| Linear SVM (`LinearSVC`) | 0.819 | 0.893 | 0.002 | 0.477 | 0.447 | n/a |
| Linear SVM (SGD) | 0.817 | 0.892 | 0.004 | 0.471 | 0.447 | n/a |
| Nystroem approx. RBF SVM (SGD) | 0.827 | 0.903 | 0.000 | 0.476 | 0.451 | n/a |
| **Weighted logistic regression** | 0.521 | 0.462 | 0.000 | **0.948** | 0.328 | 0.812 |
| Random forest* | 0.820 | 0.692 | 0.996 | 0.771 | 0.818 | 0.932 |
| Balanced random forest* | 0.821 | 0.694 | 0.996 | 0.773 | 0.820 | 0.933 |

\* Evaluated on a test set drawn from the resampled (balanced) data, so these
numbers are not comparable with the rows above; see *Known limitations*.
The SVMs use hinge loss and give no probabilities, hence no ROC-AUC.

**Final model: weighted logistic regression.** It finds about 95% of diabetes
cases in the original, imbalanced test set (6,698 of 7,069), which the team
judged most important for a screening tool, even though it flags many healthy
people (diabetes precision 0.22) and does not find prediabetes.

### What did not help

- **SMOTE** oversampling for the random forest: no clear improvement.
- **Threshold moving** for logistic regression (predict diabetes when its
  probability ≥ 0.22, with balanced class weights): 5-10 points better
  precision and accuracy, but diabetes recall dropped to about 91%, so the
  weighted model was kept.

## Known limitations

- **Prediabetes is hard to separate.** Its indicators (BMI, blood pressure,
  cholesterol, ...) overlap heavily with both other classes; no model trained
  on the original class distribution detects it.
- **Random forest test sets contain copies of training rows.** The classes are
  resampled before the train/test split, and prediabetes is up-sampled with
  replacement, so 99.7% of the prediabetes rows in the forests' test set have
  an exact copy in the training set. Their prediabetes recall (≈1.0) therefore
  measures memorisation, not generalisation, and their test set has the
  balanced class mix rather than the real one. Resampling only the training
  set after the split would give a fair estimate.
- The weighted logistic regression trades accuracy (52%) for diabetes recall.
- The data are self-reported survey answers from one year (2015) in the U.S.

## Future work

- Find additional clinical features that separate prediabetes (e.g. lab values),
  ideally together with medical experts, and possibly split prediabetes into levels.
- Collect more data for the minority classes.
- Try kernel methods and other non-linear models with more compute.

## Deploying the app on Hugging Face Spaces

1. Train the final model: `python train.py --model logreg`.
2. Create a new Space at https://huggingface.co/spaces (SDK: Gradio, hardware: CPU).
3. Upload `app/app.py`, `app/requirements.txt`, `models/model.joblib` and
   `models/scaler.joblib` to the Space's root folder.

The Space installs the requirements and starts the app automatically.

## Authors


- **Abdulelah Hesham Shewoil**: random forest models.
- **Mohammed Rumaan**: SVM models (linear, SGD, Nystroem) and the Hugging Face deployment.
- **Abdulnafe Asueid**: logistic regression model.

Preprocessing, analysis, comparison and documentation were shared by all three.

## Acknowledgements

Dataset: Alex Teboul, *Diabetes Health Indicators Dataset*, Kaggle, derived
from the CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015.

## License

The code is released under the [MIT License](LICENSE). The dataset is not part
of this repository and keeps its own license on Kaggle.
