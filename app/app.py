"""Gradio web app for the weighted logistic regression model.

Run locally (after `python train.py --model logreg`):
    python app/app.py

On Hugging Face Spaces, upload this file, requirements.txt, model.joblib and
scaler.joblib to the same folder.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Model files saved by `python train.py --model logreg`. Looked for next to
# this file first (Hugging Face Spaces), then in the repository's models/ folder.
HERE = Path(__file__).resolve().parent
MODEL_DIR = next((d for d in (HERE, HERE.parent / "models") if (d / "model.joblib").exists()), None)
if MODEL_DIR is None:
    raise FileNotFoundError("model.joblib not found: run `python train.py --model logreg` first")

model = joblib.load(MODEL_DIR / "model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")

# Labels mapping
LABEL_MAP = {
    0: "No Diabetes (0)",
    1: "Prediabetes (1)",
    2: "Diabetes (2)"
}

# Define the order of features expected by the model
FEATURE_ORDER = [
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke",
    "HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
    "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth",
    "MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"
]

def predict_diabetes(
    HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
    HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
    HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth,
    MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
):
    # Build feature vector in correct order
    features = np.array([[
        HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
        HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
        HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth,
        MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
    ]], dtype=float)

    # Scale (with column names, as the scaler was fitted on a DataFrame)
    features_scaled = scaler.transform(pd.DataFrame(features, columns=FEATURE_ORDER))

    # Predict
    pred = int(model.predict(features_scaled)[0])
    probs = model.predict_proba(features_scaled)[0]  # probabilities per class

    # Build readable probability output
    prob_dict = {LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))}

    result_text = f"Predicted class: {LABEL_MAP[pred]}\n"
    result_text += "Probabilities:\n"
    for i in range(len(probs)):
        result_text += f"  {LABEL_MAP[i]}: {probs[i]:.4f}\n"

    return LABEL_MAP[pred], prob_dict, result_text

# Gradio UI components
with gr.Blocks() as demo:
    gr.Markdown("# Diabetes stage predictor")
    gr.Markdown(
        "Provide the patient attributes below. This model predicts `No Diabetes (0)`,"
        " `Prediabetes (1)`, or `Diabetes (2)`. See the model card for limitations."
    )

    with gr.Row():
        with gr.Column():
            HighBP = gr.Dropdown(choices=[0,1], value=0, label="HighBP (0=no,1=yes)")
            HighChol = gr.Dropdown(choices=[0,1], value=0, label="HighChol (0=no,1=yes)")
            CholCheck = gr.Dropdown(choices=[0,1], value=1, label="CholCheck (0 = no cholesterol check in 5 years, 1 = yes cholesterol check in 5 years)")
            BMI = gr.Number(value=25.0, label="Body Mass Index (BMI)")
            Smoker = gr.Dropdown(choices=[0,1], value=0, label="Smoker (Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] 0 = no 1 = yes)")
            Stroke = gr.Dropdown(choices=[0,1], value=0, label="(Ever told) you had a stroke. (0=no,1=yes)")
            HeartDiseaseorAttack = gr.Dropdown(choices=[0,1], value=0, label="coronary heart disease (CHD) or myocardial infarction (MI) (0=no,1=yes)")
            PhysActivity = gr.Dropdown(choices=[0,1], value=1, label="physical activity in past 30 days - not including job (0=no,1=yes)")
            Fruits = gr.Dropdown(choices=[0,1], value=1, label="Consume Fruit 1 or more times per day (0=no,1=yes)")
            Veggies = gr.Dropdown(choices=[0,1], value=1, label="Consume Vegetables 1 or more times per day (0=no,1=yes)")
            HvyAlcoholConsump = gr.Dropdown(choices=[0,1], value=0, label="Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) (0=no,1=yes)")
            AnyHealthcare = gr.Dropdown(choices=[0,1], value=1, label="Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. (0=no,1=yes)")
            NoDocbcCost = gr.Dropdown(choices=[0,1], value=0, label="Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? (0=no,1=yes)")
            GenHlth = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Would you say that in general your health is:(1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor)")
            MentHlth = gr.Slider(minimum=0, maximum=30, step=1, value=0, label="Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? (scale 1-30 days)")
            PhysHlth = gr.Slider(minimum=0, maximum=30, step=1, value=0, label="Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? (scale 1-30 days)")
            DiffWalk = gr.Dropdown(choices=[0,1], value=0, label="Do you have serious difficulty walking or climbing stairs? (0=no,1=yes)")
            Sex = gr.Dropdown(choices=[0,1], value=0, label="Sex (0=female,1=male)")
            Age = gr.Slider(minimum=1, maximum=13, step=1, value=5, label="13-level age category (_AGEG5YR codebook) (1 = 18-24, 9 = 60-64, 13 = 80 or older)")
            Education = gr.Slider(minimum=1, maximum=6, step=1, value=4, label="Education level (EDUCA codebook) scale 1-6, 1 = Never attended school or only kindergarten, 2 = Grades 1 through 8 (Elementary), 3 = Grades 9 through 11 (Some high school), 4 = Grade 12 or GED (High school graduate), 5 = College 1 year to 3 years (Some college or technical school), 6 = College 4 years or more (College graduate)")
            Income = gr.Slider(minimum=1, maximum=8, step=1, value=4, label="Income scale (INCOME2 codebook) scale 1-8, 1 = less than $10,000, 5 = less than $35,000, 8 = $75,000 or more")

        with gr.Column():
            pred_label = gr.Textbox(label="Predicted label")
            probs_output = gr.JSON(label="Probabilities")
            result_text = gr.Textbox(label="Result explanation", lines=8)

    predict_btn = gr.Button("Predict")
    predict_btn.click(
        fn=predict_diabetes,
        inputs=[
            HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
            HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
            HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth,
            MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income
        ],
        outputs=[pred_label, probs_output, result_text]
    )

# For local testing: `python app.py` will launch gradio
if __name__ == "__main__":
    demo.launch()
