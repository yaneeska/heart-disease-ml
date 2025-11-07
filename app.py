import gradio as gr
import pandas as pd
import joblib

def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    return "Пример результата (модель будет когда мне дадут доступ)"

iface = gr.Interface(
    fn=predict,
    inputs=[gr.Number("Возраст"), gr.Number("Пол"), gr.Number("Тип боли"), gr.Number("Давление"), gr.Number("Холестерин"),
            gr.Number("Сахар >120?"), gr.Number("ECG"), gr.Number("Пульс"), gr.Number("Стенокардия"),
            gr.Number("Oldpeak"), gr.Number("Наклон"), gr.Number("Сосуды"), gr.Number("Thal")],
    outputs="text",
    title="Heart Disease Predictor"
)

iface.launch()
