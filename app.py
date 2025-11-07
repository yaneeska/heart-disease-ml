import gradio as gr
import numpy as np
import onnxruntime as ort

# Загружаем модель ONNX
sess = ort.InferenceSession("model.onnx")

def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Формируем входной массив
    X = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal]]).astype(np.float32)
    input_name = sess.get_inputs()[0].name
    pred = sess.run(None, {input_name: X})[0]

    # Обработка предсказания
    if pred.shape == (1, 1):
        label = int(pred[0][0] > 0.5)
    else:
        label = int(np.argmax(pred, axis=1)[0])

    return "💔 Болезнь сердца обнаружена" if label == 1 else "❤️ Здоров"

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Возраст"), gr.Number(label="Пол (1-м,0-ж)"), gr.Number(label="Тип боли"),
        gr.Number(label="Давление"), gr.Number(label="Холестерин"), gr.Number(label="Сахар >120?"),
        gr.Number(label="ECG"), gr.Number(label="Пульс"), gr.Number(label="Стенокардия"),
        gr.Number(label="Oldpeak"), gr.Number(label="Наклон"), gr.Number(label="Сосуды"), gr.Number(label="Thal")
    ],
    outputs="text",
    title="Heart Disease Predictor"
)

iface.launch()



