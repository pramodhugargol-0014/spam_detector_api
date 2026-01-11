# backend/app.py

from fastapi import FastAPI
from .schemas import PredictRequest, PredictResponse
from .model_loader import model, vectorizer

app = FastAPI(title="Spam Detection API")

@app.post("/predictSpam", response_model=PredictResponse)
def predict(req: PredictRequest):
    message_vec = vectorizer.transform([req.message])

    prediction = model.predict(message_vec)[0]
    probability = model.predict_proba(message_vec)[0].max()

    label = "spam" if prediction == 1 else "ham"

    return {
        "label": label,
        "confidence": round(float(probability), 3)
    }
