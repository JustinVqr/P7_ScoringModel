from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    prediction = "default"
    probability = 0.8
    return {"prediction": prediction, "probability": probability}
