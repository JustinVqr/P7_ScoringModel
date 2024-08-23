from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API FastAPI"}

@app.post("/predict")
def predict(data: dict):
    prediction = "default"
    probability = 0.8
    return {"prediction": prediction, "probability": probability}
