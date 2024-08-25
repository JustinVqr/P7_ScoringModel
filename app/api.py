from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle

app = FastAPI()

# Charger les données des clients
df_clients = pd.read_csv("app/data/clients_data.csv")

# Charger le modèle pré-entraîné
with open("app/model/best_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/predict/{client_id}")
def make_prediction(client_id: int):
    # Vérifier si l'ID client existe dans les données
    client_data = df_clients[df_clients['client_id'] == client_id]
    if client_data.empty:
        raise HTTPException(status_code=404, detail="Client not found")
    
    # Extraire les caractéristiques du client pour la prédiction
    client_features = client_data.drop(columns=["client_id"]).values

    # Faire une prédiction
    prediction = model.predict(client_features)

    # Retourner l'ID client et la prédiction
    return {"client_id": client_id, "prediction": prediction[0]}
