from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import pickle

# Charger le modèle sauvegardé
with open('app/model/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialisation de FastAPI
app = FastAPI()

# Classe de données client (basée sur les champs requis)
class ClientData(BaseModel):
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    DAYS_BIRTH: int
    FLAG_OWN_CAR: int
    # Ajoutez les autres champs ici selon votre modèle...

# Fonction pour faire des prédictions
def make_prediction(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction

# Message d'accueil
@app.get("/")
def read_root():
    return {"message": "Bonjour, vous êtes bien sur l'application de scoring, hébergée sur Heroku. "
                       "Cette API permet de prédire la probabilité de défaut de paiement pour un client "
                       "en fonction de ses caractéristiques. Envoyez une requête POST à /predict pour obtenir une prédiction."}

# Route FastAPI pour les prédictions
@app.post("/predict")
def predict(client_data: ClientData):
    # Convertir les données en liste pour les utiliser avec le modèle
    input_data = [
        client_data.AMT_CREDIT,
        client_data.AMT_ANNUITY,
        client_data.AMT_GOODS_PRICE,
        client_data.DAYS_BIRTH,
        client_data.FLAG_OWN_CAR,
        # Ajoutez les autres champs ici...
    ]
    
    # Faire la prédiction
    prediction = make_prediction(input_data)
    
    return {"prediction": int(prediction[0])}
