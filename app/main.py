import os
from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
from pydantic import BaseModel

# 
# --- Chargement du modèle et des données preprocessées ---
#

# Chemin du modèle pré-entraîné
model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
credit_clf_final = pickle.load(open(model_path, 'rb'))

# Chargement des données preprocessées (processed_data.csv)
data_path = os.path.join(os.path.dirname(__file__), "processed_data.csv")
data = pd.read_csv(data_path, sep=";", index_col="SK_ID_CURR")

# 
# --- Initialisation de FastAPI ---
#
app = FastAPI()

# 
# --- Création de la classe d'entrée (identifiant client) ---
#

# Classe définissant l'entrée de l'API (l'identifiant du client)
class ClientID(BaseModel):
    SK_ID_CURR: int

# 
# --- Création de la classe de prédiction (sortie) ---
#

# Classe définissant le format de sortie de la prédiction (cible)
class Client_Target(BaseModel):
    prediction: int  # Prédiction binaire : 0 ou 1
    probability: float  # Probabilité associée à la prédiction

# 
# --- Endpoint GET pour afficher un message de bienvenue ---
#

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de scoring"}

# 
# --- Requête POST pour obtenir la prédiction à partir de l'identifiant client ---
#

@app.post("/predict", response_model=Client_Target)
def model_predict(client: ClientID):
    """Effectue une prédiction à partir de l'identifiant du client"""
    
    # Vérification si l'identifiant client existe dans les données preprocessées
    if client.SK_ID_CURR not in data.index:
        raise HTTPException(status_code=404, detail="Client ID not found")

    # Récupération des données du client
    client_data = data.loc[client.SK_ID_CURR].values.reshape(1, -1)

    # Calcul de la probabilité de défaut de paiement
    probability = float(credit_clf_final.predict_proba(client_data)[:, 1])

    # Prédiction binaire basée sur un seuil de probabilité de 0.4
    prediction = int(probability >= 0.4)

    # Retourne la prédiction et la probabilité
    return {
        'prediction': prediction,
        'probability': probability
    }
