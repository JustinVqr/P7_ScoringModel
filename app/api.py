from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from scripts.P7_data_preprocessing_fct import preprocessing_pipeline  # Importer la fonction de prétraitement
from app.model import best_model  # Charger le modèle

app = FastAPI()

# Chemin vers le fichier CSV contenant les données des clients
DATA_FILE = "data/preprocessed/preprocessed_data.csv"  # Chemin ajusté en fonction de l'emplacement de votre fichier

# Modèle de données attendu par l'API
class ClientID(BaseModel):
    SK_ID_CURR: int

@app.post("/predict")
def predict(client_id: ClientID):
    try:
        # Lire les données du fichier CSV
        clients_data = pd.read_csv(DATA_FILE)

        # Filtrer les données du client en fonction de l'ID
        client_data = clients_data[clients_data['SK_ID_CURR'] == client_id.SK_ID_CURR]

        # Vérifier si les données du client existent
        if client_data.empty:
            raise HTTPException(status_code=404, detail="ID client non trouvé dans les données")

        # Appliquer le prétraitement aux données du client
        processed_data = preprocessing_pipeline(client_data)

        # Faire la prédiction
        prediction = best_model.predict(processed_data)
        probability = best_model.predict_proba(processed_data)[0][1]

        # Retourner les résultats
        return {"prediction": int(prediction[0]), "probability": float(probability)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
