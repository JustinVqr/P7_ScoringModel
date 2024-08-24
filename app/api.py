from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine
from scripts.P7_data_preprocessing_fct import preprocessing_pipeline  # Importer la fonction de prétraitement
from app.model import best_model  # Charger le modèle

app = FastAPI()

# Configuration de la base de données (ex: PostgreSQL)
DATABASE_URL = "postgresql://user:password@localhost/dbname"
engine = create_engine(DATABASE_URL)

# Modèle de données attendu par l'API
class ClientID(BaseModel):
    SK_ID_CURR: int

@app.post("/predict")
def predict(client_id: ClientID):
    try:
        # Requête pour récupérer les données du client
        query = f"SELECT * FROM clients_table WHERE SK_ID_CURR = {client_id.SK_ID_CURR}"
        client_data = pd.read_sql(query, con=engine)

        # Vérifier si les données du client existent
        if client_data.empty:
            raise HTTPException(status_code=404, detail="ID client non trouvé dans la base de données")

        # Appliquer le prétraitement aux données du client
        processed_data = preprocessing_pipeline(client_data)

        # Faire la prédiction
        prediction = best_model.predict(processed_data)
        probability = best_model.predict_proba(processed_data)[0][1]

        # Retourner les résultats
        return {"prediction": int(prediction[0]), "probability": float(probability)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
