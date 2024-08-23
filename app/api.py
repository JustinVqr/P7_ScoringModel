from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd

app = FastAPI()

# Charger le modèle
with open('app/model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Charger le pipeline de prétraitement
with open('app/model/preprocessing_pipeline.pkl', 'rb') as f:
    preprocessing_pipeline = pickle.load(f)

# Chemin vers le dataset contenant les données clients
DATA_FILE = "app/data/processed_data.csv"

@app.post("/predict")
def predict(SK_ID_CURR: int):
    try:
        # Charger les données clients depuis le fichier CSV
        client_data = pd.read_csv(DATA_FILE, sep=";", index_col='SK_ID_CURR')

        # Vérifier si l'ID du client existe dans les données
        if SK_ID_CURR not in client_data.index:
            raise HTTPException(status_code=404, detail="ID client non trouvé dans le dataset")

        # Extraire les données du client
        client_features = client_data.loc[[SK_ID_CURR]]

        # Appliquer le prétraitement
        processed_data = preprocessing_pipeline.transform(client_features)

        # Faire la prédiction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[0][1]

        # Retourner la prédiction et la probabilité
        return {"prediction": int(prediction[0]), "probability": float(probability)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
