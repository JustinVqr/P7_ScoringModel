from fastapi import FastAPI
import pickle
import pandas as pd

# Importer la fonction de prétraitement
from scripts.P7_data_preprocessing_fct import preprocess_data  # Assurez-vous que cette fonction existe

app = FastAPI()

# Charger le modèle depuis le fichier .pkl lors du démarrage de l'application
with open('app/model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Route d'accueil
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API FastAPI"}

# Route pour faire une prédiction
@app.post("/predict")
def predict(data: dict):
    # Récupérer l'identifiant du client
    SK_ID_CURR = data.get('SK_ID_CURR')

    # Charger les données du client à partir de votre source de données (ex: fichier CSV ou base de données)
    # Exemple avec un DataFrame Pandas :
    # client_data = pd.read_csv('chemin/vers/vos/donnees.csv')
    # features = client_data[client_data['SK_ID_CURR'] == SK_ID_CURR]

    # Exemple: si vous avez un ensemble de caractéristiques déjà stockées, on peut faire :
    features = pd.DataFrame([data])

    # Appliquer le prétraitement
    processed_data = preprocess_data(features)

    # Faire la prédiction avec le modèle chargé
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[0][1]  # Probabilité de la classe positive

    # Renvoie de la prédiction et de la probabilité
    return {"prediction": prediction[0], "probability": probability}
