from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import os
import gc
from scripts.P7_data_preprocessing_fct import preprocess_data, data_prep

# Initialisation de l'application FastAPI
app = FastAPI()

# Chemin d'accès au dossier contenant les fichiers de données brutes
DATA_FOLDER = r"C:\Users\justi\OneDrive\Cours - Travail\DATA SCIENCE\Formation - DataScientist\Projet n°7\P7_Model_Scoring\data\raw"

# Nom du fichier où seront stockées les données prétraitées
PROCESSED_DATA_FILE = 'processed_data.csv'  # Fichier CSV qui contient les données prétraitées

# Charger le modèle entraîné lors du démarrage de l'application
with open('app/model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)  # Chargement du modèle LightGBM stocké dans un fichier .pkl

# Prétraitement des données lors du démarrage de l'application
@app.on_event("startup")
def startup_event():
    # Vérifier si les données prétraitées existent déjà
    if not os.path.exists(PROCESSED_DATA_FILE):
        print("Prétraitement des données...")

        # Appel à la fonction de préparation des données (qui traite plusieurs fichiers CSV)
        df_final = data_prep(path=DATA_FOLDER, debug=False)

        # Sauvegarde des données prétraitées dans un fichier CSV
        df_final.to_csv(PROCESSED_DATA_FILE, index_label='SK_ID_CURR', sep=";")
        print("Données prétraitées et stockées.")
    else:
        print("Les données prétraitées sont déjà disponibles.")  # Si les données sont déjà disponibles, ne rien faire

# Route d'accueil de l'API pour vérifier que l'API fonctionne
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API FastAPI"}

# Route de prédiction qui utilise le modèle chargé et les données prétraitées
@app.post("/predict")
def predict(data: dict):
    # Extraction de l'identifiant client (SK_ID_CURR) à partir de la requête envoyée
    SK_ID_CURR = data.get('SK_ID_CURR')

    # Vérification que l'identifiant client a bien été fourni
    if not SK_ID_CURR:
        raise HTTPException(status_code=400, detail="L'identifiant SK_ID_CURR est requis.")

    # Vérification de l'existence du fichier contenant les données prétraitées
    if not os.path.exists(PROCESSED_DATA_FILE):
        raise HTTPException(status_code=500, detail="Les données prétraitées ne sont pas disponibles.")

    # Chargement du fichier CSV contenant les données prétraitées
    client_data = pd.read_csv(PROCESSED_DATA_FILE, sep=";", index_col='SK_ID_CURR')

    # Sélection des données correspondant à l'identifiant client
    features = client_data.loc[[SK_ID_CURR]]

    # Vérification que les données du client existent bien dans le fichier prétraité
    if features.empty:
        raise HTTPException(status_code=404, detail=f"Les données pour le client avec SK_ID_CURR = {SK_ID_CURR} n'ont pas été trouvées.")

    # Appliquer des étapes de prétraitement si nécessaire (en utilisant une fonction dédiée)
    processed_data = preprocess_data(features)

    # Utilisation du modèle chargé pour effectuer une prédiction sur les données prétraitées
    prediction = model.predict(processed_data)  # Renvoie la prédiction
    probability = model.predict_proba(processed_data)[0][1]  # Probabilité associée à la classe positive

    # Renvoi de la prédiction et de la probabilité sous forme de réponse JSON
    return {"prediction": prediction[0], "probability": probability}
