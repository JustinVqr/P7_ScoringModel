from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
import pandas as pd
from scripts.P7_data_preprocessing_fct import preprocessing_pipeline  # Importer la fonction de prétraitement
from app.model import best_model  # Charger le modèle
from fastapi.responses import HTMLResponse

app = FastAPI()

# Chemin vers le fichier CSV contenant les données des clients
DATA_FILE = "data/preprocessed/preprocessed_data.csv"  # Chemin ajusté vers les données prétraitées

# Modèle de données pour l'API
class ClientID(BaseModel):
    SK_ID_CURR: int

@app.get("/", response_class=HTMLResponse)
def home():
    # Interface simple pour saisir un ID client
    html_content = """
    <html>
        <head>
            <title>Prédiction du score client</title>
        </head>
        <body>
            <h1>Prédiction du score client</h1>
            <form action="/predict/" method="post">
                <label for="client_id">ID Client :</label>
                <input type="number" id="client_id" name="client_id" required>
                <button type="submit">Prédire</button>
            </form>
        </body>
    </html>
    """
    return html_content

@app.post("/predict/")
def predict(client_id: int = Form(...)):
    try:
        # Lire les données du fichier CSV prétraité
        clients_data = pd.read_csv(DATA_FILE)

        # Filtrer les données du client en fonction de l'ID
        client_data = clients_data[clients_data['SK_ID_CURR'] == client_id]

        # Vérifier si les données du client existent
        if client_data.empty:
            raise HTTPException(status_code=404, detail="ID client non trouvé dans les données")

        # Appliquer le prétraitement aux données du client
        processed_data = preprocessing_pipeline(client_data)

        # Faire la prédiction
        prediction = best_model.predict(processed_data)
        probability = best_model.predict_proba(processed_data)[0][1]

        # Retourner les résultats sous forme HTML
        return f"""
        <html>
            <head>
                <title>Résultat de la Prédiction</title>
            </head>
            <body>
                <h1>Résultat de la Prédiction</h1>
                <p>ID Client : {client_id}</p>
                <p>Prédiction : {int(prediction[0])}</p>
                <p>Probabilité : {float(probability)}</p>
                <a href="/">Faire une autre prédiction</a>
            </body>
        </html>
        """

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
