from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
import pandas as pd
from app.model.P7_script_model_inf import make_prediction


app = FastAPI()

# Chemin vers le fichier CSV contenant les données des clients prétraitées
DATA_FILE = "data/preprocessed/preprocessed_data.csv"  # Chemin ajusté vers les données prétraitées
clients_data = pd.read_csv(DATA_FILE)  # Charger les données une fois au démarrage de l'application

@app.get("/", response_class=HTMLResponse)
def home():
    # Interface simple pour saisir un ID client
    html_content = """
    <html>
        <head>
            <title>Prédiction du score client</title>
            <script>
                async function fetchSuggestions() {
                    let query = document.getElementById('client_id').value;
                    if (query.length > 0) {
                        let response = await fetch(`/suggest_ids?q=${query}`);
                        let data = await response.json();
                        let suggestions = document.getElementById('suggestions');
                        suggestions.innerHTML = '';
                        data.forEach(id => {
                            let option = document.createElement('option');
                            option.value = id;
                            suggestions.appendChild(option);
                        });
                    }
                }
            </script>
        </head>
        <body>
            <h1>Prédiction du score client</h1>
            <form action="/predict/" method="post">
                <label for="client_id">ID Client :</label>
                <input type="text" id="client_id" name="client_id" list="suggestions" oninput="fetchSuggestions()" required>
                <datalist id="suggestions"></datalist>
                <button type="submit">Prédire</button>
            </form>
        </body>
    </html>
    """
    return html_content

@app.get("/suggest_ids", response_class=HTMLResponse)
def suggest_ids(q: str):
    # Filtrer les IDs clients qui correspondent à la requête
    matching_ids = clients_data[clients_data['SK_ID_CURR'].astype(str).str.startswith(q)]['SK_ID_CURR'].unique()
    return matching_ids.tolist()

@app.post("/predict/")
def predict(client_id: int = Form(...)):
    try:
        # Filtrer les données du client en fonction de l'ID
        client_data = clients_data[clients_data['SK_ID_CURR'] == client_id]

        # Vérifier si les données du client existent
        if client_data.empty:
            raise HTTPException(status_code=404, detail="ID client non trouvé dans les données")

        # Préparer les données pour la prédiction (supposons que 'SK_ID_CURR' n'est pas utilisé pour la prédiction)
        input_data = client_data.drop(columns=['SK_ID_CURR']).values.flatten().tolist()

        # Faire la prédiction
        prediction = make_prediction(input_data)

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
                <a href="/">Faire une autre prédiction</a>
            </body>
        </html>
        """

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
