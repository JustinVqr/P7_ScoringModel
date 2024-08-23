import streamlit as st
import requests

# URL de votre API FastAPI (que vous hébergerez également sur Heroku)
API_URL = "https://votre-app.herokuapp.com/predict"

st.title("Prédiction de Défaut de Paiement")
st.write("Entrez l'identifiant du client pour obtenir une prédiction.")

# Entrée de l'identifiant client
client_id = st.number_input("Identifiant Client", min_value=100001, max_value=999999, step=1)

# Lorsqu'un utilisateur soumet l'identifiant, envoyez une requête à l'API
if st.button("Obtenir la Prédiction"):
    # Requête POST à l'API avec l'identifiant du client
    response = requests.post(API_URL, json={"SK_ID_CURR": int(client_id)})

    # Vérifier si la requête a réussi
    if response.status_code == 200:
        # Récupération de la prédiction et affichage
        prediction_data = response.json()
        st.write(f"Prédiction : {prediction_data['prediction']}")
        st.write(f"Probabilité : {prediction_data['probability']}")
    else:
        # Afficher un message d'erreur si l'API ne renvoie pas de succès
        st.write("Erreur : Impossible de récupérer les données pour cet identifiant.")
