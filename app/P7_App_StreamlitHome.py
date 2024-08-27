import streamlit as st
import pandas as pd
import requests
from io import StringIO
import os
import joblib
import shap
import matplotlib.pyplot as plt

# Configuration de la page d'accueil
st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title="Accueil"
)

# Système de navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisissez une page", ["Accueil", "Analyse des clients", "Prédiction"])

# Fonction pour télécharger et charger df_train depuis l'URL
@st.cache_data
def load_df_train():
    url = "https://www.dropbox.com/scl/fi/59fn2h9mapw69flpnccz6/df_train.csv?rlkey=dq6qvlj4dxnswqdegyadjfnqs&st=snvt2aue&dl=1"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df_train = pd.read_csv(csv_data, sep=',', encoding='utf-8')
            return df_train
        else:
            st.error(f"Erreur de téléchargement : Statut {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données : {e}")
        return None

# Fonction pour télécharger et charger df_new à partir d'un client ID spécifique
@st.cache_data
def load_client_data(client_id):
    url = "https://www.dropbox.com/scl/fi/2mylh9bshf5jkzg6n9m7t/df_new.csv?rlkey=m82n87j6hr9en1utkt7a8qsv4&st=0zc92hpg&dl=1"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df_new = pd.read_csv(csv_data, sep=',', index_col="SK_ID_CURR", encoding='utf-8')
            # Sélectionner uniquement les données du client spécifique
            if client_id in df_new.index:
                st.write("Données `df_new` chargées avec succès.")  # Ajout du message de succès pour df_new
                return df_new.loc[[client_id]]
            else:
                st.error("Client ID non trouvé.")
                return None
        else:
            st.error(f"Erreur de téléchargement : Statut {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données : {e}")
        return None

# Initialisation de df_train dans session_state si non déjà présent
if 'df_train' not in st.session_state:
    st.session_state.df_train = load_df_train()

# Vérification du chargement de df_train
if st.session_state.df_train is not None:
    st.write("Données `df_train` chargées avec succès.")
else:
    st.error("Erreur lors du chargement des données `df_train`.")

# Page d'accueil
if page == "Accueil":
    st.title("Prêt à dépenser")
    st.subheader("Application de support à la décision pour l'octroi de prêts")
    st.write("""Cette application assiste le chargé de prêt dans sa décision d'octroyer un prêt à un client.""")

    col1, col2 = st.columns(2)

    with col1:
        st.image("https://raw.githubusercontent.com/JustinVqr/P7_ScoringModel/main/app/images/logo.png")

    with col2:
        st.subheader("Contenu de l'application :")
        st.markdown("""
        Cette application comporte trois pages :
        1) Informations générales sur la base de données et le modèle
        2) Analyse des clients connus
        3) Prédiction des défauts de paiement pour de nouveaux clients via une API
        """)

    st.subheader("Chargement de l'application :")

    # Saisie de l'ID du client
    sk_id_curr = st.text_input("Entrez l'ID du client pour obtenir la prédiction :")

    # Bouton pour lancer la prédiction
    if st.button("Obtenir la prédiction"):
        if sk_id_curr:
            try:
                client_id = int(sk_id_curr)
                df_client = load_client_data(client_id)

                if df_client is not None:
                    st.write("Données du client chargées avec succès.")
                    
                    # Charger le modèle
                    model_path = os.path.join(os.getcwd(), 'app', 'model', 'best_model.pkl')
                    if os.path.exists(model_path):
                        Credit_clf_final = joblib.load(model_path)
                        st.write("Modèle chargé avec succès.")

                        # Vérifiez si la colonne TARGET existe avant de la supprimer
                        if "TARGET" in df_client.columns:
                            X_client = df_client.drop(columns="TARGET").fillna(0)
                        else:
                            X_client = df_client.fillna(0)

                        # Faire la prédiction pour le client
                        prediction_proba = Credit_clf_final.predict_proba(X_client)[:, 1]
                        prediction = Credit_clf_final.predict(X_client)

                        # Afficher les résultats
                        st.write(f"Prédiction : {'Oui' if prediction[0] == 1 else 'Non'}")
                        st.write(f"Probabilité de défaut : {prediction_proba[0] * 100:.2f}%")

                        # Calculer et afficher les valeurs SHAP uniquement pour ce client
                        explainer = shap.KernelExplainer(Credit_clf_final.predict_proba, X_client)
                        shap_values = explainer.shap_values(X_client)
                        st.write("Valeurs SHAP calculées.")
                        shap.initjs()
                        shap.force_plot(explainer.expected_value[1], shap_values[1], X_client, matplotlib=True)
                        st.pyplot(bbox_inches='tight')

                    else:
                        st.error(f"Le fichier {model_path} n'existe pas.")

            except Exception as e:
                st.error(f"Erreur lors de la requête de prédiction : {e}")
        else:
            st.error("Veuillez entrer un ID client valide.")
