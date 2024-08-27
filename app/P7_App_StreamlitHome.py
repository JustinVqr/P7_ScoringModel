import streamlit as st
import pandas as pd
import requests
from io import StringIO
import os
import pickle
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

# --- Initialisation de l'état de session ---
if "load_state" not in st.session_state:
    st.session_state.load_state = False

# --- Fonction pour télécharger et charger les données depuis Dropbox ---
@st.cache_data
def load_data():
    url_train = "https://www.dropbox.com/scl/fi/59fn2h9mapw69flpnccz6/df_train.csv?rlkey=dq6qvlj4dxnswqdegyadjfnqs&st=snvt2aue&dl=1"
    url_new = "https://www.dropbox.com/scl/fi/2mylh9bshf5jkzg6n9m7t/df_new.csv?rlkey=m82n87j6hr9en1utkt7a8qsv4&st=k6kj1pm5&dl=1"
    
    response_train = requests.get(url_train)
    response_new = requests.get(url_new)
    
    if response_train.status_code == 200 and response_new.status_code == 200:
        df_train = pd.read_csv(StringIO(response_train.text), sep=',', encoding='utf-8')
        df_new = pd.read_csv(StringIO(response_new.text), sep=',', index_col="SK_ID_CURR", encoding='utf-8')
        return df_train, df_new
    else:
        st.error(f"Erreur de téléchargement : Statut {response_train.status_code}, {response_new.status_code}")
        return None, None


# --- Chargement des ressources au démarrage ---
@st.cache_resource
def load_model_and_explainer(df_train):
    # Charger le modèle
    model_path = os.path.join(os.getcwd(), 'app', 'model', 'best_model.pkl')
    if os.path.exists(model_path):
        Credit_clf_final = pickle.load(open(model_path, 'rb'))
    else:
        st.error(f"Le fichier {model_path} n'existe pas.")
        return None, None
    
    # Créer l'explicateur SHAP optimisé pour les modèles basés sur les arbres
    explainer = shap.TreeExplainer(Credit_clf_final, df_train.drop(columns="TARGET").fillna(0))
    return Credit_clf_final, explainer


# Chargement initial des données et du modèle si l'état n'est pas déjà chargé
if not st.session_state.load_state:
    with st.spinner('Chargement des données et du modèle...'):
        df_train, df_new = load_data()
        if df_train is not None and df_new is not None:
            Credit_clf_final, explainer = load_model_and_explainer(df_train)
            if Credit_clf_final and explainer:
                st.session_state.df_train = df_train
                st.session_state.df_new = df_new
                st.session_state.Credit_clf_final = Credit_clf_final
                st.session_state.explainer = explainer
                st.session_state.load_state = True
                st.success('Chargement terminé !')
        else:
            st.error("Erreur lors du chargement des données.")
else:
    df_train = st.session_state.df_train
    df_new = st.session_state.df_new
    Credit_clf_final = st.session_state.Credit_clf_final
    explainer = st.session_state.explainer

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
                if client_id in df_new.index:
                    df_client = df_new.loc[[client_id]]
                    st.write("Données du client chargées avec succès.")
                    
                    # Préparation des données du client
                    X_client = df_client.fillna(0)

                    # Faire la prédiction pour le client
                    prediction_proba = Credit_clf_final.predict_proba(X_client)[:, 1]
                    prediction = Credit_clf_final.predict(X_client)

                    # Afficher les résultats
                    st.write(f"Prédiction : {'Oui' if prediction[0] == 1 else 'Non'}")
                    st.write(f"Probabilité de défaut : {prediction_proba[0] * 100:.2f}%")

                    # Calculer et afficher les valeurs SHAP pour ce client
                    shap_values = explainer.shap_values(X_client)
                    st.write("Valeurs SHAP calculées.")
                    shap.initjs()
                    shap.force_plot(explainer.expected_value[1], shap_values[1], X_client, matplotlib=True)
                    st.pyplot(bbox_inches='tight')
                else:
                    st.error("Client ID non trouvé.")
            except Exception as e:
                st.error(f"Erreur lors de la requête de prédiction : {e}")
        else:
            st.error("Veuillez entrer un ID client valide.")