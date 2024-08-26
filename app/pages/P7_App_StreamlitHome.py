import streamlit as st
import pandas as pd
import shap
import joblib
import requests
from io import BytesIO

# Configuration de la page d'accueil
st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title="Accueil"
)

# --- Système de navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisissez une page", ["Accueil", "Analyse des clients", "Prédiction"])

# --- Page d'accueil ---
if page == "Accueil":
    st.title("Prêt à dépenser")
    st.subheader("Application de support à la décision pour l'octroi de prêts")
    st.write("""Cette application assiste le chargé de prêt dans sa décision d'octroyer un prêt à un client.
         Pour ce faire, un algorithme d'apprentissage automatique est utilisé pour prédire les difficultés d'un client à rembourser le prêt.
         Par souci de transparence, cette application fournit également des informations pour expliquer l'algorithme et les prédictions.""")

    col1, col2 = st.columns(2)

    with col1:
        st.image("https://raw.githubusercontent.com/JustinVqr/P7_ScoringModel/main/app/images/logo.png")

    with col2:
        st.write(" ")
        st.write(" ")
        st.subheader("Contenu de l'application :")
        st.markdown("""
         Cette application comporte trois pages :
         1) Informations générales sur la base de données et le modèle
         2) Analyse des clients connus
         3) Prédiction des défauts de paiement pour de nouveaux clients via une API
         """)

    st.subheader("Chargement de l'application :")

    with st.spinner('Initialisation...'):
        @st.cache_data
        def load_model_from_github():
            # Lien brut vers le modèle hébergé sur GitHub
            model_url = "https://raw.githubusercontent.com/username/repository/branch/app/model/best_model.pkl"
            
            try:
                response = requests.get(model_url)
                
                if response.status_code != 200:
                    st.error(f"Erreur lors du téléchargement du modèle. Statut: {response.status_code}")
                    return None
                
                # Charger le modèle avec joblib
                model = joblib.load(BytesIO(response.content))
                return model
            
            except Exception as e:
                st.error(f"Erreur lors du chargement du modèle depuis GitHub: {e}")
                return None

        # Charger le modèle depuis GitHub
        Credit_clf_final = load_model_from_github()

        if Credit_clf_final is not None:
            st.write("Modèle chargé avec succès depuis GitHub.")

            # Charger les datasets et l'explainer (comme dans votre code précédent)
            df_train = ...  # Téléchargez vos données ici comme dans votre code original
            explainer = shap.TreeExplainer(Credit_clf_final, df_train.drop(columns="TARGET").fillna(0))

            st.session_state.Credit_clf_final = Credit_clf_final
            st.session_state.explainer = explainer

            st.success('Chargement terminé !')
