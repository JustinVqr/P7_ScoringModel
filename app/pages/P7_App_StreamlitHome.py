import streamlit as st
import pandas as pd
import shap
import pickle
import os
import requests
from io import StringIO

# Configuration de la page d'accueil
st.set_page_config(
    layout='wide',  # Disposition large de la page
    initial_sidebar_state='expanded',  # La barre latérale est étendue par défaut
    page_title="Accueil"  # Titre de la page
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
        st.image("https://raw.githubusercontent.com/JustinVqr/Projet7_Scoring_OC/App_Scoring/app_scoring/frontend_App_P7/images/logo.png")

    with col2:
        st.write(" ")
        st.write(" ")  # Espaces vides pour centrer le texte
        st.subheader("Contenu de l'application :")
        st.markdown("""
         Cette application comporte trois pages :
         1) Informations générales sur la base de données et le modèle
         2) Analyse des clients connus
         3) Prédiction des défauts de paiement pour de nouveaux clients via une API
         """)

    st.subheader("Chargement de l'application :")

    with st.spinner('Initialisation...'):
        @st.cache
        def loading_data():
            # Liens Dropbox pour les datasets
            train_url = "https://www.dropbox.com/scl/fi/6ya6hk9y2ds3bpa3ygvze/df_train.csv?rlkey=izkzxuf5k5l55el4240dltzee&st=uc651jru&dl=1"
            test_url = "https://www.dropbox.com/scl/fi/bz27bvds2bbi89tkbee2f/df_new.csv?rlkey=p3dvtbh4dupugqq44f07ds7s0&st=e7rsatfb&dl=1"
            
            # Fonction pour télécharger et lire les fichiers CSV depuis Dropbox
            def download_and_load_csv(url):
                try:
                    response = requests.get(url)
                    response.raise_for_status()  # Vérifie s'il y a des erreurs de téléchargement
                    st.write(f"Téléchargement réussi: {url}")
                    st.write(f"Statut de la réponse: {response.status_code}")
                    st.write(f"Premier caractère de la réponse: {response.text[:200]}")  # Affiche un aperçu du contenu
                    
                    csv_data = StringIO(response.text)
                    return pd.read_csv(csv_data, sep=';', index_col="SK_ID_CURR", encoding='utf-8')
                except requests.exceptions.RequestException as e:
                    st.error(f"Erreur de téléchargement des données depuis {url} : {e}")
                    return None

            # Chargement des datasets depuis Dropbox
            df_train = download_and_load_csv(train_url)
            df_new = download_and_load_csv(test_url)
            
            # Vérifiez si les DataFrames sont correctement chargés
            if df_train is None or df_new is None:
                st.error("Erreur lors du chargement des données.")
                return None, None
            
            return df_train, df_new

        df_train, df_new = loading_data()

        if df_train is not None and df_new is not None:
            st.write("1) Chargement des données")

            st.write("2) Chargement du modèle")
            # Chargement du modèle depuis le répertoire local
            model_path = os.path.join(os.getcwd(), 'app', 'model', 'best_model.pkl')
            with open(model_path, 'rb') as model_file:
                Credit_clf_final = pickle.load(model_file)

            st.write("3) Chargement de l'explainer (Shap)")
            explainer = shap.TreeExplainer(Credit_clf_final, df_train.drop(columns="TARGET").fillna(0))

            st.write("4) Sauvegarde des variables de session")
            st.session_state.df_train = df_train
            st.session_state.df_new = df_new
            st.session_state.Credit_clf_final = Credit_clf_final
            st.session_state.explainer = explainer

            st.success('Chargement terminé !')
