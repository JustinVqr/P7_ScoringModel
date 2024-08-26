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
        st.image("https://raw.githubusercontent.com/JustinVqr/P7_ScoringModel/main/app/images/logo.png")

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
        @st.cache_data
        def loading_data():
            # Liens Dropbox pour les datasets
            train_url = "https://www.dropbox.com/scl/fi/1utyp2vxauxizvs96dbnv/df_train.csv?rlkey=5y0srv1wu9muzoa2yezu472kn&st=tosffzn8&dl=1"
            test_url = "https://www.dropbox.com/scl/fi/5ihks3eng4kws0zfqwdkw/df_new.csv?rlkey=m2kd4iv4abb67w546p75ulrgk&st=dj744do8&dl=1"
            
            # Fonction pour télécharger et lire les fichiers CSV depuis Dropbox
            def download_and_load_csv(url):
                try:
                    response = requests.get(url)
                    
                    # Vérification manuelle du code de statut HTTP
                    if response.status_code != 200:
                        st.error(f"Échec du téléchargement depuis {url}. Statut: {response.status_code}")
                        return None
                    
                    # Chargement des données en DataFrame
                    csv_data = StringIO(response.text)
                    return pd.read_csv(csv_data, sep=',', index_col="SK_ID_CURR", encoding='utf-8')
                except Exception as e:
                    st.error(f"Erreur lors du téléchargement ou du chargement des données depuis {url} : {e}")
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
