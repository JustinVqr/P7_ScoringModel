import streamlit as st
import pandas as pd
import shap
import joblib
import os
import requests
from io import StringIO

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
        def loading_data():
            # Liens Dropbox pour les datasets
            train_url = "https://www.dropbox.com/scl/fi/59fn2h9mapw69flpnccz6/df_train.csv?rlkey=dq6qvlj4dxnswqdegyadjfnqs&st=d0n961rl&dl=1"
            test_url = "https://www.dropbox.com/scl/fi/5ihks3eng4kws0zfqwdkw/df_new.csv?rlkey=m2kd4iv4abb67w546p75ulrgk&st=dj744do8&dl=1"
            
            # Fonction pour télécharger et lire les fichiers CSV depuis Dropbox
            def download_and_load_csv(url):
                try:
                    response = requests.get(url)
                    
                    if response.status_code != 200:
                        st.error(f"Échec du téléchargement depuis {url}. Statut: {response.status_code}")
                        return None
                    
                    csv_data = StringIO(response.text)
                    return pd.read_csv(csv_data, sep=',', index_col="SK_ID_CURR", encoding='utf-8')
                except Exception as e:
                    st.error(f"Erreur lors du téléchargement ou du chargement des données depuis {url} : {e}")
                    return None

            # Chargement des datasets depuis Dropbox
            df_train = download_and_load_csv(train_url)
            df_new = download_and_load_csv(test_url)

            if df_train is not None and df_new is not None:
                st.write("Données chargées avec succès.")
                return df_train, df_new
            else:
                st.error("Erreur lors du chargement des données.")
                return None, None

        df_train, df_new = loading_data()

        if df_train is not None and df_new is not None:
            st.write("1) Chargement des données")

            st.write("2) Chargement du modèle")
            model_path = os.path.join(os.getcwd(), 'app', 'model', 'best_model.pkl')
            if os.path.exists(model_path):
                try:
                    Credit_clf_final = joblib.load(model_path)
                    st.write("Modèle chargé avec succès.")
                except Exception as e:
                    st.error(f"Erreur lors du chargement du modèle : {e}")
            else:
                st.error(f"Le fichier {model_path} n'existe pas.")

            st.write("3) Chargement de l'explainer (Shap)")
            try:
                explainer = shap.KernelExplainer(Credit_clf_final.predict_proba, df_train.drop(columns="TARGET").fillna(0))
                st.write("Explainer chargé avec succès.")
            except Exception as e:
                st.error(f"Erreur lors du chargement de l'explainer : {e}")

            st.write("4) Sauvegarde des variables de session")
            st.session_state.df_train = df_train
            st.session_state.df_new = df_new
            st.session_state.Credit_clf_final = Credit_clf_final
            st.session_state.explainer = explainer

            st.success('Chargement terminé !')
