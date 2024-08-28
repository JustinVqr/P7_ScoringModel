import streamlit as st
import pandas as pd
import requests
from io import StringIO
import os
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configuration de la page d'accueil
st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title="Prêt à dépenser"
)

# --- Initialisation de l'état de session ---
if "load_state" not in st.session_state:
    st.session_state.load_state = False

# Mise en page de la page d'accueil
st.title("Prêt à dépenser")
st.subheader("Application d'aide à la décision de prêt")

st.write("""
Bienvenue dans l'application **Prêt à dépenser**. Cette application est conçue pour aider les agents de crédit à prendre des décisions éclairées concernant l'octroi de prêts. Elle utilise un modèle d'apprentissage automatique pour prédire la probabilité qu'un client puisse rencontrer des difficultés à rembourser son prêt.

L'application offre également des explications transparentes pour chaque décision prédictive, en utilisant des outils avancés comme SHAP (SHapley Additive exPlanations) pour rendre les modèles plus compréhensibles.
""")

# Disposition en colonnes pour le logo et la description de l'application
col1, col2 = st.columns([1, 2])  # Rapport 1:2 entre les colonnes

# Logo dans la première colonne
with col1:
    st.image("images/logo.png", width=150)  # Assurez-vous que le chemin de l'image est correct

# Description de l'application dans la deuxième colonne
with col2:
    st.subheader("Contenu de l'application")
    st.markdown("""
    Cette application comprend trois principales fonctionnalités :
    1. **Informations générales** : Données générales sur les clients existants et leur historique.
    2. **Analyse des clients connus** : Exploration approfondie des données pour des clients spécifiques.
    3. **Prédiction** : Prédiction des défauts de paiement pour de nouveaux clients via un modèle prédictif.
    """)

# --- Chargement des données et des ressources ---
st.subheader("Chargement des ressources")

# Afficher le chargement avec un spinner pour donner du feedback à l'utilisateur
with st.spinner("Chargement en cours..."):
    # Exemple de chargement des données
    @st.cache_data
    def load_data():
        # Exemple de lien vers des fichiers de données (à ajuster)
        url_train = "https://www.dropbox.com/scl/fi/9oc8a12r2pnanhzj2r6gu/df_train.csv?rlkey=zdcao9gluupqkd3ljxwnm1pv6&st=mm5480h6&dl=1"
        url_new = "https://www.dropbox.com/scl/fi/2mylh9bshf5jkzg6n9m7t/df_new.csv?rlkey=m82n87j6hr9en1utkt7a8qsv4&st=k6kj1pm5&dl=1"
        
        response_train = requests.get(url_train)
        response_new = requests.get(url_new)
        
        if response_train.status_code == 200 and response_new.status_code == 200:
            df_train = pd.read_csv(StringIO(response_train.text), sep=',', index_col="SK_ID_CURR", encoding='utf-8')
            df_new = pd.read_csv(StringIO(response_new.text), sep=',', index_col="SK_ID_CURR", encoding='utf-8')
            return df_train, df_new
        else:
            st.error("Erreur de téléchargement des données.")
            return None, None

    df_train, df_new = load_data()

    if df_train is not None and df_new is not None:
        st.success("Données chargées avec succès.")
    else:
        st.error("Erreur lors du chargement des données.")

# --- Navigation vers d'autres pages ---
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisissez une page", ["Accueil", "Analyse des clients", "Prédiction"])

if page == "Accueil":
    st.write("Vous êtes sur la page d'accueil.")
elif page == "Analyse des clients":
    st.write("Page d'analyse des clients à venir.")
elif page == "Prédiction":
    st.write("Page de prédiction à venir.")
