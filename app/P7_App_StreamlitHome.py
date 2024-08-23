import streamlit as st
import pandas as pd
import shap
import pickle

# Configuration de la page d'accueil
st.set_page_config(
    layout='wide',  # Disposition large de la page
    initial_sidebar_state='expanded',  # La barre latérale est étendue par défaut
    page_title="Accueil")  # Titre de la page

# --- Initialisation de l'état de session ---
if "load_state" not in st.session_state:  # Vérifie si l'état de session est déjà initialisé
    st.session_state.load_state = False  # Initialise l'état de session à False

# --- Mise en page de la page d'accueil ---

st.title("Prêt à dépenser")  # Titre principal de l'application
st.subheader("Application de support à la décision pour l'octroi de prêts")  # Sous-titre de l'application

# Description de l'application
st.write("""Cette application assiste le chargé de prêt dans sa décision d'octroyer un prêt à un client.
     Pour ce faire, un algorithme d'apprentissage automatique est utilisé pour prédire les difficultés d'un client à rembourser le prêt.
     Par souci de transparence, cette application fournit également des informations pour expliquer l'algorithme et les prédictions.""")

# Création de deux colonnes
col1, col2 = st.columns(2)

# --- Logo ---
with col1:  # Affichage du logo dans la première colonne
    st.image("https://raw.githubusercontent.com/JustinVqr/Projet7_Scoring_OC/App_Scoring/app_scoring/frontend_App_P7/images/logo.png")

# --- Description des pages ---
with col2:  # Contenu texte dans la deuxième colonne

    st.write(" ")
    st.write(" ")  # Espaces vides pour centrer le texte
    st.write(" ")

    st.subheader("Contenu de l'application :")  # Sous-titre pour la description du contenu
    st.markdown("""
     Cette application comporte trois pages :
     1) Informations générales sur la base de données et le modèle
     2) Analyse des clients connus
     3) Prédiction des défauts de paiement pour de nouveaux clients via une API
     """)

# --- Chargement des données ---

st.subheader("Chargement de l'application :")  # Sous-titre pour indiquer la progression du chargement

# Affichage d'un spinner pendant le chargement
with st.spinner('initialisation...'):  # Affiche un indicateur de chargement

    @st.cache  # Mise en cache pour améliorer les performances et éviter de recharger les données inutilement
    def loading_data():
        # Chemins locaux vers les fichiers CSV compressés
        chemin_train = r"C:\Users\justi\OneDrive\Cours - Travail\DATA SCIENCE\Formation - DataScientist\Projet n°7\Projet n°7_Scoring\df_train.csv.zip"
        df_train = pd.read_csv(chemin_train,
            compression="zip",  # Les données sont compressées en ZIP
            sep=';',  # Séparateur utilisé dans le fichier CSV
            index_col="SK_ID_CURR")  # La colonne des ID clients est utilisée comme index

        chemin_test = r"C:\Users\justi\OneDrive\Cours - Travail\DATA SCIENCE\Formation - DataScientist\Projet n°7\Projet n°7_Scoring\df_test.csv.zip"
        df_new = pd.read_csv(chemin_test,
            compression="zip",
            sep=';',
            index_col="SK_ID_CURR")
            
        return df_train, df_new  # Retourne les deux DataFrames chargés

    # Indique la progression du chargement
    st.write("1) Chargement des données")
    df_train, df_new = loading_data()  # Appelle la fonction pour charger les données

    st.write("2) Chargement du modèle")
    model_path = r"C:\Users\justi\OneDrive\Cours - Travail\DATA SCIENCE\Formation - DataScientist\Projet n°7\Projet n°7_Scoring\best_model.pkl"  # Chemin vers le modèle
    Credit_clf_final = pickle.load(open(model_path, 'rb'))  # Charge le modèle avec pickle

    st.write("3) Chargement de l'explainer (Shap)")
    explainer = shap.TreeExplainer(
        Credit_clf_final, df_train.drop(
            columns="TARGET").fillna(0))  # Création de l'explainer SHAP pour interpréter les prédictions

    st.write("4) Sauvegarde des variables de session")
    # Stocke les variables dans l'état de session pour les utiliser plus tard
    st.session_state.df_train = df_train
    st.session_state.df_new = df_new
    st.session_state.Credit_clf_final = Credit_clf_final
    st.session_state.explainer = explainer

    st.success('Chargement terminé !')  # Message de succès lorsque tout est chargé
