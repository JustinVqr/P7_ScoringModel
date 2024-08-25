import streamlit as st
import pandas as pd
import shap
import pickle

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

    with st.spinner('initialisation...'):
        @st.cache
        def loading_data():
            # Lien Dropbox pour les fichiers
            train_url = "https://www.dropbox.com/scl/fi/iqw80pyid2z79f40n0m8p/preprocessed_data.csv?rlkey=mvd2bz9s1hkaxdg51giv5thvm&st=nnqiv93s&dl=1"
            test_url = "https://www.dropbox.com/s/r1p43l7ad230zjg/df_test.csv.zip?dl=1"
            
            # Chargement des DataFrames depuis Dropbox
            df_train = pd.read_csv(train_url, sep=',', index_col="SK_ID_CURR")
            df_new = pd.read_csv(test_url, compression="zip", sep=';', index_col="SK_ID_CURR")
            
            return df_train, df_new

        df_train, df_new = loading_data()

        st.write("1) Chargement des données")
        st.write("2) Chargement du modèle")
        
        # Lien Dropbox pour le modèle (veuillez ajouter votre modèle sur Dropbox et utiliser son lien)
        model_url = "https://www.dropbox.com/s/your_dropbox_model_path.pkl?dl=1"
        Credit_clf_final = pickle.load(open(model_url, 'rb'))

        st.write("3) Chargement de l'explainer (Shap)")
        explainer = shap.TreeExplainer(Credit_clf_final, df_train.drop(columns="TARGET").fillna(0))

        st.write("4) Sauvegarde des variables de session")
        st.session_state.df_train = df_train
        st.session_state.df_new = df_new
        st.session_state.Credit_clf_final = Credit_clf_final
        st.session_state.explainer = explainer

        st.success('Chargement terminé !')

# --- Page "Analyse des clients" ---
elif page == "Analyse des clients":
    st.title("Analyse des clients")
    # Ajoutez ici votre code pour analyser les données des clients connus.
    # Vous pouvez afficher des tableaux, des graphiques, etc.

# --- Page "Prédiction" ---
elif page == "Prédiction":
    st.title("Prédiction pour de nouveaux clients")
    # Ajoutez ici votre code pour prédire les défauts de paiement pour de nouveaux clients via l'API
