import streamlit as st
import sys
import os

# Configuration de la page Streamlit
st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title="2) Analyse clients connus"
)

# Ajoutez le chemin du répertoire racine au sys.path pour que Python trouve les modules dans 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.component.P7_App_FrontEnd import execute_noAPI, plot_client, nan_values

# Vérification que les données sont disponibles dans le session state
if 'df_train' not in st.session_state or 'Credit_clf_final' not in st.session_state or 'explainer' not in st.session_state:
    st.error("Les données nécessaires ne sont pas disponibles dans l'état de session. Veuillez charger les données sur la page d'accueil.")
    st.stop()

# Chargement des données depuis l'état de session
df_train = st.session_state.df_train
Credit_clf_final = st.session_state.Credit_clf_final
explainer = st.session_state.explainer

# Créer des onglets pour basculer entre différentes méthodes de saisie
tabs = st.tabs(["ID client", "Information manuelle"])

with tabs[0]:
    st.subheader("Prédiction pour un client avec ID")
    
    # Boîte de saisie pour l'ID du client
    index_client = st.number_input(
        "Entrez l'ID du client (ex : 1, 2, 3, 5)",
        format="%d",
        value=1
    )

    # Bouton de prédiction
    if st.button('Prédire', key="predire_id"):
        # Vérification de la présence de l'ID dans df_train
        if index_client in df_train.index:
            # Appel des fonctions de traitement du client
            try:
                execute_noAPI(df_train, index_client, Credit_clf_final)
                plot_client(
                    df_train.drop(columns='TARGET').fillna(0),  # Gestion des NaN
                    explainer,
                    df_reference=df_train,
                    index_client=index_client
                )
                nan_values(df_train.drop(columns='TARGET'), index_client=index_client)
            except Exception as e:
                st.error(f"Une erreur s'est produite lors de l'affichage des données du client : {e}")
        else:
            st.error("Client non présent dans la base de données")

with tabs[1]:
    st.subheader("Saisie manuelle d'informations")
    # Vous pouvez ajouter ici des champs pour saisir manuellement les informations nécessaires
    st.text("Formulaire de saisie manuelle en construction")
