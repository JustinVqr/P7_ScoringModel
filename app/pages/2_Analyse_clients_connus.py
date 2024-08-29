import streamlit as st
import sys
import os

# Configuration de la page Streamlit
st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title="2) Clients info"
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

# Affichage de l'en-tête principal
st.header("Analyse du défaut de paiement des clients connus")

# Configuration de la barre latérale
st.sidebar.header('Tableau de bord')
st.sidebar.subheader('Sélection de l\'ID du client')

# Boîte de saisie pour l'ID du client
index_client = st.sidebar.number_input(
    "Entrer l'ID du client (ex : 100002)",
    format="%d",
    value=100002
)

# Bouton d'exécution
run_btn = st.sidebar.button('Voir les données du client')

# Action déclenchée par le bouton
if run_btn:
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
        st.sidebar.error("Client non présent dans la base de données")
