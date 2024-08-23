import streamlit as st
from P7_App_FrontEnd import *  # Importation des fonctions depuis le module P7_App_FrontEnd

# Configuration de la page Streamlit : 
# 1) Layout en largeur totale
# 2) État initial de la barre latérale étendu
# 3) Titre de la page
st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title="2) Clients info"
)

# Désactivation des avertissements de dépréciation pour pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

# 
# --- Récupération des données de la page d'accueil via le session state ---
#

# Récupération du DataFrame d'entraînement (df_train), du modèle de classification final (Credit_clf_final)
# et de l'explicateur de modèle (explainer) depuis l'état de session.
df_train = st.session_state.df_train
Credit_clf_final = st.session_state.Credit_clf_final
explainer = st.session_state.explainer

#
# --- Préparation de la mise en page de la page ---
#

# Affichage d'un en-tête principal pour la page
st.header("Analyse du défaut de paiement des clients connus")

# Création de la barre latérale avec un en-tête et un sous-en-tête
st.sidebar.header('Tableau de bord')
st.sidebar.subheader('Sélection de l\'ID du client')

#
# --- Analyse du client (avec ID client, sans API) ---
#

# Boîte de saisie pour l'utilisateur afin de rentrer l'ID du client à analyser.
# Le nombre doit être un entier et 100002 est la valeur par défaut.
index_client = st.sidebar.number_input(
    "Entrer l'ID du client (ex : 100002)",
    format="%d",
    value=100002
)

# Bouton dans la barre latérale pour exécuter l'analyse des données du client.
run_btn = st.sidebar.button('Voir les données du client', on_click=None, type="primary")

# Si l'utilisateur clique sur le bouton, l'analyse du client démarre
if run_btn:
    # Vérifie si l'ID client existe dans les index du DataFrame df_train
    if index_client in set(df_train.index):
        
        # Exécution de l'analyse sans utiliser l'API pour l'ID client sélectionné
        execute_noAPI(df_train, index_client, Credit_clf_final)
        
        # Génération et affichage des graphiques relatifs au client sélectionné
        plot_client(
            df_train.drop(columns='TARGET').fillna(0),  # Suppression de la colonne TARGET et remplacement des NaN par 0
            explainer,
            df_reference=df_train,  # Référence du DataFrame pour comparaison
            index_client=index_client  # ID du client sélectionné
        )
        
        # Affichage des valeurs manquantes pour le client sélectionné
        nan_values(df_train.drop(columns='TARGET'), index_client=index_client)
    
    # Si l'ID du client n'existe pas dans la base de données, afficher un message d'erreur dans la barre latérale
    else:
        st.sidebar.write("Client non présent dans la base de données")
