import streamlit as st
from app.component.P7_App_FrontEnd import execute_noAPI, plot_client, nan_values

# Configuration de la page Streamlit
st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title="2) Clients info"
)

# Vérifiez que les données sont disponibles dans le session state
if 'df_train' in st.session_state and 'Credit_clf_final' in st.session_state and 'explainer' in st.session_state:
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

    if run_btn:
        # Vérification de la présence de l'ID dans df_train
        if index_client in df_train.index:
            execute_noAPI(df_train, index_client, Credit_clf_final)
            plot_client(
                df_train.drop(columns='TARGET').fillna(0),  # Gestion des NaN
                explainer,
                df_reference=df_train,
                index_client=index_client
            )
            nan_values(df_train.drop(columns='TARGET'), index_client=index_client)
        else:
            st.sidebar.error("Client non présent dans la base de données")
else:
    st.error("Les données nécessaires ne sont pas disponibles dans l'état de session.")
