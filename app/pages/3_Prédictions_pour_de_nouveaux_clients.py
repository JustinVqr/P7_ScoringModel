import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import shap
import json

# Configuration de la page Streamlit
st.set_page_config(page_title="3) Prédiction pour de nouveaux clients")

# Ajoutez le chemin du répertoire racine au sys.path pour que Python trouve les modules dans 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.component.P7_App_FrontEnd import execute_API, plot_client, shap_plot, nan_values, plot_gauge

# --- Récupération des données depuis la page d'accueil ---
df_train = st.session_state.df_train  # DataFrame principal contenant les données d'entraînement
df_new = st.session_state.df_new  # DataFrame contenant de nouvelles données pour la prédiction
Credit_clf_final = st.session_state.Credit_clf_final  # Modèle final de classification
explainer = st.session_state.explainer  # Explicateur du modèle pour l'interprétation des résultats

# --- Création de deux onglets ---
tab1, tab2 = st.tabs(["ID client", "Information manuelle"])

# --- Onglet 1 : Prédiction pour un client avec un ID ---
with tab1:
    st.header("Prédiction pour un client avec ID")
    
    index_client = st.number_input(
        "Entrez l'ID du client (ex : 1, 2, 3, 5)",
        format="%d",
        value=1
    )

    run_btn = st.button(
        'Prédire',
        on_click=None,
        type="primary",
        key='predict_btn1'
    )
    
    if run_btn or "updated_client" in st.session_state:
        if index_client in set(df_new.index):
            if run_btn:
                # Stocker les données originales du client
                data_client = df_new.loc[index_client].fillna(0).to_dict()
                st.session_state['updated_client'] = data_client
            else:
                data_client = st.session_state['updated_client']
            
            # Préparation des données pour SHAP et autres analyses
            X_client = pd.DataFrame([data_client]).fillna(0)

            # Affichage de la jauge avant les graphiques SHAP
            pred_prob = Credit_clf_final.predict_proba(X_client)[0][1]
            plot_gauge(pred_prob)

            # --- Volet réductible pour voir et modifier les caractéristiques du client ---
            with st.expander("Cliquez pour afficher et modifier les caractéristiques du client"):
                for feature, value in data_client.items():
                    if isinstance(value, (int, float)):
                        data_client[feature] = st.number_input(
                            label=feature,
                            value=float(value),
                            key=f"input_{feature}"
                        )
                    else:
                        data_client[feature] = st.text_input(
                            label=feature,
                            value=str(value),
                            key=f"input_{feature}"
                        )

                submit_changes = st.button("Mettre à jour les caractéristiques et prédire à nouveau")
            
            if submit_changes:
                # Mettre à jour les nouvelles données dans session_state
                st.session_state['updated_client'] = data_client

                # Relancer la prédiction avec les nouvelles valeurs
                updated_client = pd.DataFrame([data_client])
                pred_prob_updated = Credit_clf_final.predict_proba(updated_client)[0][1]
                plot_gauge(pred_prob_updated)

                # Afficher les graphiques SHAP avec les nouvelles valeurs
                shap_plot(explainer, updated_client, 0)

                # Autres visualisations
                plot_client(
                    updated_client,
                    explainer,
                    df_reference=df_train,
                    index_client=0  # Utilisation d'un index fictif pour un client modifié
                )
                nan_values(updated_client, index_client=0)
            else:
                # Afficher les graphiques SHAP avec les valeurs originales
                shap_plot(explainer, df_new, index_client)

                # Autres visualisations
                plot_client(
                    df_new,
                    explainer,
                    df_reference=df_train,
                    index_client=index_client
                )
                nan_values(df_new, index_client=index_client)
        else:
            st.write("Client non trouvé dans la base de données")

# --- Onglet 2 : Prédiction pour un nouveau client sans ID ---
with tab2:
    st.header("Prédiction pour un nouveau client")

    option = st.selectbox(
        'Comment souhaitez-vous entrer les données ?',
        ('Manuel', 'Texte', 'Fichier CSV')
    )

    if option == 'Manuel':
        with st.expander("Cliquez pour entrer les données manuellement"):
            data_client = {}
            for feature in list(df_new.columns):
                if df_train[feature].dtype == np.int64:
                    min_values = df_train[feature].min().astype('int')
                    max_values = df_train[feature].max().astype('int')
                    data_client[feature] = st.number_input(
                        str(feature), min_value=min_values, max_value=max_values, step=1)
                else:
                    min_values = df_train[feature].min().astype('float')
                    max_values = df_train[feature].max().astype('float')
                    data_client[feature] = st.number_input(
                        str(feature), min_value=min_values, max_value=max_values, step=0.1)

    elif option == 'Texte':
        with st.expander("Cliquez pour entrer les données sous forme de texte"):
            data_client = st.text_area('Entrez les données sous forme de dictionnaire', '''{"FLAG_OWN_CAR": 0, ... }''')
            data_client = json.loads(data_client)

    else:
        loader = st.file_uploader(" ")
        if loader is not None:
            data_client = pd.read_csv(loader, sep=";", index_col=0, header=None).squeeze(1).to_dict()

    run_btn2 = st.button(
        'Prédire',
        on_click=None,
        type="primary",
        key='predict_btn2'
    )
    
    if run_btn2:
        execute_API(data_client)
        data_client = pd.DataFrame(data_client, index=[0])

        # Utilisation de la fonction personnalisée pour la visualisation SHAP pour un nouveau client
        shap_plot(explainer, df_new, 0)

        # Autres visualisations et fonctionnalités
        plot_client(
            data_client,
            explainer,
            df_reference=df_train,
            index_client=0  # Utilisation d'un index fictif (0) pour un nouveau client
        )

        nan_values(data_client, index_client=0)
