import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import shap
import json

# Configuration de la page Streamlit
st.set_page_config(page_title="3) Nouvelle Prédiction")

# Ajoutez le chemin du répertoire racine au sys.path pour que Python trouve les modules dans 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.component.P7_App_FrontEnd import execute_API, plot_client, nan_values

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
        "Entrez l'ID du client (ex : 100001, 100005)",
        format="%d",
        value=100001
    )

    run_btn = st.button(
        'Prédire',
        on_click=None,
        type="primary",
        key='predict_btn1'
    )
    
    if run_btn:
        if index_client in set(df_new.index):
            data_client = df_new.loc[index_client].fillna(0).to_dict()
            execute_API(data_client)
            
            # Calculer les valeurs SHAP pour ce client
            X_client = df_new.loc[[index_client]].fillna(0)

            # Vérifiez que les données sont au bon format (DataFrame)
            if not isinstance(X_client, pd.DataFrame):
                X_client = pd.DataFrame(X_client)

            # Vérification des colonnes attendues par le modèle
            try:
                if hasattr(Credit_clf_final, 'booster_'):  # Vérifiez si le modèle a un booster
                    booster = Credit_clf_final.booster_
                    expected_columns = booster.feature_name()
                else:
                    # Cas où le modèle est directement un booster
                    expected_columns = Credit_clf_final.feature_name()
            except AttributeError as e:
                st.write(f"Erreur lors de la récupération des colonnes du modèle : {e}")

            # Comparaison des colonnes attendues avec celles des données actuelles
            if set(expected_columns) != set(X_client.columns):
                st.write("Erreur : Les colonnes ne correspondent pas aux colonnes attendues par le modèle.")
            
            # Assurez-vous que toutes les valeurs manquantes sont remplies
            X_client = X_client.fillna(0)

            # Calcul des valeurs SHAP
            shap_values_client = explainer.shap_values(X_client)

            # Si shap_values_client est une liste, prendre les valeurs pour la classe 1
            if isinstance(shap_values_client, list):
                shap_values_client = shap_values_client[1]

            # Affichage des résultats SHAP avec un waterfall plot
            st.write("Valeurs SHAP pour ce client :")
            shap.initjs()

            # Waterfall plot
            st.pyplot(shap.waterfall_plot(shap.Explanation(
                values=shap_values_client[0],
                base_values=explainer.expected_value,
                data=X_client.iloc[0, :],
                feature_names=X_client.columns
            )))

            # Force plot
            st.pyplot(shap.force_plot(
                explainer.expected_value,
                shap_values_client[0],
                X_client,
                matplotlib=True
            ))

            # Autres visualisations (ex: plot_client)
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
            for features in list(df_new.columns):
                if df_train[features].dtype == np.int64:
                    min_values = df_train[features].min().astype('int')
                    max_values = df_train[features].max().astype('int')
                    data_client[features] = st.number_input(
                        str(features), min_value=min_values, max_value=max_values, step=1)
                else:
                    min_values = df_train[features].min().astype('float')
                    max_values = df_train[features].max().astype('float')
                    data_client[features] = st.number_input(
                        str(features), min_value=min_values, max_value=max_values, step=0.1)

    elif option == 'Texte':
        with st.expander("Cliquez pour entrer les données sous forme de texte"):
            data_client = st.text_area('Entrez les données sous forme de dictionnaire',
                                       '''{"FLAG_OWN_CAR": 0,
                "AMT_CREDIT": 0,
                ...,
                "INSTAL_DAYS_ENTRY_PAYMENT_MEAN": 0,
                "CC_CNT_DRAWINGS_CURRENT_MEAN": 0,
                "CC_CNT_DRAWINGS_CURRENT_VAR": 0
                }''')

            data_client = json.loads(data_client)

    else:
        loader = st.file_uploader(" ")
        if loader is not None:
            data_client = pd.read_csv(
                loader,
                sep=";",
                index_col=0,
                header=None
            ).squeeze(1).to_dict()

    run_btn2 = st.button(
        'Prédire',
        on_click=None,
        type="primary",
        key='predict_btn2'
    )
    
    if run_btn2:
        execute_API(data_client)
        data_client = pd.DataFrame(data_client, index=[0])

        # Vérifiez et convertissez les données au bon format si nécessaire
        if not isinstance(data_client, pd.DataFrame):
            data_client = pd.DataFrame(data_client)

        # Calcul des valeurs SHAP
        shap_values_client = explainer.shap_values(data_client)

        # Si shap_values_client est une liste, prendre les valeurs pour la classe 1
        if isinstance(shap_values_client, list):
            shap_values_client = shap_values_client[1]

        # Affichage des résultats SHAP avec un waterfall plot
        st.write("Valeurs SHAP pour ce client :")
        shap.initjs()

        # Waterfall plot
        st.pyplot(shap.waterfall_plot(shap.Explanation(
            values=shap_values_client[0],
            base_values=explainer.expected_value,
            data=data_client.iloc[0, :],
            feature_names=data_client.columns
        )))

        # Force plot
        st.pyplot(shap.force_plot(
            explainer.expected_value,
            shap_values_client[0],
            data_client,
            matplotlib=True
        ))

        # Autres visualisations (ex: plot_client)
        plot_client(
            data_client,
            explainer,
            df_reference=df_train,
            index_client=0  # Utilisation d'un index fictif (0) pour un nouveau client
        )

        nan_values(data_client, index_client=0)
