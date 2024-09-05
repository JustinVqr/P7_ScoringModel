import streamlit as st
import requests
import json
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patches
import matplotlib.colors

# Fonction sans API pour afficher la prédiction et les probabilités de défaut de paiement pour un client spécifique
def execute_noAPI(df, index_client, model):
    """ 
    Fonction générant les colonnes dans l'interface Streamlit montrant la prédiction du défaut de paiement.
    """
    st.subheader('Difficultés du client : ')
    predict_proba = model.predict_proba([df.drop(columns='TARGET').fillna(0).loc[index_client]])[:, 1]
    predict_target = (predict_proba >= 0.4).astype(int)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Difficultés", str(np.where(df['TARGET'].loc[index_client] == 0, 'NON', 'OUI')))
    col2.metric("Difficultés Prédites", str(np.where(predict_target == 0, 'NON', 'OUI'))[2:-2])
    col3.metric("Probabilité", predict_proba.round(2))

# Interface Streamlit pour saisir l'ID client
st.title("Prédiction du défaut de paiement")

# Saisie de l'ID client
sk_id_curr = st.text_input("Entrez l'ID du client pour obtenir la prédiction :")

# Bouton pour lancer la prédiction
if st.button("Obtenir la prédiction via l'API"):
    if sk_id_curr:
        try:
            # Vérification si les données sont déjà chargées
            if 'df_new' in st.session_state:
                df_new = st.session_state.df_new
            else:
                st.error("Les données client ne sont pas chargées.")
                st.stop()

            # Récupérer les données du client correspondant à l'ID
            client_data = df_new.loc[int(sk_id_curr)].fillna(0).to_dict()

            # URL de l'API
            api_url = "https://app-scoring-p7-b4207842daea.herokuapp.com/predict"

            # Envoi de la requête POST à l'API avec toutes les caractéristiques du client
            response = requests.post(api_url, json=client_data)

            # Vérification du statut de la réponse
            if response.status_code == 200:
                result = response.json()
                st.write(f"Prédiction : {'Oui' si result['prediction'] == 1 else 'Non'}")
                st.write(f"Probabilité de défaut : {result['probability'] * 100:.2f}%")
            else:
                st.error(f"Erreur : {response.json()['detail']}")
        
        except Exception as e:
            st.error(f"Erreur lors de la requête à l'API : {e}")
    else:
        st.error("Veuillez entrer un ID client valide.")

def execute_API(df):
    st.subheader('Client difficulties : ')
    
    # Effectuer la requête
    request = requests.post(
        url="https://app-scoring-p7-b4207842daea.herokuapp.com/predict",
        data=json.dumps(df),
        headers={"Content-Type": "application/json"}
    )
    
    # Vérifier si la requête a réussi
    if request.status_code == 200:
        response_json = request.json()  # Obtenez la réponse JSON
        
        # Afficher la réponse JSON complète pour diagnostic
        st.write(response_json)
        
        # s'assurer que les clés sont présentes
        if "prediction" in response_json and "probability" in response_json:
            prediction = response_json["prediction"]
            probability = round(response_json["probability"], 2)
            
            # Afficher les résultats
            col1, col2 = st.columns(2)
            col1.metric("Predicted Difficulties", str(np.where(prediction == 0, 'NO', 'YES')))
            col2.metric("Probability of default", probability)
        else:
            st.error("Les clés 'prediction' ou 'probability' sont manquantes dans la réponse.")
    else:
        st.error(f"Erreur avec la requête API : {request.status_code}")


def shap_plot(explainer, df, index_client=0):
    """ 
    This function generates a plot of the main shap value.
    It helps to understand the prediction on loan default for a specific client.
    """
    try:
        # Vérification du type des données avant de les passer à SHAP
        st.write(f"Type des données passées à SHAP : {type(df)}")
        st.write(f"Index du client : {index_client}")

        # Vérification du contenu des données pour le client
        st.write(f"Données pour le client : {df.fillna(0).loc[index_client]}")
        
        # Plot shap values
        fig_shap = shap.plots.bar(
            explainer(df.fillna(0).loc[index_client]),  # Cette ligne pourrait causer un problème
            show=False)
        
        # Affichage du graphique dans Streamlit
        st.pyplot(fig_shap)
        plt.clf()

    except Exception as e:
        # Capturer et afficher toute erreur rencontrée lors de l'appel à SHAP
        st.error(f"Erreur lors de la génération des valeurs SHAP : {e}")


def plot_client(df, explainer, df_reference, index_client=0):
    """ 
    This function generates all the different plots to understand the prediction of loan default for a specific client.
    """
    
    try:
        # Vérification du type de données et des colonnes avant d'appeler shap_plot
        st.write(f"Type des données dans plot_client : {type(df)}")
        st.write(f"Colonnes des données : {df.columns}")
        
        # --- Bar plot of the shap value for a specific client ---
        shap_plot(explainer, df, index_client)

    except Exception as e:
        st.error(f"Erreur dans plot_client : {e}")

    # --- Calcul des 6 caractéristiques les plus discriminatives ---
    try:
        shap_values = explainer.shap_values(df.fillna(0).loc[index_client])
        shap_importance = pd.Series(
            shap_values,
            df.columns).abs().sort_values(
            ascending=False)

        # --- 6 discriminative features ---
        st.subheader('Explaination : Top 6 discriminative features')

        # The figures are divided in two streamlit columns, with 3 plots per column
        col1, col2 = st.columns(2)
        for col, features_slice in zip([col1, col2], [list(shap_importance.index[:3]), list(shap_importance.index[3:6])]):
            with col:
                for features in features_slice:
                    plt.figure(figsize=(5, 5))

                    # For binary features :
                    if df_reference[features].nunique() == 2:
                        figInd = sns.barplot(df_reference[['TARGET', features]].fillna(0).groupby(
                            'TARGET').value_counts(normalize=True).reset_index(), x=features, y=0, hue='TARGET')
                        plt.ylabel('Freq of client')

                        # Client data annotation
                        plt.scatter(y=df[features].loc[index_client] + 0.1, x=features, marker='o', s=100, color="r")
                        figInd.annotate(
                            f'Client ID:\n{index_client}', xy=(features, df[features].loc[index_client] + 0.1), xytext=(0, 40), 
                            textcoords='offset points', ha='center', va='bottom', bbox=dict(boxstyle="round", fc="w"), 
                            arrowprops=dict(arrowstyle="->")
                        )
                        legend_handles, _ = figInd.get_legend_handles_labels()
                        figInd.legend(legend_handles, ['No', 'Yes'], title="LOAN DEFAULT")
                        st.pyplot(figInd.figure)
                        plt.close()

                    # For non-binary features :
                    else:
                        figInd = sns.boxplot(data=df_reference, y=features, x='TARGET', showfliers=False, width=0.2)
                        plt.xlabel('LOAN DEFAULT')
                        figInd.set_xticklabels(["No", "Yes"])

                        plt.scatter(y=df[features].loc[index_client], x=0.5, marker='o', s=100, color="r")

                        figInd.annotate(
                            f'Client ID:\n{index_client}', xy=(0.5, df[features].loc[index_client]), xytext=(0, 40), 
                            textcoords='offset points', ha='center', va='bottom', bbox=dict(boxstyle="round", fc="w"), 
                            arrowprops=dict(arrowstyle="->")
                        )

                        # Mean value of each class + client:
                        figInd.axhline(y=df_reference[df_reference['TARGET'] == 0][features].mean(), zorder=0, linestyle='--', color="#1f77b4")
                        figInd.axhline(y=df_reference[df_reference['TARGET'] == 1][features].mean(), zorder=0, linestyle='--', color="#ff7f0e")

                        # Custom legend:
                        colors = ["#1f77b4", "#ff7f0e"]
                        lines = [Line
