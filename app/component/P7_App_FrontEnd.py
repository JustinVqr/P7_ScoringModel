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
    
    # Convertir en DataFrame d'une seule ligne si nécessaire
    client_data = df.drop(columns='TARGET').fillna(0).loc[[index_client]]
    
    predict_proba = model.predict_proba(client_data)[:, 1]
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
    """ Cette fonction génère un graphique des valeurs SHAP pour un client spécifique.
    Elle aide à comprendre la prédiction du défaut de paiement pour ce client.

    input :
    explainer > l'explainer SHAP
    df > dataframe pandas avec les 62 caractéristiques
    index_client > index du client

    output :
    Bar plot des valeurs SHAP, intégré dans une figure Streamlit
    """
    # Assurez-vous que df.loc[index_client] est bien un DataFrame d'une seule ligne
    client_data = df.fillna(0).loc[[index_client]]  # Double crochets pour garder un DataFrame

    # Plot shap values
    fig_shap = shap.plots.bar(explainer(client_data), show=False)

    # Personnalisation des couleurs
    default_pos_color = "#ff0051"
    default_neg_color = "#008bfb"
    positive_color = "#ff7f0e"
    negative_color = "#1f77b4"

    # Modifier les couleurs dans le graphique
    for fc in plt.gcf().get_children():
        for fcc in fc.get_children()[:-1]:
            if isinstance(fcc, matplotlib.patches.Rectangle):
                if matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color:
                    fcc.set_facecolor(positive_color)
                elif matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color:
                    fcc.set_facecolor(negative_color)
            elif isinstance(fcc, plt.Text):
                if matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color:
                    fcc.set_color(positive_color)
                elif matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color:
                    fcc.set_color(negative_color)

    st.pyplot(fig_shap)
    plt.clf()


def plot_client(df, explainer, df_reference, index_client=0):
    """ Cette fonction génère des graphiques pour comprendre la prédiction du défaut de paiement pour un client spécifique.
    Elle appelle d'abord la fonction shap_plot pour générer le graphique des valeurs SHAP,
    puis génère 6 graphiques pour les 6 caractéristiques les plus discriminantes.

    input :
    df > dataframe pandas avec les 62 caractéristiques
    explainer > l'explainer SHAP
    df_reference > le dataset d'entraînement utilisé comme référence
    index_client > index du client
    """

    # ---Graphique des valeurs SHAP pour un client spécifique---
    shap_plot(explainer, df, index_client)

    # --- Calcul des valeurs SHAP importance ---
    shap_values = explainer.shap_values(df.fillna(0).loc[[index_client]])  # Double crochets pour DataFrame
    shap_importance = pd.Series(shap_values, df.columns).abs().sort_values(ascending=False)

    # --- Top 6 caractéristiques discriminantes ---
    st.subheader('Explication : Top 6 caractéristiques discriminantes')

    # Les graphiques sont divisés en deux colonnes dans Streamlit, avec 3 graphiques par colonne
    col1, col2 = st.columns(2)

    for i, col in enumerate([col1, col2]):
        for feature in list(shap_importance.index[i*3:(i+1)*3]):
            plt.figure(figsize=(5, 5))

            if df_reference[feature].nunique() == 2:
                # Bar plot pour les features binaires
                figInd = sns.barplot(df_reference[['TARGET', feature]].fillna(0).groupby('TARGET').value_counts(normalize=True).reset_index(), x=feature, y=0, hue='TARGET')
                plt.ylabel('Fréquence du client')

                # Ajouter les données du client
                plt.scatter(y=df[feature].loc[index_client] + 0.1, x=feature, marker='o', s=100, color="r")
                figInd.annotate(f'ID Client:\n{index_client}', xy=(feature, df[feature].loc[index_client] + 0.1), xytext=(0, 40), textcoords='offset points', ha='center', va='bottom', bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
                legend_handles, _ = figInd.get_legend_handles_labels()
                figInd.legend(legend_handles, ['Non', 'Oui'], title="Défaut de paiement")
                st.pyplot(figInd.figure)
                plt.close()

            else:
                # Box plot pour les features non binaires
                figInd = sns.boxplot(data=df_reference, y=feature, x='TARGET', showfliers=False, width=0.2)
                plt.xlabel('Défaut de paiement')
                figInd.set_xticklabels(["Non", "Oui"])

                # Ajouter les données du client
                plt.scatter(y=df[feature].loc[index_client], x=0.5, marker='o', s=100, color="r")
                figInd.annotate(f'ID Client:\n{index_client}', xy=(0.5, df[feature].loc[index_client]), xytext
