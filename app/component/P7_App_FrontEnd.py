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
    # Calcul des probabilités de défaut pour le client donné
    predict_proba = model.predict_proba([df.drop(columns='TARGET').fillna(0).loc[index_client]])[:, 1]
    predict_target = (predict_proba >= 0.4).astype(int)
    
    # Affichage des résultats sous forme de colonnes Streamlit
    col1, col2, col3 = st.columns(3)
    col1.metric("Difficultés", str(np.where(df['TARGET'].loc[index_client] == 0, 'NON', 'OUI')))
    col2.metric("Difficultés Prédites", str(np.where(predict_target == 0, 'NON', 'OUI'))[2:-2])
    col3.metric("Probabilité", predict_proba.round(2))

# Interface principale Streamlit
st.title("Prédiction du défaut de paiement")

# Saisie de l'ID client
sk_id_curr = st.text_input("Entrez l'ID du client pour obtenir la prédiction :")

# Bouton pour déclencher la prédiction via l'API
if st.button("Obtenir la prédiction via l'API"):
    if sk_id_curr:
        try:
            # Chargement des données si elles sont déjà présentes
            if 'df_new' in st.session_state:
                df_new = st.session_state.df_new
            else:
                st.error("Les données client ne sont pas chargées.")
                st.stop()

            # Récupération des données du client correspondant à l'ID
            client_data = df_new.loc[int(sk_id_curr)].fillna(0).to_dict()

            # URL de l'API de prédiction
            api_url = "https://app-scoring-p7-b4207842daea.herokuapp.com/predict"

            # Envoi de la requête POST à l'API avec les données du client
            response = requests.post(api_url, json=client_data)

            # Vérification du statut de la réponse
            if response.status_code == 200:
                result = response.json()
                st.write(f"Prédiction : {'Oui' si result['prediction'] == 1 sinon 'Non'}")
                st.write(f"Probabilité de défaut : {result['probability'] * 100:.2f}%")
            else:
                st.error(f"Erreur : {response.json()['detail']}")
        
        except Exception as e:
            st.error(f"Erreur lors de la requête à l'API : {e}")
    else:
        st.error("Veuillez entrer un ID client valide.")

# Fonction pour effectuer la prédiction via l'API
def execute_API(df):
    """ Cette fonction envoie les données du client à une API pour obtenir la prédiction """
    st.subheader('Difficultés du client : ')

    # Envoi de la requête POST
    request = requests.post(
        url="https://app-scoring-p7-b4207842daea.herokuapp.com/predict",
        data=json.dumps(df),
        headers={"Content-Type": "application/json"}
    )

    # Traitement de la réponse de l'API
    if request.status_code == 200:
        response_json = request.json()
        st.write(response_json)  # Affichage pour diagnostic

        # Vérification que les clés sont présentes
        if "prediction" in response_json and "probability" in response_json:
            prediction = response_json["prediction"]
            probability = round(response_json["probability"], 2)

            # Affichage des résultats sous forme de colonnes Streamlit
            col1, col2 = st.columns(2)
            col1.metric("Difficultés Prédites", str(np.where(prediction == 0, 'NON', 'OUI')))
            col2.metric("Probabilité de défaut", probability)
        else:
            st.error("Les clés 'prediction' ou 'probability' sont manquantes dans la réponse.")
    else:
        st.error(f"Erreur avec la requête API : {request.status_code}")

# Fonction pour générer les graphiques SHAP
def shap_plot(explainer, df, index_client=0):
    """ Génère un graphique des valeurs SHAP pour comprendre la prédiction d'un client spécifique """
    fig_shap = shap.plots.bar(explainer(df.fillna(0).loc[index_client]), show=False)
    
    # Modification des couleurs pour une meilleure visualisation
    for fc in plt.gcf().get_children():
        for fcc in fc.get_children()[:-1]:
            if isinstance(fcc, matplotlib.patches.Rectangle):
                # Modification des couleurs pour les valeurs positives et négatives
                if matplotlib.colors.to_hex(fcc.get_facecolor()) == "#ff0051":
                    fcc.set_facecolor("#ff7f0e")  # Couleur personnalisée
                elif matplotlib.colors.to_hex(fcc.get_facecolor()) == "#008bfb":
                    fcc.set_color("#1f77b4")
    st.pyplot(fig_shap)
    plt.clf()

# Fonction pour afficher les graphiques et explications pour un client spécifique
def plot_client(df, explainer, df_reference, index_client=0):
    """ Génère les graphiques pour comprendre la prédiction pour un client spécifique """
    shap_plot(explainer, df, index_client)

    # Calcul des valeurs SHAP
    shap_values = explainer.shap_values(df.fillna(0).loc[index_client])
    shap_importance = pd.Series(shap_values, df.columns).abs().sort_values(ascending=False)

    st.subheader('Explication : Top 6 caractéristiques discriminantes')

    # Affichage des graphiques des 6 caractéristiques principales
    col1, col2 = st.columns(2)
    for i, col in enumerate([col1, col2]):
        for feature in shap_importance.index[i*3:(i+1)*3]:
            plt.figure(figsize=(5, 5))
            if df_reference[feature].nunique() == 2:
                figInd = sns.barplot(df_reference[['TARGET', feature]].fillna(0).groupby('TARGET').value_counts(normalize=True).reset_index(), x=feature, y=0, hue='TARGET')
                plt.scatter(y=df[feature].loc[index_client] + 0.1, x=feature, marker='o', s=100, color="r")
                st.pyplot(figInd.figure)
                plt.close()
            else:
                figInd = sns.boxplot(data=df_reference, y=feature, x='TARGET', showfliers=False)
                plt.scatter(y=df[feature].loc[index_client], x=0.5, marker='o', s=100, color="r")
                st.pyplot(figInd.figure)
                plt.close()

# Fonction pour détecter et gérer les valeurs manquantes
def nan_values(df, index_client=0):
    """ Vérifie et affiche les colonnes contenant des valeurs manquantes pour le client donné """
    if np.isnan(df.loc[index_client]).any():
        st.subheader('Colonnes avec valeurs manquantes')
        nan_col = [feature for feature in df.columns if np.isnan(df.loc[index_client][feature])]
        st.table(pd.DataFrame(nan_col, columns=['Colonnes avec valeurs manquantes']))
        st.write('Les valeurs manquantes ont été remplacées par 0.')
    else:
        st.subheader('Aucune valeur manquante')

