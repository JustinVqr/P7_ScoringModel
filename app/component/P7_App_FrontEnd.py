import streamlit as st
import requests
import json
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
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
                st.write(f"Prédiction : {'Oui' if result['prediction'] == 1 else 'Non'}")
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
    Cette fonction génère un graphique des valeurs SHAP principales.
    Elle permet de comprendre la prédiction sur le risque de défaut de prêt pour un client spécifique.

    Entrées :
    explainer > l'explainer SHAP (TreeExplainer par exemple)
    df > un DataFrame pandas avec les 62 caractéristiques et leurs valeurs
    index_client > l'index du client dans le DataFrame

    Sortie :
    Un diagramme en barres des valeurs SHAP, intégré dans une figure Streamlit
    """

    # Vérification que l'index du client existe bien dans le DataFrame
    if index_client not in df.index:
        st.error(f"L'index client {index_client} n'existe pas dans le DataFrame.")
        return

    # Sélection des données pour le client et gestion des valeurs manquantes
    X = df.fillna(0).loc[[index_client]]  # Sélection d'une seule ligne sous forme de DataFrame
    
    # Conversion explicite en NumPy array (nécessaire pour LightGBM)
    X_array = X.values.astype(np.float32)  # Conversion en float32
    
    # Affichage des informations pour le débogage
    st.write(f"Type des données du client : {type(X)}")
    st.write(f"Forme des données du client : {X.shape}")
    st.write(f"Type après conversion en array : {type(X_array)}")
    st.write(f"Forme après conversion : {X_array.shape}")
    st.write(f"Type de données après conversion : {X_array.dtype}")
    
    try:
        # Appel de l'explainer SHAP avec les données converties en NumPy array
        shap_values = explainer.shap_values(X_array)
        
        # Si shap_values est une liste, prendre les valeurs pour la classe 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Génération du graphique SHAP avec les noms de *features*
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, show=False, max_display=10, ax=ax)

        # Affichage du graphique dans Streamlit
        st.pyplot(fig)
        
        # Nettoyage du graphique pour éviter les conflits dans les graphiques suivants
        plt.clf()

    except TypeError as e:
        st.error(f"Une erreur est survenue lors de l'appel à l'explainer SHAP : {str(e)}")
        st.error("Vérifiez que les données passées à l'explainer sont correctes (DataFrame ou NumPy array).")


def plot_client(df, explainer, df_reference, index_client=0):
    """ 
    Cette fonction génère différentes visualisations pour comprendre la prédiction du défaut de prêt pour un client spécifique.
    Elle appelle d'abord la fonction shap_plot pour générer l'explication via SHAP, puis elle génère 6 graphiques pour les 6 caractéristiques les plus discriminantes.
    
    - Pour les caractéristiques binaires (0/1), le graphique est un barplot montrant la fréquence de la caractéristique dans chaque classe (TARGET).
    - Pour les autres caractéristiques, elle génère des boxplots avec la valeur du client (point rouge) et les lignes pointillées verticales montrant la moyenne de chaque classe (TARGET).

    Entrées :
    - df : DataFrame pandas contenant les 70 caractéristiques et leurs valeurs
    - explainer : explainer SHAP
    - df_reference : jeu de données d'entraînement utilisé comme référence pour les graphiques
    - index_client : index du client dans le DataFrame
    
    Sorties :
    - Graphique des valeurs SHAP
    - 6 graphiques pour les 6 caractéristiques les plus discriminantes
    """
    
    # --- Graphique SHAP pour un client spécifique ---
    shap_plot(explainer, df, index_client)

    # --- Calcul de l'importance SHAP ---
    shap_values = explainer.shap_values(df.fillna(0).loc[[index_client]])

    # Si shap_values est une liste (pour un modèle binaire), on prend les valeurs pour la classe 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Classe 1 = défaut de paiement

    # Aplatir les shap_values pour s'assurer qu'il s'agit d'une seule dimension
    shap_values = shap_values.flatten()

    # Création de la série shap_importance avec l'index des colonnes du DataFrame
    shap_importance = pd.Series(shap_values, index=df.columns).abs().sort_values(ascending=False)

    # --- 6 caractéristiques les plus discriminantes ---
    st.subheader('Explication : Top 6 caractéristiques discriminantes')

    # Les graphiques sont divisés en deux colonnes Streamlit, avec 3 graphiques par colonne
    col1, col2 = st.columns(2)

    with col1:
        for feature in list(shap_importance.index[:6])[:3]:
            plt.figure(figsize=(5, 5))

            # Pour les caractéristiques binaires :
            if df_reference[feature].nunique() == 2:
                # Barplot de la fréquence par classe :
                figInd = sns.barplot(df_reference[['TARGET', feature]].fillna(0).groupby(
                    'TARGET').value_counts(normalize=True).reset_index(), x=feature, y=0, hue='TARGET')
                plt.ylabel('Fréquence des clients')

                # Ajout des données du client :
                plt.scatter(y=df[feature].loc[index_client] + 0.1, x=feature, marker='o', s=100, color="r")
                figInd.annotate('Client ID:\n{}'.format(index_client), xy=(feature, df[feature].loc[index_client] + 0.1),
                                xytext=(0, 40), textcoords='offset points', ha='center', va='bottom',
                                bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))

                # Personnalisation de la légende
                legend_handles, _ = figInd.get_legend_handles_labels()
                figInd.legend(legend_handles, ['Non', 'Oui'], title="Défaut de prêt")
                st.pyplot(figInd.figure)
                plt.close()

            # Pour les caractéristiques non binaires :
            else:
                figInd = sns.boxplot(data=df_reference, y=feature, x='TARGET', showfliers=False, width=0.2)
                plt.xlabel('Défaut de prêt')
                figInd.set_xticklabels(["Non", "Oui"])

                # Ajout des données du client :
                plt.scatter(y=df[feature].loc[index_client], x=0.5, marker='o', s=100, color="r")
                figInd.annotate('Client ID:\n{}'.format(index_client), xy=(0.5, df[feature].loc[index_client]),
                                xytext=(0, 40), textcoords='offset points', ha='center', va='bottom',
                                bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))

                # Ajout des moyennes de chaque classe :
                figInd.axhline(y=df_reference[df_reference['TARGET'] == 0][feature].mean(), linestyle='--', color="#1f77b4")
                figInd.axhline(y=df_reference[df_reference['TARGET'] == 1][feature].mean(), linestyle='--', color="#ff7f0e")

                # Légende personnalisée :
                colors = ["#1f77b4", "#ff7f0e"]
                lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='--') for c in colors]
                labels = ["Moyenne Sans Défaut", "Moyenne Avec Défaut"]
                plt.legend(lines, labels, title="Moyenne des clients :")
                st.pyplot(figInd.figure)
                plt.close()

    with col2:
        for feature in list(shap_importance.index[:6])[3:]:
            plt.figure(figsize=(5, 5))

            # Pour les caractéristiques binaires :
            if df_reference[feature].nunique() == 2:
                figInd = sns.barplot(df_reference[['TARGET', feature]].fillna(0).groupby(
                    'TARGET').value_counts(normalize=True).reset_index(), x=feature, y=0, hue='TARGET')
                plt.ylabel('Fréquence des clients')

                # Ajout des données du client :
                plt.scatter(y=df[feature].loc[index_client] + 0.1, x=feature, marker='o', s=100, color="r")
                figInd.annotate('Client ID:\n{}'.format(index_client), xy=(feature, df[feature].loc[index_client] + 0.1),
                                xytext=(0, 40), textcoords='offset points', ha='center', va='bottom',
                                bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))

                legend_handles, _ = figInd.get_legend_handles_labels()
                figInd.legend(legend_handles, ['Non', 'Oui'], title="Défaut de prêt")
                st.pyplot(figInd.figure)
                plt.close()

            # Pour les caractéristiques non binaires :
            else:
                figInd = sns.boxplot(data=df_reference, y=feature, x='TARGET', showfliers=False, width=0.2)
                plt.xlabel('Défaut de prêt')
                figInd.set_xticklabels(["Non", "Oui"])

                # Ajout des données du client :
                plt.scatter(y=df[feature].loc[index_client], x=0.5, marker='o', s=100, color="r")
                figInd.annotate('Client ID:\n{}'.format(index_client), xy=(0.5, df[feature].loc[index_client]),
                                xytext=(0, 40), textcoords='offset points', ha='center', va='bottom',
                                bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))

                # Ajout des moyennes de chaque classe :
                figInd.axhline(y=df_reference[df_reference['TARGET'] == 0][feature].mean(), linestyle='--', color="#1f77b4")
                figInd.axhline(y=df_reference[df_reference['TARGET'] == 1][feature].mean(), linestyle='--', color="#ff7f0e")

                colors = ["#1f77b4", "#ff7f0e"]
                lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='--') for c in colors]
                labels = ["Moyenne Sans Défaut", "Moyenne Avec Défaut"]
                plt.legend(lines, labels, title="Moyenne des clients :")
                st.pyplot(figInd.figure)
                plt.close()


    # --- Analysis of unknown values ---


def nan_values(df, index_client=0):
    if np.isnan(df.loc[index_client]).any():

        st.subheader('Warnings : Columns with unknown values')
        nan_col = []
        for features in list(df.columns):
            if np.isnan(df.loc[index_client][features]):
                nan_col.append(features)

        col1, col2 = st.columns(2)
        with col1:
            st.table(
                data=pd.DataFrame(
                    nan_col,
                    columns=['FEATURES WITH MISSING VALUES']))
        with col2:
            st.write('All the missing values has been replaced by 0.')
    else:
        st.subheader('There is no missing value')