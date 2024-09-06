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
    Cette fonction génère un graphique waterfall des valeurs SHAP pour un client spécifique.
    """

    # Vérification que l'index du client existe dans le DataFrame
    if index_client not in df.index:
        st.error(f"L'index client {index_client} n'existe pas dans le DataFrame.")
        return

    # Sélection des données pour le client
    X = df.fillna(0).loc[[index_client]]

    try:
        # Appel de l'explainer SHAP pour obtenir un objet Explanation
        shap_values = explainer(X)

        # Génération du waterfall plot pour visualiser les valeurs SHAP d'un client spécifique
        st.write("Valeurs SHAP pour ce client :")
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap_values[0], show=False)  # Waterfall plot
        
        # Affichage du graphique dans Streamlit
        st.pyplot(fig)
        plt.clf()

        # --- Ajout du bar plot pour l'importance globale des features ---
        st.write("Importance globale des caractéristiques :")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, show=False)  # Bar plot pour l'importance globale des features
        st.pyplot(fig)
        plt.clf()

    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'appel à l'explainer SHAP : {str(e)}")
        st.error("Vérifiez que les données passées à l'explainer sont correctes.")


def plot_client(df, explainer, df_reference, index_client=0):
    """ 
    Cette fonction génère différentes visualisations pour comprendre la prédiction du défaut de prêt pour un client spécifique.
    Elle génère 6 graphiques pour les 6 caractéristiques les plus discriminantes.
    """
    
    # --- Calcul de l'importance SHAP ---
    try:
        shap_values = explainer(df.fillna(0).loc[[index_client]])  # Utilisation de explainer() au lieu de shap_values()
        
        # Extraire les valeurs SHAP pour la classe 1 si nécessaire
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Classe 1 = défaut de paiement
        
        # Extraire les valeurs SHAP
        shap_values = shap_values.values.flatten()
        
        # Création de la série shap_importance
        shap_importance = pd.Series(shap_values, index=df.columns).abs().sort_values(ascending=False)

    except Exception as e:
        st.error(f"Erreur lors du calcul des valeurs SHAP : {e}")
        return

    # --- 6 caractéristiques les plus discriminantes ---
    st.subheader('Explication : Top 6 caractéristiques discriminantes')

    # Les graphiques sont divisés en deux colonnes Streamlit, avec 3 graphiques par colonne
    col1, col2 = st.columns(2)

    with col1:
        for feature in list(shap_importance.index[:6])[:3]:
            try:
                plt.figure(figsize=(5, 5))

                # Pour les caractéristiques binaires :
                if df_reference[feature].nunique() == 2:
                    figInd = sns.barplot(df_reference[['TARGET', feature]].fillna(0).groupby(
                        'TARGET').value_counts(normalize=True).reset_index(), x=feature, y=0, hue='TARGET')
                    plt.ylabel('Fréquence des clients')

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

                    plt.scatter(y=df[feature].loc[index_client], x=0.5, marker='o', s=100, color="r")
                    figInd.annotate('Client ID:\n{}'.format(index_client), xy=(0.5, df[feature].loc[index_client]),
                                    xytext=(0, 40), textcoords='offset points', ha='center', va='bottom',
                                    bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))

                    figInd.axhline(y=df_reference[df_reference['TARGET'] == 0][feature].mean(), linestyle='--', color="#1f77b4")
                    figInd.axhline(y=df_reference[df_reference['TARGET'] == 1][feature].mean(), linestyle='--', color="#ff7f0e")

                    colors = ["#1f77b4", "#ff7f0e"]
                    lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='--') for c in colors]
                    labels = ["Moyenne Sans Défaut", "Moyenne Avec Défaut"]
                    plt.legend(lines, labels, title="Moyenne des clients :")
                    st.pyplot(figInd.figure)
                    plt.close()

            except Exception as e:
                st.error(f"Erreur lors de la génération du graphique pour la caractéristique {feature} : {e}")

    with col2:
        for feature in list(shap_importance.index[:6])[3:]:
            try:
                plt.figure(figsize=(5, 5))

                if df_reference[feature].nunique() == 2:
                    figInd = sns.barplot(df_reference[['TARGET', feature]].fillna(0).groupby(
                        'TARGET').value_counts(normalize=True).reset_index(), x=feature, y=0, hue='TARGET')
                    plt.ylabel('Fréquence des clients')

                    plt.scatter(y=df[feature].loc[index_client] + 0.1, x=feature, marker='o', s=100, color="r")
                    figInd.annotate('Client ID:\n{}'.format(index_client), xy=(feature, df[feature].loc[index_client] + 0.1),
                                    xytext=(0, 40), textcoords='offset points', ha='center', va='bottom',
                                    bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))

                    legend_handles, _ = figInd.get_legend_handles_labels()
                    figInd.legend(legend_handles, ['Non', 'Oui'], title="Défaut de prêt")
                    st.pyplot(figInd.figure)
                    plt.close()

                else:
                    figInd = sns.boxplot(data=df_reference, y=feature, x='TARGET', showfliers=False, width=0.2)
                    plt.xlabel('Défaut de prêt')
                    figInd.set_xticklabels(["Non", "Oui"])

                    plt.scatter(y=df[feature].loc[index_client], x=0.5, marker='o', s=100, color="r")
                    figInd.annotate('Client ID:\n{}'.format(index_client), xy=(0.5, df[feature].loc[index_client]),
                                    xytext=(0, 40), textcoords='offset points', ha='center', va='bottom',
                                    bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))

                    figInd.axhline(y=df_reference[df_reference['TARGET'] == 0][feature].mean(), linestyle='--', color="#1f77b4")
                    figInd.axhline(y=df_reference[df_reference['TARGET'] == 1][feature].mean(), linestyle='--', color="#ff7f0e")

                    colors = ["#1f77b4", "#ff7f0e"]
                    lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='--') for c in colors]
                    labels = ["Moyenne Sans Défaut", "Moyenne Avec Défaut"]
                    plt.legend(lines, labels, title="Moyenne des clients :")
                    st.pyplot(figInd.figure)
                    plt.close()

            except Exception as e:
                st.error(f"Erreur lors de la génération du graphique pour la caractéristique {feature} : {e}")


    # --- Analysis des valeurs manquantes ---

def nan_values(df, index_client=0):
    # Utiliser pd.isna() 
    if df.loc[index_client].isna().any():
        st.subheader('Warnings : Columns with unknown values')
        nan_col = []
        for feature in df.columns:
            # Vérifier les valeurs manquantes pour chaque caractéristique
            if pd.isna(df.loc[index_client, feature]):
                nan_col.append(feature)

        col1, col2 = st.columns(2)
        with col1:
            st.table(data=pd.DataFrame(nan_col, columns=['FEATURES WITH MISSING VALUES']))
        with col2:
            st.write('Toutes les valeurs manquantes ont été remplacées par 0.')
    else:
        st.subheader('Il n\'y a pas de valeurs manquantes dans la base de données concernant ce client')
