import streamlit as st
import requests
import json
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

# Fonction sans API pour afficher la prédiction et les probabilités de défaut de paiement pour un client spécifique
def execute_noAPI(df, index_client, model):
    """ 
    Cette fonction génère des colonnes dans l'interface Streamlit montrant la prédiction du défaut de paiement pour un client spécifique.

    Entrée :
    df : un dataframe pandas
    index_client : index du client à analyser
    model : le modèle de machine learning (ici LightGBM)

    Sortie :
    3 colonnes affichant la cible connue (Difficultés), la cible prédite et la probabilité
    """
    
    # Préparation des données
    st.subheader('Difficultés du client : ')
    predict_proba = model.predict_proba([df.drop(columns='TARGET').fillna(0).loc[index_client]])[:, 1]
    predict_target = (predict_proba >= 0.4).astype(int)
    
    # Affichage des résultats dans des colonnes
    col1, col2, col3 = st.columns(3)
    col1.metric("Difficultés", str(np.where(df['TARGET'].loc[index_client] == 0, 'NON', 'OUI')))
    col2.metric("Difficultés Prédites", str(np.where(predict_target == 0, 'NON', 'OUI'))[2:-2])
    col3.metric("Probabilité", predict_proba.round(2))


# --- Interface Streamlit pour saisir l'ID client ---
st.title("Prédiction du défaut de paiement")

# Saisie de l'ID client
sk_id_curr = st.text_input("Entrez l'ID du client pour obtenir la prédiction :")

# Bouton pour lancer la prédiction
if st.button("Obtenir la prédiction via l'API"):
    if sk_id_curr:
        # Préparer la requête à l'API
        api_url = "https://app-scoring-p7-b4207842daea.herokuapp.com/predict"  # URL de l'API FastAPI, à adapter selon votre configuration
        client_id = {"SK_ID_CURR": int(sk_id_curr)}  # Créer un dictionnaire avec l'ID client
        
        try:
            # Envoi de la requête POST à l'API
            response = requests.post(api_url, json=client_id)
            
            # Vérification du statut de la réponse
            if response.status_code == 200:
                result = response.json()
                
                # Affichage des résultats
                st.write(f"Prédiction : {'Oui' if result['prediction'] == 1 else 'Non'}")
                st.write(f"Probabilité de défaut : {result['probability'] * 100:.2f}%")
            else:
                st.error(f"Erreur : {response.json()['detail']}")
        
        except Exception as e:
            st.error(f"Erreur lors de la requête à l'API : {e}")
    else:
        st.error("Veuillez entrer un ID client valide.")


# Fonction pour générer un graphique SHAP montrant l'importance des caractéristiques
def shap_plot(explainer, df, index_client=0):
    """
    Cette fonction génère un graphique montrant les principales valeurs SHAP (importance des caractéristiques) pour expliquer la prédiction.

    Entrée :
    explainer : l'objet SHAP explainer
    df : dataframe pandas avec les 62 caractéristiques
    index_client : index du client

    Sortie :
    Graphique en barres des valeurs SHAP, intégré dans une figure Streamlit
    """
    
    # Graphique des valeurs SHAP
    fig_shap = shap.plots.bar(explainer(df.fillna(0).loc[index_client]), show=False)

    # Modification des couleurs par défaut pour personnaliser l'affichage
    default_pos_color = "#ff0051"
    default_neg_color = "#008bfb"
    positive_color = "#ff7f0e"
    negative_color = "#1f77b4"

    for fc in plt.gcf().get_children():
        for fcc in fc.get_children()[:-1]:
            if isinstance(fcc, matplotlib.patches.Rectangle):
                if matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color:
                    fcc.set_facecolor(positive_color)
                elif matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color:
                    fcc.set_color(negative_color)
            elif isinstance(fcc, plt.Text):
                if matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color:
                    fcc.set_color(positive_color)
                elif matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color:
                    fcc.set_color(negative_color)

    # Affichage du graphique dans Streamlit
    st.pyplot(fig_shap)
    plt.clf()

# Fonction pour afficher les graphiques SHAP et les caractéristiques discriminantes
def plot_client(df, explainer, df_reference, index_client=0):
    """
    Cette fonction génère tous les graphiques pour comprendre la prédiction du défaut de paiement pour un client spécifique.
    Elle appelle d'abord shap_plot pour générer le graphique des valeurs SHAP, puis génère 6 graphiques pour les caractéristiques les plus discriminantes.

    Entrée :
    df : dataframe pandas avec les 62 caractéristiques
    explainer : l'objet SHAP explainer
    df_reference : le dataset d'entraînement utilisé comme référence
    index_client : index du client

    Sortie :
    1) Le graphique des valeurs SHAP
    2) 6 graphiques pour les 6 caractéristiques les plus discriminantes
    """
    
    # --- Graphique des valeurs SHAP pour un client spécifique ---
    shap_plot(explainer, df, index_client)

    # Calcul de l'importance des valeurs SHAP
    shap_values = explainer.shap_values(df.fillna(0).loc[index_client])
    shap_importance = pd.Series(shap_values, df.columns).abs().sort_values(ascending=False)

    # --- Affichage des 6 caractéristiques les plus discriminantes ---
    st.subheader('Explication : Top 6 caractéristiques les plus discriminantes')

    col1, col2 = st.columns(2)

    # Affichage des 3 premières caractéristiques dans la première colonne
    with col1:
        for feature in list(shap_importance.index[:6])[:3]:
            plt.figure(figsize=(5, 5))

            # Pour les caractéristiques binaires :
            if df_reference[feature].nunique() == 2:
                figInd = sns.barplot(df_reference[['TARGET', feature]].fillna(0).groupby('TARGET').value_counts(normalize=True).reset_index(), x=feature, y=0, hue='TARGET')
                plt.ylabel('Fréquence des clients')
                plt.scatter(y=df[feature].loc[index_client] + 0.1, x=feature, marker='o', s=100, color="r")
                figInd.annotate(f'ID Client:\n{index_client}', xy=(feature, df[feature].loc[index_client] + 0.1), xytext=(0, 40), textcoords='offset points', ha='center', va='bottom', bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
                legend_handles, _ = figInd.get_legend_handles_labels()
                figInd.legend(legend_handles, ['Non', 'Oui'], title="Défaut de paiement")
                st.pyplot(figInd.figure)
                plt.close()

            # Pour les caractéristiques non binaires :
            else:
                figInd = sns.boxplot(data=df_reference, y=feature, x='TARGET', showfliers=False, width=0.2)
                plt.xlabel('Défaut de paiement')
                figInd.set_xticklabels(["Non", "Oui"])
                plt.scatter(y=df[feature].loc[index_client], x=0.5, marker='o', s=100, color="r")
                figInd.annotate(f'ID Client:\n{index_client}', xy=(0.5, df[feature].loc[index_client]), xytext=(0, 40), textcoords='offset points', ha='center', va='bottom', bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
                figInd.axhline(y=df_reference[df_reference['TARGET'] == 0][feature].mean(), zorder=0, linestyle='--', color="#1f77b4")
                figInd.axhline(y=df_reference[df_reference['TARGET'] == 1][feature].mean(), zorder=0, linestyle='--', color="#ff7f0e")
                colors = ["#1f77b4", "#ff7f0e"]
                lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='--') for c in colors]
                labels = ["Moyenne sans défaut", "Moyenne avec défaut"]
                plt.legend(lines, labels, title="Valeur moyenne des clients:")
                st.pyplot(figInd.figure)
                plt.close()

    # Affichage des 3 autres caractéristiques dans la deuxième colonne
    with col2:
        for feature in list(shap_importance.index[:6])[3:]:
            plt.figure(figsize=(5, 5))

            if df_reference[feature].nunique() == 2:
                figInd = sns.barplot(df_reference[['TARGET', feature]].fillna(0).groupby('TARGET').value_counts(normalize=True).reset_index(), x=feature, y=0, hue='TARGET')
                plt.ylabel('Fréquence des clients')
                plt.scatter(y=df[feature].loc[index_client] + 0.1, x=feature, marker='o', s=100, color="r")
                figInd.annotate(f'ID Client:\n{index_client}', xy=(feature, df[feature].loc[index_client] + 0.1), xytext=(0, 40), textcoords='offset points', ha='center', va='bottom', bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
                legend_handles, _ = figInd.get_legend_handles_labels()
                figInd.legend(legend_handles, ['Non', 'Oui'], title="Défaut de paiement")
                st.pyplot(figInd.figure)
                plt.close()

            else:
                figInd = sns.boxplot(data=df_reference, y=feature, x='TARGET', showfliers=False, width=0.2)
                plt.xlabel('Défaut de paiement')
                figInd.set_xticklabels(["Non", "Oui"])
                plt.scatter(y=df[feature].loc[index_client], x=0.5, marker='o', s=100, color="r")
                figInd.annotate(f'ID Client:\n{index_client}', xy=(0.5, df[feature].loc[index_client]), xytext=(0, 40), textcoords='offset points', ha='center', va='bottom', bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
                figInd.axhline(y=df_reference[df_reference['TARGET'] == 0][feature].mean(), zorder=0, linestyle='--', color="#1f77b4")
                figInd.axhline(y=df_reference[df_reference['TARGET'] == 1][feature].mean(), zorder=0, linestyle='--', color="#ff7f0e")
                colors = ["#1f77b4", "#ff7f0e"]
                lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='--') for c in colors]
                labels = ["Moyenne sans défaut", "Moyenne avec défaut"]
                plt.legend(lines, labels, title="Valeur moyenne des clients:")
                st.pyplot(figInd.figure)
                plt.close()

# Fonction pour détecter et afficher les colonnes avec des valeurs manquantes
def nan_values(df, index_client=0):
    """
    Cette fonction détecte les colonnes avec des valeurs manquantes pour un client spécifique et affiche les résultats dans Streamlit.

    Entrée :
    df : dataframe pandas avec les 62 caractéristiques
    index_client : index du client

    Sortie :
    Affiche un tableau des colonnes avec des valeurs manquantes (s'il y en a) ou un message indiquant qu'il n'y a pas de valeurs manquantes.
    """

    # Vérifie s'il existe des valeurs manquantes pour le client
    if np.isnan(df.loc[index_client]).any():
        st.subheader('Avertissements : Colonnes avec des valeurs manquantes')
        nan_col = [features for features in list(df.columns) if np.isnan(df.loc[index_client][features])]

        col1, col2 = st.columns(2)
        with col1:
            st.table(data=pd.DataFrame(nan_col, columns=['CARACTÉRISTIQUES AVEC DES VALEURS MANQUANTES']))
        with col2:
            st.write('Toutes les valeurs manquantes ont été remplacées par 0.')
    else:
        st.subheader("Il n'y a pas de valeurs manquantes.")
