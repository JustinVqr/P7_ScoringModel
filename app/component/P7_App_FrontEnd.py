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
    """ This function generates a plot of the main shap value.
    It helps to understand the prediction on loan default for a specific client.


    input :
    explainer > the shap explainer
    df > pandas dataframe with the 62 features and their value
    index_client > index of the client

    output :
    Bar plot of the shap values, integrated in a streamlit figure
    """

    # Plot shap values
    # Default SHAP colors
    default_pos_color = "#ff0051"
    default_neg_color = "#008bfb"

    # Custom colors
    negative_color = "#1f77b4"
    positive_color = "#ff7f0e"

    fig_shap = shap.plots.bar(
        explainer(
            df.fillna(0).loc[index_client]),
        show=False)
    # Change the colormap of the artists
    for fc in plt.gcf().get_children():
        # Ignore last Rectangle
        for fcc in fc.get_children()[:-1]:
            if (isinstance(fcc, matplotlib.patches.Rectangle)):
                if (matplotlib.colors.to_hex(
                        fcc.get_facecolor()) == default_pos_color):
                    fcc.set_facecolor(positive_color)
                elif (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color):
                    fcc.set_color(negative_color)
            elif (isinstance(fcc, plt.Text)):
                if (matplotlib.colors.to_hex(
                        fcc.get_color()) == default_pos_color):
                    fcc.set_color(positive_color)
                elif (matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color):
                    fcc.set_color(negative_color)
    st.pyplot(fig_shap)
    plt.clf()


def plot_client(df, explainer, df_reference, index_client=0):
    """ This function generates all the different plots to understand the prediction of loan default for a specific client
    First, this function call the shap_plot function to generate the explainer plot.
    then, it generate 6 plots (for the 6 main features affecting the prediction)

    - for binary features (0/1), the plot is a barplot of the train dataframe showing the frequency (feature) within each class (TARGET)
    - else : boxplots of the feature, with the client value (red dot) and vertical dashed lines showing the mean of each class (TARGET)


    input :
    df > pandas dataframe with the 62 features and their value
    explainer > the shap explainer
    df_reference > the training dataset used as a baseline for plots
    index_client > index of the client

    output :
    1) the shap bar plot
    2) 6 plots for the 6 most discriminative features
    """

    # ---Bar plot of the shap value for a specific client ---
    shap_plot(explainer, df, index_client)

    # --- Calcul of the shap_importance ---
    shap_values = explainer.shap_values(df.fillna(0).loc[index_client])
    shap_importance = pd.Series(
        shap_values,
        df.columns).abs().sort_values(
        ascending=False)

    # --- 6 discriminative features ---
    st.subheader('Explaination : Top 6 discriminative features')

    # The figures are divided in two streamlit columns, with 3 plots per
    # columns
    col1, col2 = st.columns(2)
    with col1:
        for features in list(shap_importance.index[:6])[:3]:
            plt.figure(figsize=(5, 5))

            # For binary features :
            if df_reference[features].nunique() == 2:

                # Bar plot sof the frequency per class :
                figInd = sns.barplot(df_reference[['TARGET', features]].fillna(0).groupby(
                    'TARGET').value_counts(normalize=True).reset_index(), x=features, y=0, hue='TARGET')
                plt.ylabel('Freq of client')

                # Addition of the client data (+ box with client ID):
                plt.scatter(
                    y=df[features].loc[index_client] +
                    0.1,
                    x=features,
                    marker='o',
                    s=100,
                    color="r")
                figInd.annotate(
                    'Client ID:\n{}'.format(index_client), xy=(
                        features, df[features].loc[index_client] + 0.1), xytext=(
                        0, 40), textcoords='offset points', ha='center', va='bottom', bbox=dict(
                        boxstyle="round", fc="w"), arrowprops=dict(
                        arrowstyle="->"))
                legend_handles, _= figInd.get_legend_handles_labels()
                figInd.legend(legend_handles,['No','Yes'], title="LOAN DEFAULT")
                st.pyplot(figInd.figure)
                plt.close()

            # For non binary features :
            else:
                figInd = sns.boxplot(
                    data=df_reference,
                    y=features,
                    x='TARGET',
                    showfliers=False,
                    width=0.2)
                plt.xlabel('LOAN DEFAULT')
                figInd.set_xticklabels(["No", "Yes"])
                                # Addition of the client data (+ box with client ID):
                plt.scatter(y=df[features].loc[index_client],
                            x=0.5, marker='o', s=100, color="r")

                figInd.annotate(
                    'Client ID:\n{}'.format(index_client), xy=(
                        0.5, df[features].loc[index_client]), xytext=(
                        0, 40), textcoords='offset points', ha='center', va='bottom', bbox=dict(
                        boxstyle="round", fc="w"), arrowprops=dict(
                        arrowstyle="->"))

                # Addition of mean of each class + client:
                figInd.axhline(y=df_reference[df_reference['TARGET'] == 0][features].mean(
                ), zorder=0, linestyle='--', color="#1f77b4")
                figInd.axhline(y=df_reference[df_reference['TARGET'] == 1][features].mean(
                ), zorder=0, linestyle='--', color="#ff7f0e")

                # Custom legend:
                colors = ["#1f77b4", "#ff7f0e"]
                lines = [
                    Line2D(
                        [0],
                        [0],
                        color=c,
                        linewidth=1,
                        linestyle='--') for c in colors]
                labels = ["No Loan Default", "Loan Default"]
                plt.legend(lines, labels, title="Mean value of clients:")
                st.pyplot(figInd.figure)
                plt.close()

    with col2:
        # Plot top 6 discriminant features
        for features in list(shap_importance.index[:6])[3:]:
            plt.figure(figsize=(5, 5))

            # For binary features :
            if df_reference[features].nunique() == 2:

                # Bar plot sof the frequency per class :
                figInd = sns.barplot(df_reference[['TARGET', features]].fillna(0).groupby(
                    'TARGET').value_counts(normalize=True).reset_index(), x=features, y=0, hue='TARGET')
                plt.ylabel('Freq of client')

                # Addition of the client data (+ box with client ID):
                plt.scatter(
                    y=df[features].loc[index_client] +
                    0.1,
                    x=features,
                    marker='o',
                    s=100,
                    color="r")
                figInd.annotate(
                    'Client ID:\n{}'.format(index_client), xy=(
                        features, df[features].loc[index_client] + 0.1), xytext=(
                        0, 40), textcoords='offset points', ha='center', va='bottom', bbox=dict(
                        boxstyle="round", fc="w"), arrowprops=dict(
                        arrowstyle="->"))
                legend_handles, _= figInd.get_legend_handles_labels()
                figInd.legend(legend_handles,['No','Yes'], title="LOAN DEFAULT")
                st.pyplot(figInd.figure)
                plt.close()

            # For non binary features :
            else:
                figInd = sns.boxplot(
                    data=df_reference,
                    y=features,
                    x='TARGET',
                    showfliers=False,
                    width=0.2)
                plt.xlabel('LOAN DEFAULT')
                figInd.set_xticklabels(["No", "Yes"])
                                # Addition of the client data (+ box with client ID):
                plt.scatter(y=df[features].loc[index_client],
                            x=0.5, marker='o', s=100, color="r")

                figInd.annotate(
                    'Client ID:\n{}'.format(index_client), xy=(
                        0.5, df[features].loc[index_client]), xytext=(
                        0, 40), textcoords='offset points', ha='center', va='bottom', bbox=dict(
                        boxstyle="round", fc="w"), arrowprops=dict(
                        arrowstyle="->"))

                # Addition of mean of each class + client:
                figInd.axhline(y=df_reference[df_reference['TARGET'] == 0][features].mean(
                ), zorder=0, linestyle='--', color="#1f77b4")
                figInd.axhline(y=df_reference[df_reference['TARGET'] == 1][features].mean(
                ), zorder=0, linestyle='--', color="#ff7f0e")

                # Custom legend:
                colors = ["#1f77b4", "#ff7f0e"]
                lines = [
                    Line2D(
                        [0],
                        [0],
                        color=c,
                        linewidth=1,
                        linestyle='--') for c in colors]
                labels = ["No Loan Default", "Loan Default"]
                plt.legend(lines, labels, title="Mean value of clients:")
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
