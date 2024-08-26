import streamlit as st
import pandas as pd
import shap
import joblib
import os
import requests
from io import StringIO

# Configuration de la page d'accueil
st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title="Accueil"
)

# Système de navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisissez une page", ["Accueil", "Analyse des clients", "Prédiction"])

# Page d'accueil
if page == "Accueil":
    st.title("Prêt à dépenser")
    st.subheader("Application de support à la décision pour l'octroi de prêts")
    st.write("""Cette application assiste le chargé de prêt dans sa décision d'octroyer un prêt à un client.""")

    col1, col2 = st.columns(2)

    with col1:
        st.image("https://raw.githubusercontent.com/JustinVqr/P7_ScoringModel/main/app/images/logo.png")

    with col2:
        st.subheader("Contenu de l'application :")
        st.markdown("""
        Cette application comporte trois pages :
        1) Informations générales sur la base de données et le modèle
        2) Analyse des clients connus
        3) Prédiction des défauts de paiement pour de nouveaux clients via une API
        """)

    st.subheader("Chargement de l'application :")

    with st.spinner('Initialisation...'):
        @st.cache_data
        def load_client_data(client_id):
            # Charger uniquement les données du client spécifique
            test_url = f"https://www.dropbox.com/scl/fi/5ihks3eng4kws0zfqwdkw/df_new.csv?rlkey=m2kd4iv4abb67w546p75ulrgk&st=dj744do8&dl=1"
            
            # Télécharger et filtrer les données pour un seul client
            def download_and_load_client_csv(url, client_id):
                try:
                    response = requests.get(url)
                    if response.status_code != 200:
                        st.error(f"Échec du téléchargement depuis {url}. Statut: {response.status_code}")
                        return None
                    csv_data = StringIO(response.text)
                    df_new = pd.read_csv(csv_data, sep=',', index_col="SK_ID_CURR", encoding='utf-8')
                    # Sélectionner uniquement les données du client spécifique
                    if client_id in df_new.index:
                        return df_new.loc[[client_id]]
                    else:
                        st.error("Client ID non trouvé.")
                        return None
                except Exception as e:
                    st.error(f"Erreur lors du téléchargement ou du chargement des données depuis {url} : {e}")
                    return None

            return download_and_load_client_csv(test_url, client_id)

        # Saisie de l'ID du client
        sk_id_curr = st.text_input("Entrez l'ID du client pour obtenir la prédiction :")

        # Bouton pour lancer la prédiction
        if st.button("Obtenir la prédiction"):
            if sk_id_curr:
                try:
                    client_id = int(sk_id_curr)
                    df_client = load_client_data(client_id)

                    if df_client is not None:
                        st.write("Données du client chargées avec succès.")

                        # Charger le modèle
                        model_path = os.path.join(os.getcwd(), 'app', 'model', 'best_model.pkl')
                        if os.path.exists(model_path):
                            Credit_clf_final = joblib.load(model_path)
                            st.write("Modèle chargé avec succès.")

                            # Faire la prédiction pour le client
                            X_client = df_client.drop(columns="TARGET").fillna(0)
                            prediction_proba = Credit_clf_final.predict_proba(X_client)[:, 1]
                            prediction = Credit_clf_final.predict(X_client)

                            # Afficher les résultats
                            st.write(f"Prédiction : {'Oui' if prediction[0] == 1 else 'Non'}")
                            st.write(f"Probabilité de défaut : {prediction_proba[0] * 100:.2f}%")

                            # Calculer et afficher les valeurs SHAP uniquement pour ce client
                            explainer = shap.KernelExplainer(Credit_clf_final.predict_proba, X_client)
                            shap_values = explainer.shap_values(X_client)
                            st.write("Valeurs SHAP calculées.")
                            shap.initjs()
                            shap.force_plot(explainer.expected_value[1], shap_values[1], X_client, matplotlib=True)
                            st.pyplot(bbox_inches='tight')

                        else:
                            st.error(f"Le fichier {model_path} n'existe pas.")

                except Exception as e:
                    st.error(f"Erreur lors de la requête de prédiction : {e}")
            else:
                st.error("Veuillez entrer un ID client valide.")
