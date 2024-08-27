import streamlit as st
import pandas as pd
import requests
from io import StringIO
import os
import joblib
import shap
import matplotlib.pyplot as plt


# Configuration de la page d'accueil
st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title="Accueil"
)

# Système de navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisissez une page", ["Accueil", "Analyse des clients", "Prédiction"])

# --- Initialisation de l'état de session ---
if "load_state" not in st.session_state:
    st.session_state.load_state = False

# --- Fonction pour télécharger et charger les données depuis Dropbox ---
@st.cache_data
def load_data():
    url_train = "https://www.dropbox.com/scl/fi/59fn2h9mapw69flpnccz6/df_train.csv?rlkey=dq6qvlj4dxnswqdegyadjfnqs&st=snvt2aue&dl=1"
    url_new = "https://www.dropbox.com/scl/fi/2mylh9bshf5jkzg6n9m7t/df_new.csv?rlkey=m82n87j6hr9en1utkt7a8qsv4&st=k6kj1pm5&dl=1"
    
    response_train = requests.get(url_train)
    response_new = requests.get(url_new)
    
    if response_train.status_code == 200 and response_new.status_code == 200:
        df_train = pd.read_csv(StringIO(response_train.text), sep=',', encoding='utf-8')
        df_new = pd.read_csv(StringIO(response_new.text), sep=',', index_col="SK_ID_CURR", encoding='utf-8')
        return df_train, df_new
    else:
        st.error(f"Erreur de téléchargement : Statut {response_train.status_code}, {response_new.status_code}")
        return None, None


# --- Chargement des ressources au démarrage ---

def load_model_and_explainer(df_train):
    model_path = os.path.join(os.getcwd(), 'app', 'model', 'best_model.pkl')
    
    if os.path.exists(model_path):
        try:
            # Utilisez joblib pour charger le modèle
            Credit_clf_final = joblib.load(model_path)
            st.write("Modèle chargé avec succès.")
            
            # Tentons d'utiliser TreeExplainer, si une erreur survient, basculons sur KernelExplainer
            try:
                st.write("Tentative de création de l'explicateur SHAP avec TreeExplainer...")
                explainer = shap.TreeExplainer(Credit_clf_final, df_train.drop(columns="TARGET").fillna(0))
            except Exception as e:
                st.write("TreeExplainer non compatible, utilisation de KernelExplainer à la place.")
                explainer = shap.KernelExplainer(Credit_clf_final.predict, df_train.drop(columns="TARGET").fillna(0))
            
            return Credit_clf_final, explainer
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle ou de l'explicateur : {e}")
            return None, None
    else:
        st.error(f"Le fichier {model_path} n'existe pas.")
        return None, None

# Utilisation dans l'application Streamlit
if not st.session_state.get("load_state"):
    df_train, df_new = load_data()  # Assurez-vous que les données sont chargées
    if df_train is not None and df_new is not None:
        # Chargement du modèle et de l'explicateur SHAP
        Credit_clf_final, explainer = load_model_and_explainer(df_train)
        
        if Credit_clf_final and explainer:
            st.session_state.Credit_clf_final = Credit_clf_final
            st.session_state.explainer = explainer
            st.session_state.df_train = df_train
            st.session_state.df_new = df_new
            st.session_state.load_state = True
            st.success("Modèle et explicateur SHAP chargés avec succès.")
        else:
            st.error("Échec du chargement du modèle ou de l'explicateur.")
else:
    Credit_clf_final = st.session_state.Credit_clf_final
    explainer = st.session_state.explainer
    df_train = st.session_state.df_train
    df_new = st.session_state.df_new

if page == "Prédiction":
    # Entrez l'ID du client
    sk_id_curr = st.text_input("Entrez l'ID du client pour obtenir la prédiction :")
    
    if st.button("Obtenir la prédiction"):
        if sk_id_curr and Credit_clf_final and explainer:
            try:
                client_id = int(sk_id_curr)
                
                if client_id in df_new.index:
                    df_client = df_new.loc[[client_id]]
                    X_client = df_client.fillna(0)

                    # Faire la prédiction
                    prediction_proba = Credit_clf_final.predict_proba(X_client)[:, 1]
                    prediction = Credit_clf_final.predict(X_client)

                    # Afficher les résultats
                    st.write(f"Prédiction : {'Oui' si prediction[0] == 1 else 'Non'}")
                    st.write(f"Probabilité de défaut : {prediction_proba[0] * 100:.2f}%")

                    # Calculer les valeurs SHAP pour ce client
                    shap_values = explainer.shap_values(X_client)
                    st.write("Valeurs SHAP calculées.")
                    shap.initjs()

                    # Utilisation correcte de shap.force_plot
                    # Pour un classificateur binaire, shap_values est une liste, on accède à la classe positive
                    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                    shap.force_plot(expected_value, shap_values[1][0], X_client, matplotlib=True)
                    st.pyplot(bbox_inches='tight')

                else:
                    st.error("Client ID non trouvé.")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
        else:
            st.error("Veuillez entrer un ID client valide et assurez-vous que le modèle est chargé.")

