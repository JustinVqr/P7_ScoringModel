import streamlit as st

# Ajoutez le chemin du répertoire racine au sys.path pour que Python trouve les modules dans 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.component.P7_App_FrontEnd import *

# Configuration de la page Streamlit
st.set_page_config(page_title="3) Nouvelle Prédiction")  # Titre de la page de l'application

# Option pour supprimer les avertissements de dépréciation concernant pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

#
# --- Récupération des données depuis la page d'accueil ---
#

df_train = st.session_state.df_train  # DataFrame principal contenant les données d'entraînement
df_new = st.session_state.df_new  # DataFrame contenant de nouvelles données pour la prédiction
Credit_clf_final = st.session_state.Credit_clf_final  # Modèle final de classification
explainer = st.session_state.explainer  # Explicateur du modèle pour l'interprétation des résultats

#
# --- Création de deux onglets ---
#

# Création de deux onglets dans l'application : "ID client" et "Information manuelle"
tab1, tab2 = st.tabs(["ID client", "Information manuelle"])

# --- Onglet 1 : Prédiction pour un client avec un ID ---
with tab1:
    st.header("Prédiction pour un client avec ID")
    
    # Champ d'entrée pour l'ID du client
    index_client = st.number_input(
        "Entrez l'ID du client (ex : 100001, 100005)",  # Message d'invite pour entrer l'ID du client
        format="%d",  # Format de l'entrée (entier)
        value=100001  # Valeur par défaut pour l'ID du client
    )

    # Bouton pour lancer la prédiction en utilisant une API (FastAPI)
    run_btn = st.button(
        'Prédire',  # Titre du bouton
        on_click=None,
        type="primary",
        key='predict_btn1'
    )
    
    # Si le bouton est cliqué
    if run_btn:
        # Vérification si l'ID du client est présent dans les nouvelles données
        if index_client in set(df_new.index):
            # Récupération des données du client et traitement des valeurs manquantes
            data_client = df_new.loc[index_client].fillna(0).to_dict()
            
            # Exécution de l'appel à l'API pour faire la prédiction
            execute_API(data_client)
            
            # Affichage des résultats pour ce client avec les graphiques explicatifs
            plot_client(
                df_new,
                explainer,
                df_reference=df_train,
                index_client=index_client
            )
            
            # Affichage des valeurs manquantes pour ce client
            nan_values(df_new, index_client=index_client)
        else:
            # Si l'ID du client n'est pas trouvé dans la base de données, affichage d'un message
            st.write("Client non trouvé dans la base de données")

# --- Onglet 2 : Prédiction pour un nouveau client sans ID ---

# Ce second onglet présente trois options pour entrer des données pour la prédiction :
# 1. Manuellement avec des champs d'entrée numériques
# 2. Manuellement avec un texte pré-formaté (comme un dictionnaire)
# 3. En chargeant un fichier CSV

with tab2:
    st.header("Prédiction pour un nouveau client")

    # Sélection du mode de saisie des données (Manuel, Texte, ou CSV)
    option = st.selectbox(
        'Comment souhaitez-vous entrer les données ?',
        ('Manuel', 'Texte', 'Fichier CSV')
    )

    # 1ère option : Entrée manuelle des données avec des champs d'entrée numériques
    if option == 'Manuel':
        with st.expander("Cliquez pour entrer les données manuellement"):
            data_client = {}
            for features in list(df_new.columns):
                # Détection des types de données pour générer les champs d'entrée numériques correspondants
                if df_train[features].dtype == np.int64:
                    # Pour les entiers, les valeurs minimales et maximales sont définies selon les valeurs présentes dans le DataFrame d'entraînement
                    min_values = df_train[features].min().astype('int')
                    max_values = df_train[features].max().astype('int')
                    data_client["{0}".format(features)] = st.number_input(
                        str(features), min_value=min_values, max_value=max_values, step=1)
                else:
                    # Pour les flottants, les valeurs minimales et maximales sont définies selon les valeurs présentes dans le DataFrame d'entraînement
                    min_values = df_train[features].min().astype('float')
                    max_values = df_train[features].max().astype('float')
                    data_client["{0}".format(features)] = st.number_input(
                        str(features), min_value=min_values, max_value=max_values, step=0.1)

    # 2ème option : Entrée manuelle des données sous forme de texte (dictionnaire)
    elif option == 'Texte':
        with st.expander("Cliquez pour entrer les données sous forme de texte"):
            # Zone de texte pour entrer les données sous forme de dictionnaire pré-formaté
            data_client = st.text_area('Entrez les données sous forme de dictionnaire',
                                       '''{"FLAG_OWN_CAR": 0,
                "AMT_CREDIT": 0,
                ...,
                "INSTAL_DAYS_ENTRY_PAYMENT_MEAN": 0,
                "CC_CNT_DRAWINGS_CURRENT_MEAN": 0,
                "CC_CNT_DRAWINGS_CURRENT_VAR": 0
                }''')

            # Conversion du texte en un dictionnaire Python
            data_client = json.loads(data_client)

    # 3ème option : Chargement des données à partir d'un fichier CSV
    else:
        # Chargement d'un fichier CSV avec un séparateur ";" et deux colonnes (nom de la caractéristique et valeur)
        loader = st.file_uploader(" ")
        if loader is not None:
            # Chargement du fichier et transformation en dictionnaire
            data_client = pd.read_csv(
                loader,
                sep=";",
                index_col=0,
                header=None
            ).squeeze(1).to_dict()

    # Bouton pour lancer la prédiction en utilisant l'API (FastAPI)
    run_btn2 = st.button(
        'Prédire',  # Titre du bouton
        on_click=None,
        type="primary",
        key='predict_btn2'
    )
    
    # Si le bouton est cliqué
    if run_btn2:
        # Exécution de l'appel à l'API pour faire la prédiction
        execute_API(data_client)
        
        # Conversion des données client en DataFrame pour l'affichage
        data_client = pd.DataFrame(data_client, index=[0])
        
        # Affichage des résultats avec les graphiques explicatifs
        plot_client(
            data_client,
            explainer,
            df_reference=df_train,
            index_client=0  # Utilisation d'un index fictif (0) pour un nouveau client
        )
        
        # Affichage des valeurs manquantes pour ce nouveau client
        nan_values(data_client, index_client=0)
