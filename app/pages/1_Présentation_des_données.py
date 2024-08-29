import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

# Configuration de la page (titre de l'application)
st.set_page_config(page_title="1) Présentation des données")

# --- Vérification et récupération des données depuis la session ---
if "df_train" not in st.session_state or "Credit_clf_final" not in st.session_state or "explainer" not in st.session_state:
    st.error("Les données nécessaires ne sont pas disponibles dans la session. Veuillez vous assurer que le modèle et les données sont chargés depuis la page d'accueil.")
    st.stop()

# Chargement des données depuis l'état de session
df_train = st.session_state.df_train
Credit_clf_final = st.session_state.Credit_clf_final
explainer = st.session_state.explainer

# --- Création de la mise en page de la page (3 onglets) ---
tab1, tab2, tab3 = st.tabs(["Data", "Indicators", "Model"])

# --- Onglet 1 : Présentation du dataframe ---
with tab1:
    st.header("Aperçu du Dataframe")
    st.subheader("Contenu du dataframe")

    col1, col2 = st.columns(2)
    col1.metric("Nombre de clients enregistrés", df_train.shape[0])
    col2.metric("Nombre de caractéristiques des clients", df_train.drop(columns='TARGET').shape[1])

    # Analyse de la cible : Diagramme en anneau
    st.subheader("Analyse de la cible")
    fig1, ax = plt.subplots()
    ax.pie(df_train.TARGET.value_counts(normalize=True),
           labels=["0", "1"],
           autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, (p / 100) * sum(df_train.TARGET.value_counts())),
           startangle=0,
           pctdistance=0.8,
           explode=(0.05, 0.05))
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    plt.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.title('Répartition des clients ayant des difficultés (1) ou non (0) à rembourser le prêt')
    plt.tight_layout()
    st.pyplot(fig1)

    # Analyse des valeurs manquantes
    st.subheader("Analyse des valeurs manquantes")
    with st.spinner('Chargement du graphique...'):
        figNAN = msno.matrix(df_train.drop(columns='TARGET'), labels=True, sort="ascending")
        st.pyplot(figNAN.figure)
        st.markdown("""
        **Informations sur les valeurs manquantes :**
        1) Les variables avec plus de 80% de valeurs manquantes ont été supprimées.
        2) Toutes les valeurs manquantes restantes ont été remplacées par 0.
        """)

# --- Onglet 2 : Présentation des caractéristiques ---
with tab2:
    st.subheader("Présentation des caractéristiques")
    
    cola, colb = st.columns(2)
    
    # Affichage des 31 premières caractéristiques dans la première colonne
    with cola:
        for features in list(df_train.drop(columns='TARGET').columns)[:31]:
            if df_train[features].nunique() == 2:
                # Diagramme en barres pour les caractéristiques binaires
                figInd = sns.barplot(df_train[['TARGET', features]].fillna(0).groupby('TARGET').value_counts(normalize=True).reset_index(),
                                     x=features, y=0, hue="TARGET")
                figInd.set_xticklabels(["Non", "Oui"])
                plt.close()
                st.pyplot(figInd.figure)
            else:
                # Box plot pour les autres caractéristiques
                figInd = sns.boxplot(data=df_train, y=features, x='TARGET', showfliers=False)
                plt.close()
                st.pyplot(figInd.figure)

    # Affichage des caractéristiques suivantes dans la deuxième colonne
    with colb:
        for features in list(df_train.drop(columns='TARGET').columns)[31:]:
            if df_train[features].nunique() == 2:
                figInd = sns.barplot(df_train[['TARGET', features]].fillna(0).groupby('TARGET').value_counts(normalize=True).reset_index(),
                                     x=features, y=0, hue="TARGET")
                figInd.set_xticklabels(["Non", "Oui"])
                plt.close()
                st.pyplot(figInd.figure)
            else:
                figInd = sns.boxplot(data=df_train, y=features, x='TARGET', showfliers=False)
                plt.close()
                st.pyplot(figInd.figure)

# --- Onglet 3 : Présentation du modèle ---
with tab3:
    st.header("Description du modèle")

    # Importance des caractéristiques
    st.subheader("Importance des caractéristiques du modèle LightGBM")
    st.image("../images/Plot_importance.png")

    # Paramètres optimisés du modèle
    st.subheader("Paramètres (optimisés avec Optuna)")
    if hasattr(Credit_clf_final, 'get_params'):
        st.table(pd.DataFrame.from_dict(Credit_clf_final.get_params(), orient='index', columns=['Paramètre']))
    else:
        st.warning("Les paramètres du modèle ne sont pas disponibles.")

    # Scores obtenus par validation croisée
    st.subheader("Scores obtenus par validation croisée")
    st.write(pd.DataFrame({
        'Metrics': ["AUC", "Accuracy", "F1", "Precision", "Recall", "Profit"],
        'Valeur': [0.764, 0.869, 0.311, 0.271, 0.366, 1.928],
    }))

    # Matrice de confusion et courbe ROC sur le jeu de test
    st.subheader("Courbe ROC et matrice de confusion sur un jeu de test")
    col1, col2 = st.columns(2)
    with col1:
        st.image("../images/Test_ROC_AUC.png")
    with col2:
        st.image("../images/Test_confusion_matrix.png")
