import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

# Configuration de la page (titre de l'application)
st.set_page_config(page_title="1) General information")

#
# --- Récupération des données depuis la page d'accueil ---
#

# Chargement des données depuis l'état de session
df_train = st.session_state.df_train  # DataFrame principal contenant les données des clients
Credit_clf_final = st.session_state.Credit_clf_final  # Modèle final de classification
explainer = st.session_state.explainer  # Variable explicative (potentiellement utilisée pour interpréter le modèle)

#
# --- Création de la mise en page de la page (3 onglets) ---
#

# Création de trois onglets pour l'application : "Data", "Indicators", et "Model"
tab1, tab2, tab3 = st.tabs(["Data", "Indicators", "Model"])

# --- Onglet 1 : Présentation du dataframe (contenu, cible, valeurs manquantes) ---
with tab1:

    # Mise en page de l'onglet
    st.header("Aperçu du Dataframe")
    st.subheader("Contenu du dataframe")

    # Nombre de clients enregistrés :
    col1, col2 = st.columns(2)
    # Affichage du nombre total de clients connus
    col1.metric("Nombre de clients enregistrés", df_train.shape[0])
    # Affichage du nombre de caractéristiques (colonnes) sans la colonne TARGET
    col2.metric("Nombre de caractéristiques des clients", df_train.drop(columns='TARGET').shape[1])

    # Analyse de la cible : Diagramme en anneau (Donut chart)
    st.subheader("Analyse de la cible")
    fig1, ax = plt.subplots()
    # Création d'un diagramme circulaire représentant la répartition des classes dans la colonne TARGET
    ax.pie(df_train.TARGET.value_counts(normalize=True),
           labels=["0", "1"],
           autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, (p / 100) * sum(df_train.TARGET.value_counts())),
           startangle=0,
           pctdistance=0.8,
           explode=(0.05, 0.05))  # Séparation des segments du diagramme pour une meilleure visibilité
    # Ajout d'un cercle au centre pour faire un effet d'anneau
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    plt.gca().add_artist(centre_circle)
    plt.axis('equal')  # Assure que le diagramme est dessiné de manière circulaire
    plt.title('Répartition des clients ayant des difficultés (1) ou non (0)\nà rembourser le prêt')
    plt.tight_layout()
    plt.close()
    st.pyplot(fig1)  # Affichage du diagramme en anneau dans Streamlit

    # Analyse des valeurs manquantes (avec missingno)
    st.subheader("Analyse des valeurs manquantes")
    with st.spinner('Chargement du graphique...'):
        # Utilisation de missingno pour visualiser les valeurs manquantes dans le dataframe (sans la colonne TARGET)
        figNAN = msno.matrix(df_train.drop(columns='TARGET'), labels=True, sort="ascending")
        plt.close()
        st.pyplot(figNAN.figure)  # Affichage du graphique de missingno dans Streamlit

        # Informations supplémentaires sur la gestion des valeurs manquantes
        st.markdown("""
        Informations sur les valeurs manquantes :
        1) Les variables avec plus de 80% de valeurs manquantes ont été supprimées
        2) Toutes les valeurs manquantes restantes ont été remplacées par 0
        """)

# --- Onglet 2 : Présentation des caractéristiques ---
# Les 62 caractéristiques sont présentées en deux colonnes avec 31 graphiques chacune.
# Pour les caractéristiques binaires : diagramme en barres montrant la fréquence des valeurs par classe (TARGET)
# Pour les autres caractéristiques : Box plot coloré par classe (TARGET)
with tab2:
    cola, colb = st.columns(2)  # Création de deux colonnes pour l'affichage des graphiques
    with cola:
        for features in list(df_train.drop(columns='TARGET').columns)[:31]:  # Affichage des 31 premières caractéristiques
            if df_train[features].nunique() == 2:  # Si la caractéristique est binaire (deux valeurs distinctes)
                # Diagramme en barres pour les caractéristiques binaires
                figInd = sns.barplot(df_train[['TARGET', features]].fillna(0).groupby('TARGET').value_counts(normalize=True).reset_index(),
                                     x=features, y=0, hue="TARGET")
                plt.ylabel('Fréquence des clients')
                legend_handles, _= figInd.get_legend_handles_labels()
                figInd.legend(legend_handles, ['Non', 'Oui'], title="DÉFAUT DE PAIEMENT")
                figInd.set_xticklabels(["Non", "Oui"])
                plt.close()
                st.pyplot(figInd.figure)  # Affichage du graphique dans Streamlit
            else:
                # Box plot pour les autres types de caractéristiques
                figInd = sns.boxplot(data=df_train, y=features, x='TARGET', showfliers=False)
                plt.xlabel('DÉFAUT DE PAIEMENT')
                figInd.set_xticklabels(["Non", "Oui"])
                plt.close()
                st.pyplot(figInd.figure)  # Affichage du box plot dans Streamlit

    with colb:
        for features in list(df_train.drop(columns='TARGET').columns)[31:]:  # Affichage des 31 caractéristiques suivantes
            if df_train[features].nunique() == 2:  # Si la caractéristique est binaire
                # Même procédure que dans la première colonne pour les caractéristiques binaires
                figInd = sns.barplot(df_train[['TARGET', features]].fillna(0).groupby('TARGET').value_counts(normalize=True).reset_index(),
                                     x=features, y=0, hue="TARGET")
                plt.ylabel('Fréquence des clients')
                figInd.set_xticklabels(["Non", "Oui"])
                legend_handles, _= figInd.get_legend_handles_labels()
                figInd.legend(legend_handles, ['Non', 'Oui'], title="DÉFAUT DE PAIEMENT")
                plt.close()
                st.pyplot(figInd.figure)  # Affichage du graphique dans Streamlit
            else:
                # Même procédure que dans la première colonne pour les autres types de caractéristiques
                figInd = sns.boxplot(data=df_train, y=features, x='TARGET', showfliers=False)
                plt.xlabel('DÉFAUT DE PAIEMENT')
                figInd.set_xticklabels(["Non", "Oui"])
                plt.close()
                st.pyplot(figInd.figure)  # Affichage du box plot dans Streamlit

# --- Onglet 3 : Présentation du modèle ---
with tab3:
    st.header("Description du modèle")

    # Importance des caractéristiques
    st.subheader("Importance des caractéristiques du modèle LightGBM")
    st.image("image/Plot_importance.png")  # Affichage d'une image de l'importance des caractéristiques

    # Paramètres optimisés du modèle
    st.subheader("Paramètres (optimisés avec Optuna)")
    st.table(pd.DataFrame.from_dict(Credit_clf_final.get_params(), orient='index', columns=['Paramètre']))  # Affichage des paramètres du modèle sous forme de tableau

    # Scores obtenus par validation croisée
    st.subheader("Scores obtenus par validation croisée")
    st.write(pd.DataFrame({
        'Metrics': ["AUC", "Accuracy", "F1", "Precision", "Recall", "Profit"],
        'Valeur': [0.764, 0.869, 0.311, 0.271, 0.366, 1.928],
    }))  # Affichage des scores sous forme de tableau

    # Matrice de confusion et courbe ROC sur le jeu de test
    st.subheader("Courbe ROC et matrice de confusion sur un jeu de test")
    col1, col2 = st.columns(2)
    with col1:
        st.image("image/Test_ROC_AUC.png")  # Affichage de la courbe ROC
    with col2:
        st.image("image/Test_confusion_matrix.png")  # Affichage de la matrice de confusion
