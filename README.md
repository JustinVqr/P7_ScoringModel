# P7_ScoringModel
Répertoire dédié à l'application de scoring dans le cadre du projet OC.
Projet MLops de Scoring avec LightGBM
Description du projet
Ce projet vise à développer un pipeline de machine learning (ML) complet pour un modèle de scoring, utilisant LightGBM (LGBM) pour la modélisation. En plus de la partie machine learning, une interface utilisateur (UI) a été développée avec une application web avec une API, "FAST API", pour faciliter l'accès et l'utilisation du modèle de scoring par des utilisateurs non techniques, en l'occurence les chargés de clientèle de la banque Prêt-à-dépenser.

Le projet suit une approche MLops pour assurer une intégration et une livraison continues (CI/CD), des tests automatisés, la gestion des données, et le suivi des performances du modèle.

Contenu du projet
Le repository est structuré de la manière suivante :

app/ : Contient l'application web pour l'interface utilisateur. Cette interface permet de soumettre des données et de recevoir des scores en retour.
data/ : Données brutes ou traitées utilisées pour entraîner le modèle LightGBM.
notebooks/ : Jupyter notebooks pour l'exploration de données, la modélisation, et les expérimentations préliminaires avec LightGBM.
présentation/ : Documentation et diapositives pour présenter le projet.
pytests/ : Scripts de tests automatisés (Pytest) pour s'assurer que les différentes fonctionnalités de l'application fonctionnent correctement.
scripts/ : Contient les scripts pour l'entraînement du modèle, la gestion des données et le scoring.
requirements.txt : Liste des dépendances pour exécuter le projet.
setup.sh : Script de configuration pour préparer l'environnement de développement.
Procfile : Fichier de configuration pour le déploiement sur des plateformes comme Heroku.
