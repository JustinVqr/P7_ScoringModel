import pickle
import numpy as np

# Charger le modèle sauvegardé
with open('app/model/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

def faire_prediction(input_data):
    """
    Fonction pour faire des prédictions avec le modèle chargé.
    :param input_data: array-like, shape (n_samples, n_features)
    :return: valeurs prédites
    """
    input_data = np.array(input_data).reshape(1, -1)  # Redimensionner les données si nécessaire
    prediction = model.predict(input_data)
    return prediction
