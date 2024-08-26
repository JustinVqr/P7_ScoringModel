import joblib
import numpy as np

# Charger le modèle sauvegardé
with open('app/model/best_model.pkl', 'rb') as file:
    model = joblib.load(file)

def make_prediction(input_data):
    """
    Fonction pour faire des prédictions avec le modèle chargé.
    :param input_data: array-like, shape (n_samples, n_features)
    :return: valeurs prédites
    """
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction
