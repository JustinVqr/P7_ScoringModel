import pytest
import numpy as np
from app.model.P7_script_model_inf import make_prediction

@pytest.fixture
def mock_input_data():
    # Créez un jeu de données d'entrée simulé pour les tests
    return np.array([0.5, 1.2, -0.3, 2.1])

def test_make_prediction_positive_class(mock_input_data, monkeypatch):
    # Mock du modèle pour retourner une probabilité de 0.8 pour la classe 1
    class MockModel:
        def predict_proba(self, input_data):
            return np.array([[0.2, 0.8]])  # Classe 1 a une probabilité de 0.8

    # Utilisez monkeypatch pour remplacer le modèle original par le mock
    monkeypatch.setattr('app.model.P7_script_model_inf.model', MockModel())

    prediction, probability = make_prediction(mock_input_data)

    assert prediction == 1  # Devrait prédire la classe positive
    assert probability == 0.8  # La probabilité de la classe positive doit être 0.8

def test_make_prediction_negative_class(mock_input_data, monkeypatch):
    # Mock du modèle pour retourner une probabilité de 0.3 pour la classe 1
    class MockModel:
        def predict_proba(self, input_data):
            return np.array([[0.7, 0.3]])  # Classe 1 a une probabilité de 0.3

    # Utilisez monkeypatch pour remplacer le modèle original par le mock
    monkeypatch.setattr('app.model.P7_script_model_inf.model', MockModel())

    prediction, probability = make_prediction(mock_input_data)

    assert prediction == 0  # Devrait prédire la classe négative
    assert probability == 0.3  # La probabilité de la classe positive doit être 0.3

def test_make_prediction_with_different_threshold(mock_input_data, monkeypatch):
    # Mock du modèle pour retourner une probabilité de 0.4 pour la classe 1
    class MockModel:
        def predict_proba(self, input_data):
            return np.array([[0.6, 0.4]])  # Classe 1 a une probabilité de 0.4

    # Utilisez monkeypatch pour remplacer le modèle original par le mock
    monkeypatch.setattr('app.model.P7_script_model_inf.model', MockModel())

    # Test avec un seuil plus bas
    prediction, probability = make_prediction(mock_input_data, threshold=0.5)

    assert prediction == 0  # Devrait prédire la classe négative
    assert probability == 0.4  # La probabilité de la classe positive doit être 0.4
