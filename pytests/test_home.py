import pytest
import pandas as pd
from io import StringIO
from unittest.mock import patch, MagicMock
from unittest import mock
import streamlit as st
from app.Home import load_data, stratified_sampling, load_model

# Test pour la fonction load_data
@patch('requests.get')
def test_load_data(mock_get):
    mock_response_train = MagicMock()
    mock_response_train.status_code = 200
    mock_response_train.text = 'SK_ID_CURR,TARGET\n100001,1\n100002,0\n'
    
    mock_response_new = MagicMock()
    mock_response_new.status_code = 200
    mock_response_new.text = 'SK_ID_CURR,TARGET\n100003,1\n100004,0\n'
    
    mock_get.side_effect = [mock_response_train, mock_response_new]
    
    df_train, df_new = load_data()
    
    assert not df_train.empty, "Le dataframe train ne devrait pas être vide"
    assert not df_new.empty, "Le dataframe new ne devrait pas être vide"
    assert 'TARGET' in df_train.columns, "Le dataframe train doit contenir la colonne TARGET"
    assert 'TARGET' in df_new.columns, "Le dataframe new doit contenir la colonne TARGET"

# Test pour la fonction stratified_sampling
def test_stratified_sampling():
    data = {'SK_ID_CURR': [100001, 100002, 100003, 100004], 'TARGET': [1, 0, 1, 0]}
    df = pd.DataFrame(data)
    
    df_sampled = stratified_sampling(df, sample_size=0.5)
    
    assert len(df_sampled) == 2, "Le dataframe échantillonné devrait contenir 2 lignes"
    assert df_sampled['TARGET'].value_counts().sum() == 2, "Le dataframe échantillonné doit avoir deux entrées"

# Test pour la fonction load_model
@patch('joblib.load')
@patch('os.path.exists')
def test_load_model(mock_exists, mock_load):
    # Simuler l'existence du fichier de modèle
    mock_exists.return_value = True
    
    # Simuler le chargement du modèle avec joblib
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    
    with patch('shap.TreeExplainer') as mock_explainer:
        # Simuler l'explainer SHAP
        mock_explainer_instance = MagicMock()
        mock_explainer.return_value = mock_explainer_instance
        
        model = load_model()
        
        # Vérifier que le modèle a été chargé correctement
        assert model is not None, "Le modèle ne doit pas être None"
        
        # Vérifier que l'explicateur SHAP a été correctement initialisé
        assert 'explainer' in st.session_state, "L'explicateur SHAP doit être dans l'état de session"
        assert 'Credit_clf_final' in st.session_state, "Le modèle doit être dans l'état de session"
        assert st.session_state.Credit_clf_final == mock_model, "Le modèle dans l'état de session doit correspondre au modèle chargé"
        assert st.session_state.explainer == mock_explainer_instance, "L'explicateur dans l'état de session doit correspondre à l'explicateur initialisé"

# Test pour vérifier l'état initial de session
def test_initial_session_state():
    # Simuler un état de session vide
    with mock.patch.dict(st.session_state, {}, clear=True):
        if 'load_state' not in st.session_state:
            st.session_state.load_state = False

        assert st.session_state.load_state == False, "L'état initial de load_state doit être False"
