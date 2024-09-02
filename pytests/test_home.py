import pytest
import pandas as pd
from io import StringIO
from unittest.mock import patch, MagicMock
from unittest import mock
import streamlit as st
from app.Home import load_data, stratified_sampling, load_model_and_explainer

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

from unittest.mock import patch, MagicMock
import pandas as pd
import shap

# Test pour la fonction load_model_and_explainer
@patch('joblib.load')
@patch('os.path.exists')
@patch('shap.kmeans')
@patch('shap.TreeExplainer')
@patch('shap.KernelExplainer')
def test_load_model_and_explainer(mock_kernel_explainer, mock_tree_explainer, mock_kmeans, mock_exists, mock_load):
    mock_exists.return_value = True
    
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    
    mock_background_data = MagicMock()
    mock_kmeans.return_value = mock_background_data
    
    mock_tree_exp = MagicMock()
    mock_tree_explainer.return_value = mock_tree_exp
    
    data = {'SK_ID_CURR': [100001, 100002, 100003, 100004], 'TARGET': [1, 0, 1, 0]}
    df_train = pd.DataFrame(data)
    
    model, explainer = load_model_and_explainer(df_train)
    
    assert model is not None, "Le modèle ne doit pas être None"
    assert explainer is not None, "L'explicateur ne doit pas être None"


## --- Initialisation de l'état de session --- 
def test_initial_session_state():
    # Simuler un état de session vide
    with mock.patch.dict(st.session_state, {}, clear=True):
        if 'load_state' not in st.session_state:
            st.session_state.load_state = False

        assert st.session_state.load_state == False, "L'état initial de load_state doit être False"

