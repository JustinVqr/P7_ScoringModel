import re
import numpy as np
import pandas as pd
import gc
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Preprocess application_train.csv and application_test.csv
def application_train_test(path, df_file, num_rows=None):
    df = pd.read_csv(path + df_file, nrows=num_rows)
    df = df.drop(columns='CODE_GENDER')

    for bin_feature in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    df, cat_cols = one_hot_encoder(df, nan_as_category=True)
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    return df


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(path, num_rows=None):
    bureau = pd.read_csv(path + 'bureau.csv', nrows=num_rows)
    bb = pd.read_csv(path + 'bureau_balance.csv', nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category=True)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category=True)

    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }

    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


# Preprocess previous_applications.csv
def previous_applications(path, num_rows=None):
    prev = pd.read_csv(path + 'previous_application.csv', nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum']
    }

    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(path, num_rows=None):
    pos = pd.read_csv(path + 'POS_CASH_balance.csv', nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)

    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(path, num_rows=None):
    ins = pd.read_csv(path + 'installments_payments.csv', nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }

    for cat in cat_cols:
        aggregations[cat] = ['mean']

    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(path, num_rows=None):
    cc = pd.read_csv(path + 'credit_card_balance.csv', nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


# Choix de sélection du modèle avec un paramètres de seuil du taux de remplissage pour la conservation de feature dans le df
def select_features_by_fill_rate(df, threshold=0.8):
    """
    Sélectionne des colonnes d'un DataFrame en fonction d'un seuil de taux de remplissage défini manuellement.
    
    Arguments:
    df -- DataFrame d'entrée.
    threshold -- Seuil pour la sélection des colonnes
    
    Retourne:
    DataFrame avec les colonnes sélectionnées.
    """
    # Calculer le taux de remplissage pour chaque colonne
    fill_rates = df.notna().mean()
    
    # Sélectionner les colonnes dont le taux de remplissage est supérieur au seuil
    selected_columns = fill_rates[fill_rates >= threshold].index
    
    # Retourner le DataFrame avec les colonnes sélectionnées
    return df[selected_columns]


# Fonction de sélection de caractéristiques avec SVM L1
def feature_selection_with_svm_l1(df, target_column='TARGET', C_value=0.01, max_iter=10000):
    # Séparer les features et la cible
    X = df.drop(columns=target_column)
    y = df[target_column]
    
    # Remplacer les valeurs infinies par NaN pour les gérer par la suite
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Gérer les valeurs manquantes en les remplissant avec 0 ou une autre méthode
    X.fillna(0, inplace=True)

    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Application du SVM L1 avec pénalité L1
    lsvc = LinearSVC(C=C_value, penalty="l1", dual=False, max_iter=max_iter)
    
    try:
        lsvc.fit(X_scaled, y)
        model = SelectFromModel(lsvc, prefit=True)
        selected_features = X.columns[model.get_support()]
        
        # Retourner le dataframe avec les caractéristiques sélectionnées et la cible
        df_selec = df[selected_features.tolist() + [target_column]].copy()
        return df_selec, selected_features
    
    except ValueError as e:
        print(f"Erreur lors de l'ajustement du modèle SVM L1 : {e}")
        return None, None


# Fonction de sélection de caractéristiques avec LightGBM
def prepare_train_analyze_select(df, target_column='TARGET', threshold=95):
    df = df.dropna(subset=[target_column])

    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)

    X = df.drop(columns=target_column).fillna(0)
    y = df[target_column]

    clf = lgb.LGBMClassifier(objective="binary", n_jobs=-1, is_unbalance=True)
    clf.fit(X, y)

    feature_importances = clf.feature_importances_
    features = X.columns

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    importance_df['pourcentage_exp'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
    importance_df['cumulative_percentage'] = importance_df['pourcentage_exp'].cumsum()

    selected_features = importance_df[importance_df['cumulative_percentage'] <= threshold]
    selected_columns = selected_features['Feature'].tolist()

    filtered_df = df[selected_columns + [target_column]].copy()

    filtered_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    filtered_df.fillna(0, inplace=True)

    filtered_df.to_csv('df_mod.csv', index=True)
    return filtered_df, clf, selected_features




# Fonction principale de préparation des données avec les deux modes de sélection
def data_prep(path="C:/Users/justi/OneDrive/Cours - Travail/DATA SCIENCE/Formation - DataScientist/Projet n°7/Projet n°7_Scoring/",
              df_file='application_train.csv', debug=False, svm_C_value=0.01, lgbm_threshold=95):
    num_rows = 10000 if debug else None
    print("Path of the data folder : " + path)
    print("Dataframe : " + df_file)
    print("_" * 10)
    print(" ")

    # Étape 1 : Chargement et preprocessing des données principales
    print("1) Loading of df")
    df = application_train_test(path, df_file, num_rows)
    print("Raw df shape:", df.shape)
    print("_" * 10)
    print(" ")

    # Étapes suivantes pour le traitement des différents fichiers
    print("2) Processing of bureau")
    bureau = bureau_and_balance(path, num_rows)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau
    gc.collect()

    print("3) Process previous_applications")
    prev = previous_applications(path, num_rows)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    del prev
    gc.collect()

    print("4) Process POS-CASH balance")
    pos = pos_cash(path, num_rows)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    del pos
    gc.collect()

    print("5) Process installments payments")
    ins = installments_payments(path, num_rows)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    del ins
    gc.collect()

    print("6) Process credit card balance")
    cc = credit_card_balance(path, num_rows)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    del cc
    gc.collect()

    # Sélection des caractéristiques par taux de remplissage
    print("7) Feature selection by fill rate")
    df = select_features_by_fill_rate(df, threshold=0.8)
    print("Dataframe shape after fill rate selection:", df.shape)
    print("_" * 10)
    print(" ")

    # Sélection de caractéristiques avec SVM L1
    print("8) Feature selection with SVM L1")
    df_selec_svm, selected_features_svm = feature_selection_with_svm_l1(df, C_value=svm_C_value)
    print("SVM Selected Features:", len(selected_features_svm))
    print("_" * 10)
    print(" ")

    # Sélection de caractéristiques avec LightGBM après SVM
    print("9) Feature selection with LightGBM after SVM")
    df_final, clf, selected_features_lgbm = prepare_train_analyze_select(df_selec_svm, threshold=lgbm_threshold)
    print("LGBM Selected Features:", len(selected_features_lgbm))
    print("Final df shape:", df_final.shape)
    print("_" * 10)
    print(" ")

    return df_final


# Appel de la fonction de préparation des données avec les deux modes de sélection
df = data_prep()

print("Saving the transformed dataframe ...")
df.to_csv("output.csv", index_label='SK_ID_CURR', sep=";")
print("File saved")
