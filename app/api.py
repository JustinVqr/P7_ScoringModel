from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import pickle

# Charger le modèle sauvegardé
with open('app/model/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialisation de FastAPI
app = FastAPI()

# Classe de données client (basée sur les champs requis)
class ClientData(BaseModel):
    PAYMENT_RATE: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    DAYS_BIRTH: int
    AMT_ANNUITY: float
    INSTAL_AMT_PAYMENT_SUM: float
    PREV_CNT_PAYMENT_MEAN: float
    AMT_CREDIT: float
    INSTAL_DPD_MEAN: float
    APPROVED_CNT_PAYMENT_MEAN: float
    AMT_GOODS_PRICE: float
    DAYS_ID_PUBLISH: int
    DAYS_EMPLOYED: float
    BURO_DAYS_CREDIT_MAX: float
    POS_MONTHS_BALANCE_SIZE: float
    INSTAL_DAYS_ENTRY_PAYMENT_MAX: float
    DAYS_LAST_PHONE_CHANGE: float
    INSTAL_AMT_PAYMENT_MIN: float
    BURO_DAYS_CREDIT_ENDDATE_MAX: float
    INSTAL_DAYS_ENTRY_PAYMENT_MEAN: float
    BURO_AMT_CREDIT_SUM_DEBT_MEAN: float
    APPROVED_AMT_ANNUITY_MEAN: float
    INSTAL_PAYMENT_DIFF_MEAN: float
    PREV_NAME_CONTRACT_STATUS_Refused_MEAN: float
    DAYS_EMPLOYED_PERC: float
    PREV_PRODUCT_COMBINATION_CashXSelllow_MEAN: float
    INSTAL_AMT_PAYMENT_MAX: float
    BURO_AMT_CREDIT_SUM_SUM: float
    ANNUITY_INCOME_PERC: float
    PREV_AMT_DOWN_PAYMENT_MAX: float
    INCOME_CREDIT_PERC: float
    INSTAL_DAYS_ENTRY_PAYMENT_SUM: float
    PREV_APP_CREDIT_PERC_MIN: float
    NAME_CONTRACT_TYPE_Cashloans: bool
    REGION_RATING_CLIENT_W_CITY: int
    OCCUPATION_TYPE_Drivers: bool
    NAME_EDUCATION_TYPE_Highereducation: bool
    BURO_CREDIT_TYPE_Mortgage_MEAN: float
    APPROVED_CNT_PAYMENT_SUM: float
    DAYS_REGISTRATION: float
    POS_SK_DPD_DEF_MEAN: float
    BURO_MONTHS_BALANCE_SIZE_SUM: float
    PREV_NAME_YIELD_GROUP_high_MEAN: float
    PREV_NAME_YIELD_GROUP_low_action_MEAN: float
    APPROVED_APP_CREDIT_PERC_MAX: float
    REGION_POPULATION_RELATIVE: float
    PREV_APP_CREDIT_PERC_MEAN: float
    INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE: float
    BURO_AMT_CREDIT_SUM_LIMIT_SUM: float
    BURO_AMT_CREDIT_SUM_MAX: float
    INSTAL_DBD_MAX: float
    NAME_FAMILY_STATUS_Married: bool
    PREV_NAME_PAYMENT_TYPE_XNA_MEAN: float
    BURO_DAYS_CREDIT_MEAN: float
    FLAG_OWN_CAR: int
    BURO_CREDIT_TYPE_Microloan_MEAN: float
    APPROVED_DAYS_DECISION_MAX: float
    BURO_AMT_CREDIT_SUM_DEBT_SUM: float
    INSTAL_PAYMENT_PERC_MEAN: float
    PREV_NAME_CLIENT_TYPE_New_MEAN: float
    INSTAL_AMT_PAYMENT_MEAN: float
    BURO_AMT_CREDIT_SUM_OVERDUE_MEAN: float
    INSTAL_DBD_MEAN: float
    BURO_AMT_CREDIT_SUM_MEAN: float
    INCOME_PER_PERSON: float
    BURO_DAYS_CREDIT_ENDDATE_MEAN: float
    AMT_REQ_CREDIT_BUREAU_QRT: float
    INSTAL_PAYMENT_DIFF_SUM: float
    BURO_CREDIT_ACTIVE_Active_MEAN: float
    POS_MONTHS_BALANCE_MEAN: float
    PREV_CNT_PAYMENT_SUM: float
    PREV_DAYS_DECISION_MIN: float
    PREV_DAYS_DECISION_MEAN: float
    INSTAL_DBD_SUM: float
    PREV_PRODUCT_COMBINATION_CashStreetlow_MEAN: float
    APPROVED_AMT_ANNUITY_MAX: float
    APPROVED_AMT_CREDIT_MAX: float
    PREV_NAME_GOODS_CATEGORY_Furniture_MEAN: float
    HOUR_APPR_PROCESS_START: int
    OCCUPATION_TYPE_Laborers: bool
    APPROVED_AMT_APPLICATION_MIN: float
    POS_NAME_CONTRACT_STATUS_Active_MEAN: float
    SK_ID_CURR_1: int
    PREV_PRODUCT_COMBINATION_POSindustrywithinterest_MEAN: float
    POS_NAME_CONTRACT_STATUS_Completed_MEAN: float
    NAME_INCOME_TYPE_Working: bool
    PREV_NAME_GOODS_CATEGORY_XNA_MEAN: float
    DEF_60_CNT_SOCIAL_CIRCLE: float
    FLAG_DOCUMENT_3: int
    APPROVED_AMT_CREDIT_MIN: float
    PREV_AMT_ANNUITY_MIN: float
    INSTAL_DPD_MAX: float
    INSTAL_PAYMENT_DIFF_MAX: float
    DEF_30_CNT_SOCIAL_CIRCLE: float
    BURO_CREDIT_TYPE_Carloan_MEAN: float
    POS_SK_DPD_DEF_MAX: float
    APPROVED_HOUR_APPR_PROCESS_START_MAX: float
    ORGANIZATION_TYPE_Construction: bool
    PREV_CHANNEL_TYPE_Channelofcorporatesales_MEAN: float

# Fonction pour faire des prédictions
def make_prediction(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction

# Message d'accueil
@app.get("/")
def read_root():
    return {"message": "Bonjour, vous êtes bien sur l'application de scoring, hébergée sur Heroku. "
                       "Cette API permet de prédire la probabilité de défaut de paiement pour un client "
                       "en fonction de ses caractéristiques. Envoyez une requête POST à /predict pour obtenir une prédiction."}

@app.post("/predict")
def predict(client_data: ClientData):
    # Convertir les données en liste pour les utiliser avec le modèle
    input_data = [
        client_data.PAYMENT_RATE,
        client_data.EXT_SOURCE_2,
        client_data.EXT_SOURCE_3,
        client_data.DAYS_BIRTH,
        client_data.AMT_ANNUITY,
        client_data.INSTAL_AMT_PAYMENT_SUM,
        client_data.PREV_CNT_PAYMENT_MEAN,
        client_data.AMT_CREDIT,
        client_data.INSTAL_DPD_MEAN,
        client_data.APPROVED_CNT_PAYMENT_MEAN,
        client_data.AMT_GOODS_PRICE,
        client_data.DAYS_ID_PUBLISH,
        client_data.DAYS_EMPLOYED,
        client_data.BURO_DAYS_CREDIT_MAX,
        client_data.POS_MONTHS_BALANCE_SIZE,
        client_data.INSTAL_DAYS_ENTRY_PAYMENT_MAX,
        client_data.DAYS_LAST_PHONE_CHANGE,
        client_data.INSTAL_AMT_PAYMENT_MIN,
        client_data.BURO_DAYS_CREDIT_ENDDATE_MAX,
        client_data.INSTAL_DAYS_ENTRY_PAYMENT_MEAN,
        client_data.BURO_AMT_CREDIT_SUM_DEBT_MEAN,
        client_data.APPROVED_AMT_ANNUITY_MEAN,
        client_data.INSTAL_PAYMENT_DIFF_MEAN,
        client_data.PREV_NAME_CONTRACT_STATUS_Refused_MEAN,
        client_data.DAYS_EMPLOYED_PERC,
        client_data.PREV_PRODUCT_COMBINATION_CashXSelllow_MEAN,
        client_data.INSTAL_AMT_PAYMENT_MAX,
        client_data.BURO_AMT_CREDIT_SUM_SUM,
        client_data.ANNUITY_INCOME_PERC,
        client_data.PREV_AMT_DOWN_PAYMENT_MAX,
        client_data.INCOME_CREDIT_PERC,
        client_data.INSTAL_DAYS_ENTRY_PAYMENT_SUM,
        client_data.PREV_APP_CREDIT_PERC_MIN,
        client_data.NAME_CONTRACT_TYPE_Cashloans,
        client_data.REGION_RATING_CLIENT_W_CITY,
        client_data.OCCUPATION_TYPE_Drivers,
        client_data.NAME_EDUCATION_TYPE_Highereducation,
        client_data.BURO_CREDIT_TYPE_Mortgage_MEAN,
        client_data.APPROVED_CNT_PAYMENT_SUM,
        client_data.DAYS_REGISTRATION,
        client_data.POS_SK_DPD_DEF_MEAN,
        client_data.BURO_MONTHS_BALANCE_SIZE_SUM,
        client_data.PREV_NAME_YIELD_GROUP_high_MEAN,
        client_data.PREV_NAME_YIELD_GROUP_low_action_MEAN,
        client_data.APPROVED_APP_CREDIT_PERC_MAX,
        client_data.REGION_POPULATION_RELATIVE,
        client_data.PREV_APP_CREDIT_PERC_MEAN,
        client_data.INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE,
        client_data.BURO_AMT_CREDIT_SUM_LIMIT_SUM,
        client_data.BURO_AMT_CREDIT_SUM_MAX,
        client_data.INSTAL_DBD_MAX,
        client_data.NAME_FAMILY_STATUS_Married,
        client_data.PREV_NAME_PAYMENT_TYPE_XNA_MEAN,
        client_data.BURO_DAYS_CREDIT_MEAN,
        client_data.FLAG_OWN_CAR,
        client_data.BURO_CREDIT_TYPE_Microloan_MEAN,
        client_data.APPROVED_DAYS_DECISION_MAX,
        client_data.BURO_AMT_CREDIT_SUM_DEBT_SUM,
        client_data.INSTAL_PAYMENT_PERC_MEAN,
        client_data.PREV_NAME_CLIENT_TYPE_New_MEAN,
        client_data.INSTAL_AMT_PAYMENT_MEAN,
        client_data.BURO_AMT_CREDIT_SUM_OVERDUE_MEAN,
        client_data.INSTAL_DBD_MEAN,
        client_data.BURO_AMT_CREDIT_SUM_MEAN,
        client_data.INCOME_PER_PERSON,
        client_data.BURO_DAYS_CREDIT_ENDDATE_MEAN,
        client_data.AMT_REQ_CREDIT_BUREAU_QRT,
        client_data.INSTAL_PAYMENT_DIFF_SUM,
        client_data.BURO_CREDIT_ACTIVE_Active_MEAN,
        client_data.POS_MONTHS_BALANCE_MEAN,
        client_data.PREV_CNT_PAYMENT_SUM,
        client_data.PREV_DAYS_DECISION_MIN,
        client_data.PREV_DAYS_DECISION_MEAN,
        client_data.INSTAL_DBD_SUM,
        client_data.PREV_PRODUCT_COMBINATION_CashStreetlow_MEAN,
        client_data.APPROVED_AMT_ANNUITY_MAX,
        client_data.APPROVED_AMT_CREDIT_MAX,
        client_data.PREV_NAME_GOODS_CATEGORY_Furniture_MEAN,
        client_data.HOUR_APPR_PROCESS_START,
        client_data.OCCUPATION_TYPE_Laborers,
        client_data.APPROVED_AMT_APPLICATION_MIN,
        client_data.POS_NAME_CONTRACT_STATUS_Active_MEAN,
        client_data.SK_ID_CURR_1,
        client_data.PREV_PRODUCT_COMBINATION_POSindustrywithinterest_MEAN,
        client_data.POS_NAME_CONTRACT_STATUS_Completed_MEAN,
        client_data.NAME_INCOME_TYPE_Working,
        client_data.PREV_NAME_GOODS_CATEGORY_XNA_MEAN,
        client_data.DEF_60_CNT_SOCIAL_CIRCLE,
        client_data.FLAG_DOCUMENT_3,
        client_data.APPROVED_AMT_CREDIT_MIN,
        client_data.PREV_AMT_ANNUITY_MIN,
        client_data.INSTAL_DPD_MAX,
        client_data.INSTAL_PAYMENT_DIFF_MAX,
        client_data.DEF_30_CNT_SOCIAL_CIRCLE,
        client_data.BURO_CREDIT_TYPE_Carloan_MEAN,
        client_data.POS_SK_DPD_DEF_MAX,
        client_data.APPROVED_HOUR_APPR_PROCESS_START_MAX,
        client_data.ORGANIZATION_TYPE_Construction,
        client_data.PREV_CHANNEL_TYPE_Channelofcorporatesales_MEAN
    ]  # Assurez-vous de fermer la liste ici.

    # Faire la prédiction
    prediction = make_prediction(input_data)

    return {"prediction": int(prediction[0])}
