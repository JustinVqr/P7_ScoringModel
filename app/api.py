from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import joblib

# Charger le modèle sauvegardé
with open('app/model/best_model.pkl', 'rb') as file:
    model = joblib.load(file)

# Initialisation de FastAPI
app = FastAPI()

# Classe de données client (avec des valeurs par défaut pour certains champs)
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
    NAME_CONTRACT_TYPE_Cashloans: bool = False
    REGION_RATING_CLIENT_W_CITY: int = 0
    OCCUPATION_TYPE_Drivers: bool = False
    NAME_EDUCATION_TYPE_Highereducation: bool = False
    BURO_CREDIT_TYPE_Mortgage_MEAN: float = 0.0
    APPROVED_CNT_PAYMENT_SUM: float = 0.0
    DAYS_REGISTRATION: float = 0.0
    POS_SK_DPD_DEF_MEAN: float = 0.0
    BURO_MONTHS_BALANCE_SIZE_SUM: float = 0.0
    PREV_NAME_YIELD_GROUP_high_MEAN: float = 0.0
    PREV_NAME_YIELD_GROUP_low_action_MEAN: float = 0.0
    APPROVED_APP_CREDIT_PERC_MAX: float = 0.0
    REGION_POPULATION_RELATIVE: float = 0.0
    PREV_APP_CREDIT_PERC_MEAN: float = 0.0
    INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE: float = 0.0
    BURO_AMT_CREDIT_SUM_LIMIT_SUM: float = 0.0
    BURO_AMT_CREDIT_SUM_MAX: float = 0.0
    INSTAL_DBD_MAX: float = 0.0
    NAME_FAMILY_STATUS_Married: bool = False
    PREV_NAME_PAYMENT_TYPE_XNA_MEAN: float = 0.0
    BURO_DAYS_CREDIT_MEAN: float = 0.0
    FLAG_OWN_CAR: int = 0
    BURO_CREDIT_TYPE_Microloan_MEAN: float = 0.0
    APPROVED_DAYS_DECISION_MAX: float = 0.0
    BURO_AMT_CREDIT_SUM_DEBT_SUM: float = 0.0
    INSTAL_PAYMENT_PERC_MEAN: float = 0.0
    PREV_NAME_CLIENT_TYPE_New_MEAN: float = 0.0
    INSTAL_AMT_PAYMENT_MEAN: float = 0.0
    BURO_AMT_CREDIT_SUM_OVERDUE_MEAN: float = 0.0
    INSTAL_DBD_MEAN: float = 0.0
    BURO_AMT_CREDIT_SUM_MEAN: float = 0.0
    INCOME_PER_PERSON: float = 0.0
    BURO_DAYS_CREDIT_ENDDATE_MEAN: float = 0.0
    AMT_REQ_CREDIT_BUREAU_QRT: float = 0.0
    INSTAL_PAYMENT_DIFF_SUM: float = 0.0
    BURO_CREDIT_ACTIVE_Active_MEAN: float = 0.0
    POS_MONTHS_BALANCE_MEAN: float = 0.0
    PREV_CNT_PAYMENT_SUM: float = 0.0
    PREV_DAYS_DECISION_MIN: float = 0.0
    PREV_DAYS_DECISION_MEAN: float = 0.0
    INSTAL_DBD_SUM: float = 0.0
    PREV_PRODUCT_COMBINATION_CashStreetlow_MEAN: float = 0.0
    APPROVED_AMT_ANNUITY_MAX: float = 0.0
    APPROVED_AMT_CREDIT_MAX: float = 0.0
    PREV_NAME_GOODS_CATEGORY_Furniture_MEAN: float = 0.0
    HOUR_APPR_PROCESS_START: int = 0
    OCCUPATION_TYPE_Laborers: bool = False
    APPROVED_AMT_APPLICATION_MIN: float = 0.0
    POS_NAME_CONTRACT_STATUS_Active_MEAN: float = 0.0
    PREV_PRODUCT_COMBINATION_POSindustrywithinterest_MEAN: float = 0.0
    POS_NAME_CONTRACT_STATUS_Completed_MEAN: float = 0.0
    NAME_INCOME_TYPE_Working: bool = False
    PREV_NAME_GOODS_CATEGORY_XNA_MEAN: float = 0.0
    DEF_60_CNT_SOCIAL_CIRCLE: float = 0.0
    FLAG_DOCUMENT_3: int = 0
    APPROVED_AMT_CREDIT_MIN: float = 0.0
    PREV_AMT_ANNUITY_MIN: float = 0.0
    INSTAL_DPD_MAX: float = 0.0
    INSTAL_PAYMENT_DIFF_MAX: float = 0.0
    DEF_30_CNT_SOCIAL_CIRCLE: float = 0.0
    BURO_CREDIT_TYPE_Carloan_MEAN: float = 0.0
    POS_SK_DPD_DEF_MAX: float = 0.0
    APPROVED_HOUR_APPR_PROCESS_START_MAX: float = 0.0
    ORGANIZATION_TYPE_Construction: bool = False
    PREV_CHANNEL_TYPE_Channelofcorporatesales_MEAN: float = 0.0

def make_prediction(input_data, threshold=0.4):
    try:
        input_data = np.array(input_data).reshape(1, -1)
        probability = model.predict_proba(input_data)
        
        if probability is None or len(probability) == 0:
            raise ValueError("Les probabilités ne sont pas disponibles pour cette entrée.")
        
        prediction = (probability[0][1] >= threshold).astype(int)
        return prediction, probability
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

# Message d'accueil
@app.get("/")
def read_root():
    return {"message": "Bonjour, vous êtes bien sur l'application de scoring, hébergée sur Heroku. "
                       "Cette API permet de prédire la probabilité de défaut de paiement pour un client "
                       "en fonction de ses caractéristiques. Envoyez une requête POST à /predict pour obtenir une prédiction."}

@app.post("/predict")
def predict(client_data: ClientData):
    try:
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
        ]

        # Faire la prédiction et obtenir les probabilités
        prediction, probability = make_prediction(input_data, threshold=0.4)

        return {"prediction": int(prediction[0]), "probability": float(probability[0][1])}
    
    except HTTPException as e:
        raise e
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")
