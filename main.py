from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import json 
import requests
from pydantic import BaseModel
import pickle

class CropInput(BaseModel):
    Fertilizer: float
    Pesticide: float
   # Annual_Rainfall: float
    Area: float
    #Crop: str
    Season: str
    State: str
    Crop_Year: int
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    Moisture: float
    Soil_Type: str

app = FastAPI()

# Enable CORS so frontend (different port) can access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],    
)
#Production model
# --- Load artifacts ---
model = joblib.load("model.pkl")         # your trained model
scaler_X = joblib.load("scaler_x.pkl")   # scaler for features
#scaler_y = joblib.load("scaler_y.pkl")   # scaler for target
ohe = joblib.load("ohe_features.pkl")             # fitted OneHotEncoder
model_columns = joblib.load("model_columns.pkl")

# Columns
numeric_cols = ['Pesticide', 'Fertilizer', 'Area', 'Annual_Rainfall']
categorical_cols =['State', 'Crop', 'Season']

# Crop suggestion model

crop_model = joblib.load("crop_model.pkl")
crop_scaler = joblib.load("crop_scaler.pkl")
crop_prediction_le = joblib.load("crop_label_encoder.pkl")

print("model:", type(model))
print("crop_model:", type(crop_model))

with open("crop_features.json", "r") as f:
    crop_feature_list = json.load(f)

# Fertilizer prediction model

with open("fertilizer_model.pkl", "rb") as f:
    fertilizer_model = pickle.load(f)


print("fertilizer_model:", type(fertilizer_model))

with open("fertilizer_scaler.pkl", "rb") as f:
    fertilizer_scaler = pickle.load(f)

soil_le = pickle.load(open("Soil_Type_encoder.pkl", "rb"))
crop_type_le = pickle.load(open("Crop_Type_encoder.pkl", "rb"))
target_le = pickle.load(open("fertilizer_target_encoder.pkl", "rb"))

print("crop_prediction_le:", type(crop_prediction_le))
print("crop_type_le:", type(crop_type_le))
print("soil_le:", type(soil_le))
print("target_le:", type(target_le))

@app.post("/predict")
async def predict(data: CropInput):
    try:
        # ===== Crop Prediction =====
        input_df = pd.DataFrame([data.dict()])[crop_feature_list]
        input_scaled = crop_scaler.transform(input_df)

        pred_label = crop_model.predict(input_scaled)[0]
        pred_crop = crop_prediction_le.inverse_transform([pred_label])[0]

        # ===== Yield Prediction =====
        df = pd.DataFrame([data.dict()])
        df["Crop"] = pred_crop.capitalize()
        df["Annual_Rainfall"] = data.rainfall

        # Log transform
        for col in numeric_cols:
            df[f"{col}_log"] = np.log1p(df[col])

        area = float(np.expm1(df["Area_log"].iloc[0]))

        # Scale numeric
        log_cols = [f"{c}_log" for c in numeric_cols]
        scaled = scaler_X.transform(df[log_cols])

        scaled_df = pd.DataFrame(
            scaled,
            columns=[f"{c}_scaled" for c in numeric_cols]
        )

        df = pd.concat([df, scaled_df], axis=1)

        # Encode categorical
        encoded = ohe.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=ohe.get_feature_names_out(categorical_cols)
        )

        df = pd.concat([df, encoded_df], axis=1)

        # Match training columns
        df = df.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )
        
        yield_pred = np.expm1(model.predict(df))[0]

        # ===== Fertilizer Prediction =====
        if pred_crop not in crop_type_le.classes_:
            return {
                "predicted_crop": pred_crop,
                "prediction_scaled": round(float(yield_pred[0]), 2),
                "Yield": round(float(yield_pred[0] / area), 2),
                "warning": f"{pred_crop} not supported by fertilizer model"
            }
        print(crop_prediction_le.classes_)  
        print(crop_type_le.classes_)
        soil_encoded = soil_le.transform(
            [data.Soil_Type.strip()]
        )[0]

        crop_encoded = crop_type_le.transform(
            [pred_crop]
        )[0]

        num_values = np.array([
            data.temperature,
            data.humidity,
            data.Moisture,
            data.N,
            data.P,
            data.K
        ]).reshape(1, -1)

        num_scaled = fertilizer_scaler.transform(num_values)

        X_input = np.hstack([
            num_scaled,
            [[soil_encoded, crop_encoded]]
        ])

        fert_pred = fertilizer_model.predict(X_input)[0]
        pred_fertilizer = target_le.inverse_transform(
            [fert_pred]
        )[0]

        return {
            "predicted_crop": pred_crop,
            "predicted_fertilizer": pred_fertilizer,
            "prediction_scaled": round(float(yield_pred[0]), 2),
            "Yield": round(float(yield_pred[0] / area), 2)
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    






    
# Alternatively, manually save the list from training
# @app.post("/predict")
# async def predict(data: CropInput):
#     input_df = pd.DataFrame([data.dict()])[crop_feature_list]
#     input_scaled = crop_scaler.transform(input_df)

#     pred_label = crop_model.predict(input_scaled)[0]
#     pred_crop = crop_prediction_le.inverse_transform([pred_label])[0]

#     df = pd.DataFrame([data.dict()])
#     df["Crop"] = pred_crop.capitalize()
#     df["Annual_Rainfall"] = data.rainfall

#     log_cols = [col + "_log" for col in numeric_cols]

#     for col in numeric_cols:
#         df[col + "_log"] = np.log1p(df[col])
#     df_scaled = scaler_X.transform(df[log_cols])
#     soil_encoded = int(
#         soil_le.transform([data.Soil_Type.strip()])[0]
#     )

#     crop_encoded = int(
#         crop_type_le.transform([pred_crop.strip()])[0]
#     )

#     return {
#         "soil_encoded": soil_encoded,
#         "crop_encoded": crop_encoded
    # }