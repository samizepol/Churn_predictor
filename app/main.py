import joblib
from app.model import MODEL_PATH 
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from app.features import single_user_features
from app.model import predict

app = FastAPI(title="Churn Predictor")

class Product(BaseModel):
    productId:int
    category:str
    price:float
    quantity:int

class Cart(BaseModel):
    id:int
    userId:int
    date:str
    products:List[Product]

class User(BaseModel):
    id:int
    email:str
    registered_at:str
    address:dict

class PredictRequest(BaseModel):
    user:User
    carts:List[Cart]

@app.get("/health")
def health():
    return {"status":"ok"}


@app.post("/predict")
def predict_endpoint(payload: PredictRequest):
    # Build features from the payload
    feats_df = single_user_features(payload.model_dump())
    
    # Drop identifiers
    X_live = feats_df.drop(columns=["user_id", "churn"], errors="ignore")
    
    # Load the model's memory dynamically to extract the EXACT training columns and order
    loaded_brain = joblib.load(MODEL_PATH)
    model_features = list(loaded_brain.feature_names_in_)
    
    for col in model_features:
        if col not in X_live.columns:
            X_live[col] = 0.0
            
    # Reindex to match the EXACT sequence order the model is expecting
    X_live = X_live[model_features]

    return predict(X_live)