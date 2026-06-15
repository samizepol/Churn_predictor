import joblib
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from app.features import load_training_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent

# Force the model path to be in that exact folder
MODEL_PATH = ROOT_DIR / "saved_model.joblib"

def train_model():
    df = load_training_dataset()
    X = df.drop(columns=["user_id","churn"])
    y = df["churn"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train,y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    return model

def load_model():
    return joblib.load(MODEL_PATH)

def predict(df_features):
    model = load_model()

    X = df_features.drop(columns=["user_id"], errors="ignore")

    prob = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])
    
    return {"churn_probability": prob, "churn_class": pred}


if __name__ == "__main__":
    trained_model = train_model()
    print(f"SUCCESS! Look for your file here: {MODEL_PATH}")


