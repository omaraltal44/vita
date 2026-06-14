import json
import time

from fastapi.testclient import TestClient
from sqlalchemy import text

import main


payloads = {
    "diabetes": {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 31,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 2,
        "PhysHlth": 1,
        "DiffWalk": 0,
        "Sex": 1,
        "Age": 8,
        "Education": 5,
        "Income": 6,
    },
    "obesity": {
        "Gender": 1,
        "Age": 28,
        "Height": 1.72,
        "Weight": 84,
        "family_history_with_overweight": 1,
        "FAVC": 1,
        "FCVC": 2,
        "NCP": 3,
        "CAEC": 2,
        "SMOKE": 0,
        "CH2O": 2,
        "SCC": 0,
        "FAF": 1,
        "TUE": 1,
        "CALC": 1,
        "BMI": 28.4,
        "MTRANS_Automobile": 0,
        "MTRANS_Bike": 0,
        "MTRANS_Motorbike": 0,
        "MTRANS_Public_Transportation": 1,
        "MTRANS_Walking": 0,
    },
    "hypertension": {
        "age": 54,
        "bmi": 29.4,
        "family_history_hypertension": 1,
        "diabetes": 0,
        "smoking": 0,
        "alcohol_heavy": 0,
        "physically_active": 1,
        "high_salt_diet": 1,
        "stroke_history": 0,
        "myocardial_infarction": 0,
        "heart_failure": 0,
        "total_cholesterol_mg_dl": 210,
        "ldl_mg_dl": 130,
        "hdl_mg_dl": 45,
        "creatinine_mg_dl": 1.0,
        "sex_Male": 1,
        "residence_Urban": 1,
    },
    "heart": {
        "male": 1,
        "age": 52,
        "education": 2,
        "currentSmoker": 0,
        "cigsPerDay": 0,
        "BPMeds": 0,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
        "diabetes": 0,
        "totChol": 210,
        "sysBP": 135,
        "diaBP": 84,
        "BMI": 28.5,
        "heartRate": 76,
        "glucose": 96,
    },
}


def run():
    email = f"qa_mysql_{int(time.time())}@example.com"
    password = "Secret123!"
    out = {}

    with TestClient(main.app) as client:
        out["register_request"] = {"name": "QA MySQL", "email": email, "password": "***"}
        register = client.post("/auth/register", json={"name": "QA MySQL", "email": email, "password": password})
        out["register_status"] = register.status_code
        out["register_response"] = register.json()

        out["login_request"] = {"email": email, "password": "***"}
        login = client.post("/auth/login", json={"email": email, "password": password})
        out["login_status"] = login.status_code
        out["login_response"] = login.json()

        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        out["prediction_tests"] = {}
        for disease, payload in payloads.items():
            response = client.post(f"/predict/{disease}", json=payload, headers=headers)
            out["prediction_tests"][disease] = {
                "request": payload,
                "status": response.status_code,
                "response": response.json(),
            }

    with main.engine.connect() as conn:
        out["users_rows"] = [dict(row._mapping) for row in conn.execute(text("SELECT id, name, email, password, created_at FROM users WHERE email = :email"), {"email": email})]
        user_id = out["users_rows"][0]["id"]
        out["predictions_rows"] = [
            dict(row._mapping)
            for row in conn.execute(
                text("SELECT id, user_id, disease_type, input_data, prediction, probability, created_at FROM predictions WHERE user_id = :user_id ORDER BY id"),
                {"user_id": user_id},
            )
        ]

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    run()
