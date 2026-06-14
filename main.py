from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import json
import logging
import os
import pickle
import time

import bcrypt
import httpx
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, inspect, text as sql_text
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PREDICTION_DEBUG_LOGS = os.getenv("PREDICTION_DEBUG_LOGS", "false").strip().lower() in {"1", "true", "yes", "on"}
CHATBOT_MAX_MESSAGE_LENGTH = 1000
CHATBOT_RATE_LIMIT = 10
CHATBOT_RATE_WINDOW_SECONDS = 60
LOGIN_RATE_LIMIT = 10
REGISTER_RATE_LIMIT = 5
AUTH_RATE_WINDOW_SECONDS = 60
COMMON_WEAK_PASSWORDS = {
    "12345678",
    "password",
    "password123",
    "qwerty123",
    "11111111",
    "admin123",
    "letmein",
}
PASSWORD_RULE_MESSAGE = (
    "Password is too weak. It must be at least 8 characters and include uppercase letter, "
    "lowercase letter, number, and special character. Do not use common passwords like "
    "12345678 or password123, and do not include your email name in the password. "
    "كلمة المرور ضعيفة. يجب أن تكون 8 أحرف على الأقل وتحتوي على حرف كبير، حرف صغير، رقم، ورمز خاص."
)

logger = logging.getLogger("vitapro")
logging.basicConfig(level=logging.INFO)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing. Create backend/.env from .env.example.")
if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is missing. Create backend/.env from .env.example.")

MODELS_DIR = BASE_DIR / "models"
MODEL_PATHS = {
    "diabetes": MODELS_DIR / "diabetes_risk_model.pkl",
    "obesity": MODELS_DIR / "obesity_model.pkl",
    "hypertension": MODELS_DIR / "hypertension_model.pkl",
    "heart": MODELS_DIR / "heart_model.pkl",
}

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    disease_type = Column(String(32), nullable=False, index=True)
    input_data = Column(JSON, nullable=False)
    prediction = Column(String(32), nullable=False)
    probability = Column(Float, nullable=False)
    percentage = Column(Float, nullable=True)
    risk_label = Column(String(80), nullable=True)
    model_probability = Column(Float, nullable=True)
    important_inputs = Column(JSON, nullable=True)
    warning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    user = relationship("User", back_populates="predictions")


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any]


class PredictionResponse(BaseModel):
    prediction: Any
    probability: float
    model_probability: float | None = None
    risk_level: str | None = None
    warning: str | None = None


class HistoryItem(BaseModel):
    id: int
    check_type: str
    percentage: float
    risk_label: str | None = None
    model_probability: float | None = None
    important_inputs: dict[str, Any] | None = None
    warning: str | None = None
    prediction: Any
    created_at: str


class ChatRequest(BaseModel):
    message: str
    latest_result: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    reply: str


class DiabetesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    HighBP: int = Field(..., ge=0, le=1)
    HighChol: int = Field(..., ge=0, le=1)
    CholCheck: int = Field(..., ge=0, le=1)
    BMI: float = Field(..., ge=10, le=60)
    Smoker: int = Field(..., ge=0, le=1)
    Stroke: int = Field(..., ge=0, le=1)
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1)
    PhysActivity: int = Field(..., ge=0, le=1)
    Fruits: int = Field(..., ge=0, le=1)
    Veggies: int = Field(..., ge=0, le=1)
    HvyAlcoholConsump: int = Field(..., ge=0, le=1)
    AnyHealthcare: int = Field(..., ge=0, le=1)
    NoDocbcCost: int = Field(..., ge=0, le=1)
    GenHlth: int = Field(..., ge=1, le=5)
    MentHlth: int = Field(..., ge=0, le=30)
    PhysHlth: int = Field(..., ge=0, le=30)
    DiffWalk: int = Field(..., ge=0, le=1)
    Sex: int = Field(..., ge=0, le=1)
    Age: int = Field(..., ge=1, le=13)
    Education: int = Field(..., ge=1, le=6)
    Income: int = Field(..., ge=1, le=8)


class FlexibleDiseaseInput(BaseModel):
    model_config = ConfigDict(extra="allow")


def parse_cors_origins(value: str | None) -> list[str]:
    if not value:
        return ["http://127.0.0.1:5500", "http://localhost:5500"]
    origins = [origin.strip().strip("[]") for origin in value.split(",")]
    return [origin for origin in origins if origin]


app = FastAPI(title="VitaPro AI Backend", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(os.getenv("CORS_ORIGINS")),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot_rate_limits: dict[str, list[float]] = {}
auth_rate_limits: dict[str, list[float]] = {}


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    for error in exc.errors():
        location = error.get("loc", ())
        if "email" in location:
            return JSONResponse(status_code=422, content={"detail": "Invalid email format."})
        if "password" in location:
            return JSONResponse(status_code=422, content={"detail": PASSWORD_RULE_MESSAGE})
    return JSONResponse(status_code=422, content={"detail": "Invalid request data."})


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8")[:72], password_hash.encode("utf-8"))


def validate_strong_password(password: str, email: str) -> None:
    email_username = email.split("@", 1)[0].lower()
    password_lower = password.lower()
    has_upper = any(char.isupper() for char in password)
    has_lower = any(char.islower() for char in password)
    has_digit = any(char.isdigit() for char in password)
    has_special = any(not char.isalnum() for char in password)

    if (
        len(password) < 8
        or not has_upper
        or not has_lower
        or not has_digit
        or not has_special
        or password_lower in COMMON_WEAK_PASSWORDS
        or (email_username and email_username in password_lower)
    ):
        raise HTTPException(status_code=400, detail=PASSWORD_RULE_MESSAGE)


def create_access_token(user: User) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": str(user.id), "email": user.email, "exp": expires_at}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_current_user(authorization: str | None = Header(None), db: Session = Depends(get_db)) -> User:
    if not authorization:
        raise HTTPException(status_code=401, detail="Bearer token required")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    with path.open("rb") as model_file:
        return pickle.load(model_file)


models = {name: load_model(path) for name, path in MODEL_PATHS.items()}
for model in models.values():
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
model_features = {name: list(getattr(model, "feature_names_in_", [])) for name, model in models.items()}

FIELD_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "heart": {
        "male": (0, 1),
        "age": (1, 120),
        "education": (1, 4),
        "currentSmoker": (0, 1),
        "cigsPerDay": (0, 100),
        "BPMeds": (0, 1),
        "prevalentStroke": (0, 1),
        "prevalentHyp": (0, 1),
        "diabetes": (0, 1),
        "totChol": (80, 500),
        "sysBP": (60, 260),
        "diaBP": (40, 160),
        "BMI": (10, 70),
        "heartRate": (30, 220),
        "glucose": (40, 500),
    },
    "hypertension": {
        "age": (1, 120),
        "bmi": (10, 70),
        "family_history_hypertension": (0, 1),
        "diabetes": (0, 1),
        "smoking": (0, 1),
        "alcohol_heavy": (0, 1),
        "physically_active": (0, 1),
        "high_salt_diet": (0, 1),
        "stroke_history": (0, 1),
        "myocardial_infarction": (0, 1),
        "heart_failure": (0, 1),
        "total_cholesterol_mg_dl": (80, 500),
        "ldl_mg_dl": (20, 350),
        "hdl_mg_dl": (10, 150),
        "creatinine_mg_dl": (0.2, 15),
        "sex_Male": (0, 1),
        "residence_Urban": (0, 1),
    },
    "obesity": {
        "Gender": (0, 1),
        "Age": (1, 120),
        "Height": (0.8, 2.5),
        "Weight": (20, 350),
        "family_history_with_overweight": (0, 1),
        "FAVC": (0, 1),
        "FCVC": (1, 3),
        "NCP": (1, 6),
        "CAEC": (0, 3),
        "SMOKE": (0, 1),
        "CH2O": (0, 5),
        "SCC": (0, 1),
        "FAF": (0, 3),
        "TUE": (0, 3),
        "CALC": (0, 3),
        "BMI": (10, 80),
        "MTRANS_Automobile": (0, 1),
        "MTRANS_Bike": (0, 1),
        "MTRANS_Motorbike": (0, 1),
        "MTRANS_Public_Transportation": (0, 1),
        "MTRANS_Walking": (0, 1),
    },
}


def enrich_diabetes_features(data: dict[str, Any]) -> dict[str, Any]:
    features = dict(data)
    bmi = float(features["BMI"])
    age = float(features["Age"])
    conditions = int(features["HighBP"]) + int(features["HighChol"]) + int(features["Stroke"]) + int(features["HeartDiseaseorAttack"])
    lifestyle = int(features["PhysActivity"]) + int(features["Fruits"]) + int(features["Veggies"]) + int(1 - features["Smoker"]) + int(1 - features["HvyAlcoholConsump"])
    features.update(
        {
            "risk_score": conditions + int(features["DiffWalk"]) + int(features["GenHlth"] >= 4),
            "lifestyle_score": lifestyle,
            "is_overweight": int(bmi >= 25),
            "is_obese": int(bmi >= 30),
            "age_bmi_interaction": age * bmi,
            "critical_risk": int(conditions >= 2),
            "poor_health": int(features["GenHlth"] >= 4 or features["PhysHlth"] >= 15),
            "multiple_conditions": int(conditions >= 2),
        }
    )
    return features


def risk_label_from_probability(probability: float) -> str:
    if probability >= 0.65:
        return "High Risk"
    if probability >= 0.35:
        return "Moderate Risk"
    return "Low Risk"


def obesity_risk_label(data: dict[str, Any]) -> str:
    bmi = float(data.get("BMI", 0))
    if bmi >= 30:
        return "High Obesity Risk"
    if bmi >= 25:
        return "Elevated Obesity Risk"
    if bmi >= 18.5:
        return "Healthy BMI Range"
    return "Low BMI Range"


def validate_model_ranges(disease_type: str, input_data: dict[str, Any]) -> None:
    for field_name, (low, high) in FIELD_RANGES.get(disease_type, {}).items():
        if field_name not in input_data:
            continue
        numeric_value = float(input_data[field_name])
        if numeric_value < low or numeric_value > high:
            raise HTTPException(status_code=400, detail=f"{field_name} must be between {low:g} and {high:g}.")


def apply_prediction_safety(
    disease_type: str,
    input_data: dict[str, Any],
    prediction: Any,
    probability: float,
) -> dict[str, Any]:
    adjusted_probability = probability
    adjusted_prediction = prediction
    warning: str | None = None

    if disease_type == "heart":
        age = float(input_data.get("age", 0))
        sys_bp = float(input_data.get("sysBP", 0))
        dia_bp = float(input_data.get("diaBP", 0))
        cholesterol = float(input_data.get("totChol", 0))
        bmi = float(input_data.get("BMI", 0))
        glucose = float(input_data.get("glucose", 0))
        smoker = int(float(input_data.get("currentSmoker", 0))) == 1
        cigs = float(input_data.get("cigsPerDay", 0))
        diabetes = int(float(input_data.get("diabetes", 0))) == 1
        hyp = int(float(input_data.get("prevalentHyp", 0))) == 1 or sys_bp >= 140 or dia_bp >= 90
        stroke = int(float(input_data.get("prevalentStroke", 0))) == 1
        bp_meds = int(float(input_data.get("BPMeds", 0))) == 1
        chest_pain = int(float(input_data.get("chest_pain", 0))) == 1
        family_history = int(float(input_data.get("family_history", 0))) == 1

        risk_points = 0
        risk_points += 2 if age >= 65 else 1 if age >= 55 else 0
        risk_points += 2 if smoker and cigs >= 20 else 1 if smoker else 0
        risk_points += 1 if diabetes else 0
        risk_points += 2 if hyp else 0
        risk_points += 1 if cholesterol >= 240 else 0
        risk_points += 1 if bmi >= 30 else 0
        risk_points += 1 if glucose >= 126 else 0
        risk_points += 2 if stroke else 0
        risk_points += 1 if bp_meds else 0
        risk_points += 2 if chest_pain else 0
        risk_points += 1 if family_history else 0

        if risk_points >= 6 and adjusted_probability < 0.65:
            adjusted_probability = 0.65
            adjusted_prediction = 1
            warning = (
                "Several major risk factors were present, so Vita shows a higher safety risk than the raw model score. "
                "This is educational only; please speak with a doctor for personal medical advice."
            )
        elif risk_points >= 4 and adjusted_probability < 0.35:
            adjusted_probability = 0.35
            warning = (
                "Several risk factors were present even though the raw model score was low. "
                "Please review the inputs and consider medical advice."
            )
        if chest_pain:
            urgent = "If chest pain is severe, current, or comes with breathing trouble, fainting, or sweating, seek emergency help now."
            warning = f"{warning} {urgent}" if warning else urgent

    if disease_type == "obesity":
        warning = (
            "The obesity model is multiclass, so this percentage is model confidence. "
            "The displayed risk label also considers BMI."
        )

    return {
        "prediction": adjusted_prediction,
        "probability": max(0.0, min(1.0, adjusted_probability)),
        "model_probability": max(0.0, min(1.0, probability)),
        "risk_level": obesity_risk_label(input_data) if disease_type == "obesity" else risk_label_from_probability(adjusted_probability),
        "warning": warning,
    }


def predict(disease_type: str, input_data: dict[str, Any]) -> dict[str, Any]:
    model = models[disease_type]
    features = model_features[disease_type]
    model_input = enrich_diabetes_features(input_data) if disease_type == "diabetes" else input_data
    missing = [feature for feature in features if feature not in model_input]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing, "required_features": features})

    feature_values = [model_input[feature] for feature in features]
    frame = pd.DataFrame([feature_values], columns=features)
    if PREDICTION_DEBUG_LOGS:
        logger.info("Prediction raw input [%s]: %s", disease_type, input_data)
        logger.info("Prediction cleaned input [%s]: %s", disease_type, model_input)
        logger.info("Prediction feature order [%s]: %s", disease_type, features)
        logger.info("Prediction feature array [%s]: %s", disease_type, feature_values)
    raw_prediction = model.predict(frame)[0]
    prediction = raw_prediction.item() if hasattr(raw_prediction, "item") else raw_prediction

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        classes = list(getattr(model, "classes_", []))
        if len(classes) == 2 and 1 in classes:
            probability = float(probabilities[classes.index(1)])
        else:
            probability = float(max(probabilities))
    else:
        probability = float(prediction) if isinstance(prediction, (int, float)) else 0.0

    probability = max(0.0, min(1.0, probability))
    result = apply_prediction_safety(disease_type, input_data, prediction, probability)
    if PREDICTION_DEBUG_LOGS:
        logger.info("Prediction result [%s]: prediction=%s probability=%s", disease_type, prediction, probability)
        logger.info("Prediction response [%s]: %s", disease_type, result)
    else:
        logger.info("Prediction completed [%s]: risk=%s probability=%.4f", disease_type, result.get("risk_level"), result.get("probability", 0.0))
    return result


def validate_flexible_prediction_input(input_data: dict[str, Any]) -> None:
    if not input_data:
        raise HTTPException(status_code=400, detail="Prediction input is required.")

    for field_name, value in input_data.items():
        if value is None or value == "":
            raise HTTPException(status_code=400, detail="All prediction values are required.")
        if isinstance(value, bool):
            continue
        if isinstance(value, str):
            try:
                numeric_value = float(value)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Prediction values must be numeric.") from exc
        elif isinstance(value, (int, float)):
            numeric_value = float(value)
        else:
            raise HTTPException(status_code=400, detail="Prediction values must be numeric.")

        if not -1 <= numeric_value <= 1000:
            raise HTTPException(status_code=400, detail=f"{field_name} has an invalid value.")


def migrate_prediction_history_columns() -> None:
    existing = {column["name"] for column in inspect(engine).get_columns("predictions")}
    column_sql = {
        "percentage": "ALTER TABLE predictions ADD COLUMN percentage FLOAT NULL",
        "risk_label": "ALTER TABLE predictions ADD COLUMN risk_label VARCHAR(80) NULL",
        "model_probability": "ALTER TABLE predictions ADD COLUMN model_probability FLOAT NULL",
        "important_inputs": "ALTER TABLE predictions ADD COLUMN important_inputs JSON NULL",
        "warning": "ALTER TABLE predictions ADD COLUMN warning TEXT NULL",
    }
    with engine.begin() as connection:
        for column_name, statement in column_sql.items():
            if column_name not in existing:
                connection.execute(sql_text(statement))


def save_prediction(db: Session, user: User, disease_type: str, input_data: dict[str, Any], result: dict[str, Any]) -> None:
    probability = float(result["probability"])
    row = Prediction(
        user_id=user.id,
        disease_type=disease_type,
        input_data=input_data,
        prediction=str(result["prediction"]),
        probability=probability,
        percentage=round(probability * 100, 3),
        risk_label=result.get("risk_level"),
        model_probability=result.get("model_probability"),
        important_inputs=summarize_prediction_inputs(disease_type, input_data),
        warning=result.get("warning"),
    )
    db.add(row)
    db.commit()


def summarize_prediction_inputs(disease_type: str, data: dict[str, Any]) -> dict[str, Any]:
    if disease_type == "heart":
        return {
            "age": data.get("age"),
            "gender": "male" if data.get("male") == 1 else "female",
            "smoker": "yes" if data.get("currentSmoker") == 1 else "no",
            "cigarettes_per_day": data.get("cigsPerDay"),
            "blood_pressure": f"{data.get('sysBP')}/{data.get('diaBP')}",
            "high_blood_pressure_or_meds": "yes" if data.get("prevalentHyp") == 1 or data.get("BPMeds") == 1 else "no",
            "cholesterol": data.get("totChol"),
            "diabetes": "yes" if data.get("diabetes") == 1 else "no",
            "glucose": data.get("glucose"),
            "bmi": data.get("BMI"),
            "heart_rate": data.get("heartRate"),
            "previous_stroke": "yes" if data.get("prevalentStroke") == 1 else "no",
            "family_history": "yes" if data.get("family_history") == 1 else "no",
            "chest_pain": "yes" if data.get("chest_pain") == 1 else "no",
        }
    if disease_type == "diabetes":
        return {
            "age_category": data.get("Age"),
            "bmi": data.get("BMI"),
            "high_blood_pressure": "yes" if data.get("HighBP") == 1 else "no",
            "high_cholesterol": "yes" if data.get("HighChol") == 1 else "no",
            "smoker": "yes" if data.get("Smoker") == 1 else "no",
            "stroke_history": "yes" if data.get("Stroke") == 1 else "no",
            "heart_disease_or_attack": "yes" if data.get("HeartDiseaseorAttack") == 1 else "no",
            "physical_activity": "yes" if data.get("PhysActivity") == 1 else "no",
            "general_health_score": data.get("GenHlth"),
        }
    if disease_type == "hypertension":
        return {
            "age": data.get("age"),
            "bmi": data.get("bmi"),
            "family_history": "yes" if data.get("family_history_hypertension") == 1 else "no",
            "diabetes": "yes" if data.get("diabetes") == 1 else "no",
            "smoker": "yes" if data.get("smoking") == 1 else "no",
            "high_salt_diet": "yes" if data.get("high_salt_diet") == 1 else "no",
            "physically_active": "yes" if data.get("physically_active") == 1 else "no",
            "cholesterol": data.get("total_cholesterol_mg_dl"),
            "ldl": data.get("ldl_mg_dl"),
            "hdl": data.get("hdl_mg_dl"),
        }
    if disease_type == "obesity":
        return {
            "age": data.get("Age"),
            "gender": "male" if data.get("Gender") == 1 else "female",
            "height_m": data.get("Height"),
            "weight_kg": data.get("Weight"),
            "bmi": data.get("BMI"),
            "family_history_with_overweight": "yes" if data.get("family_history_with_overweight") == 1 else "no",
            "frequent_high_calorie_food": "yes" if data.get("FAVC") == 1 else "no",
            "smoker": "yes" if data.get("SMOKE") == 1 else "no",
            "physical_activity_score": data.get("FAF"),
            "water_intake_score": data.get("CH2O"),
        }
    return {}


def history_item(row: Prediction) -> dict[str, Any]:
    percentage = row.percentage if row.percentage is not None else round(float(row.probability) * 100, 3)
    return {
        "id": row.id,
        "check_type": row.disease_type,
        "percentage": percentage,
        "risk_label": row.risk_label or risk_label_from_probability(float(row.probability)),
        "model_probability": row.model_probability if row.model_probability is not None else float(row.probability),
        "important_inputs": row.important_inputs or row.input_data,
        "warning": row.warning,
        "prediction": row.prediction,
        "created_at": row.created_at.isoformat() if row.created_at else "",
    }


def disease_display_name(disease_type: str | None) -> str:
    names = {
        "heart": "Heart Risk",
        "diabetes": "Diabetes Risk",
        "hypertension": "Hypertension Risk",
        "obesity": "Obesity Risk",
    }
    return names.get(str(disease_type or ""), str(disease_type or "Health Check"))


def latest_history_context(db: Session, user: User) -> dict[str, Any] | None:
    row = (
        db.query(Prediction)
        .filter(Prediction.user_id == user.id)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .first()
    )
    if not row:
        return None
    item = history_item(row)
    return {
        "type": disease_display_name(item["check_type"]),
        "percentage": f"{item['percentage']}%",
        "risk_label": item["risk_label"],
        "inputs": item["important_inputs"],
        "timestamp": item["created_at"],
    }


def enforce_chatbot_rate_limit(user: User, request: Request) -> None:
    client_host = request.client.host if request.client else "unknown"
    key = f"user:{user.id}:ip:{client_host}"
    now = time.monotonic()
    recent_requests = [
        timestamp
        for timestamp in chatbot_rate_limits.get(key, [])
        if now - timestamp < CHATBOT_RATE_WINDOW_SECONDS
    ]
    if len(recent_requests) >= CHATBOT_RATE_LIMIT:
        chatbot_rate_limits[key] = recent_requests
        raise HTTPException(status_code=429, detail="Too many chatbot requests. Please wait a minute and try again.")
    recent_requests.append(now)
    chatbot_rate_limits[key] = recent_requests


def enforce_auth_rate_limit(request: Request, action: str, identifier: str, limit: int) -> None:
    client_host = request.client.host if request.client else "unknown"
    normalized_identifier = identifier.lower()[:128]
    key = f"{action}:ip:{client_host}:id:{normalized_identifier}"
    now = time.monotonic()
    recent_requests = [
        timestamp
        for timestamp in auth_rate_limits.get(key, [])
        if now - timestamp < AUTH_RATE_WINDOW_SECONDS
    ]
    if len(recent_requests) >= limit:
        auth_rate_limits[key] = recent_requests
        raise HTTPException(status_code=429, detail="Too many attempts. Please wait a minute and try again.")
    recent_requests.append(now)
    auth_rate_limits[key] = recent_requests


def extract_openrouter_reply(data: dict[str, Any]) -> str:
    try:
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected OpenRouter response format") from exc
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("Empty OpenRouter reply")
    return clean_chatbot_plain_text(reply)


def clean_chatbot_plain_text(reply: str) -> str:
    text = reply.strip()
    replacements = {
        "**": "",
        "__": "",
        "`": "",
        "###": "",
        "##": "",
        "#": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        while stripped.startswith(("- ", "* ", "• ")):
            stripped = stripped[2:].strip()
        cleaned_lines.append(stripped)
    return "\n".join(line for line in cleaned_lines if line).strip()


def build_local_chatbot_reply(message: str) -> str:
    arabic_chars = sum(1 for char in message if "\u0600" <= char <= "\u06ff")
    is_arabic = arabic_chars > 0
    lower_message = message.lower()

    if is_arabic:
        if any(word in message for word in ["ألم شديد في الصدر", "ألم صدر", "الصدر", "ضيق تنفس", "إغماء", "جلطة", "سكتة"]):
            return (
                "إذا كان لديك ألم شديد في الصدر، صعوبة شديدة في التنفس، إغماء، أعراض جلطة، أو أي عرض يهدد الحياة، "
                "اتصل بالطوارئ فورًا أو اذهب لأقرب طوارئ. لا تنتظر رد المساعد في الحالات العاجلة."
            )
        if "bmi" in lower_message or "مؤشر كتلة" in message or "كتلة الجسم" in message:
            return (
                "BMI يعني مؤشر كتلة الجسم. هو رقم تقريبي يقارن الوزن بالطول، ويُحسب بقسمة الوزن بالكيلوجرام على مربع الطول بالمتر. "
                "يساعد في فهم الوزن بشكل عام، لكنه لا يشخص الصحة وحده. للنصيحة الطبية الشخصية، راجع طبيبًا."
            )
        if any(word in message for word in ["نصائح", "نصيحة", "صحية", "صحتي", "الصحة", "عامة"]):
            return (
                "لصحة أفضل، حاول تناول وجبات متوازنة فيها خضار وبروتين وحبوب كاملة. "
                "اشرب مياه كفاية، تحرك يوميًا مثل المشي، نم 7 إلى 9 ساعات، وقلل السكريات والتدخين. "
                "يمكنك أيضًا متابعة الوزن ومؤشر كتلة الجسم وضغط الدم عند الحاجة. وللنصيحة الطبية الشخصية، راجع طبيبًا."
            )
        if "bmi" in lower_message or "مؤشر كتلة" in message or "كتلة الجسم" in message:
            return (
                "BMI يعني مؤشر كتلة الجسم. هو رقم تقريبي يساعد على فهم علاقة الوزن بالطول. "
                "يتم حسابه بقسمة الوزن بالكيلو على مربع الطول بالمتر. "
                "هو أداة تعليمية فقط، ولا يكفي وحده لتقييم الصحة أو تشخيص أي حالة."
            )
        emergency = any(word in message for word in ["ألم صدر", "صدر", "تنفس", "إغماء", "جلطة", "سكتة"])
        if emergency:
            return (
                "إذا لديك ألم صدر شديد، صعوبة تنفس، إغماء، أعراض جلطة، أو أي عرض خطير، "
                "اتصل بالطوارئ فوراً. لا تعتمد على فيتا في الحالات الطارئة."
            )
        if any(word in message for word in ["قلب", "نتيجة", "مخاطر"]):
            return (
                "بعد نتيجة فحص القلب، ركز على المتابعة مع طبيب إذا كانت النتيجة مرتفعة أو لديك أعراض. "
                "حافظ على ضغط وسكر ووزن صحي، وابتعد عن التدخين، وامشِ بانتظام إذا كان ذلك مناسباً لك."
            )
        return (
            "أنا مساعد فيتا. أستطيع تقديم معلومات صحية تعليمية عامة فقط، ولا أشخص أو أصف علاجاً. "
            "للحصول على نصيحة شخصية دقيقة، راجع طبيباً، خاصة إذا لديك أعراض مستمرة أو شديدة."
        )

    emergency_terms = ["chest pain", "shortness of breath", "faint", "stroke", "emergency", "severe breathing"]
    if "bmi" in lower_message or "body mass index" in lower_message:
        return (
            "BMI means Body Mass Index. It is a simple estimate that compares weight with height. "
            "It can be useful for education, but it does not diagnose health by itself."
        )
    if any(term in lower_message for term in emergency_terms):
        return (
            "If you have chest pain, severe breathing trouble, fainting, stroke symptoms, "
            "or any life-threatening symptom, call emergency services now."
        )
    if any(term in lower_message for term in ["heart", "risk", "result"]):
        return (
            "After a heart risk result, use it as education, not a diagnosis. If risk is high or you have symptoms, "
            "see a doctor. Helpful next steps include checking blood pressure, blood sugar, cholesterol, avoiding smoking, "
            "and doing safe regular activity."
        )
    return (
        "For general health, try balanced meals with vegetables, protein, and whole grains. "
        "Drink enough water, move daily like walking, sleep 7-9 hours, and reduce sugary foods and smoking. "
        "You can also track weight, BMI, and blood pressure when needed. For personal medical advice, please see a doctor."
    )


def message_is_arabic(message: str) -> bool:
    return sum(1 for char in message if "\u0600" <= char <= "\u06ff") > 0


def message_asks_about_latest_result(message: str) -> bool:
    lower_message = message.lower()
    english_terms = [
        "according to my result",
        "according to my results",
        "my result",
        "my results",
        "what should i do",
        "explain my result",
        "explain my results",
        "recommendation",
        "recommendations",
        "advice based on",
    ]
    arabic_terms = [
        "\u062d\u0633\u0628 \u0646\u062a\u064a\u062c\u062a\u064a",
        "\u062d\u0633\u0628 \u0627\u0644\u0646\u062a\u064a\u062c\u0629",
        "\u0627\u0634\u0631\u062d \u0646\u062a\u064a\u062c\u062a\u064a",
        "\u0627\u0634\u0631\u062d\u064a \u0646\u062a\u064a\u062c\u062a\u064a",
        "\u0627\u0634\u0631\u062d \u0646\u062a\u064a\u062c\u062a\u0649",
        "\u0646\u0635\u064a\u062d\u0629 \u062d\u0633\u0628",
        "\u0646\u0635\u0627\u0626\u062d \u062d\u0633\u0628",
        "\u0645\u0627\u0630\u0627 \u0623\u0641\u0639\u0644",
        "\u0627\u0639\u0645\u0644 \u0627\u064a\u0647",
        "\u0623\u0639\u0645\u0644 \u0625\u064a\u0647",
        "\u0627\u062f\u064a\u0646\u064a \u0646\u0635\u064a\u062d\u0629",
        "\u0627\u062f\u064a\u0646\u064a \u0646\u0635\u0627\u064a\u062d",
    ]
    return any(term in lower_message for term in english_terms) or any(term in message for term in arabic_terms)


def clean_latest_prediction_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(context, dict):
        return None
    cleaned: dict[str, Any] = {}
    for key in ["type", "percentage", "risk_label", "timestamp"]:
        value = context.get(key)
        if isinstance(value, (str, int, float)):
            cleaned[key] = str(value)[:120]
    inputs = context.get("inputs")
    if isinstance(inputs, dict):
        cleaned_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            if len(cleaned_inputs) >= 20:
                break
            if isinstance(value, (str, int, float, bool)):
                cleaned_inputs[str(key)[:60]] = str(value)[:120]
        cleaned["inputs"] = cleaned_inputs
    if not cleaned.get("type") or not cleaned.get("percentage") or not cleaned.get("risk_label"):
        return None
    return cleaned


def format_latest_prediction_context(context: dict[str, Any] | None) -> str:
    cleaned = clean_latest_prediction_context(context)
    if not cleaned:
        return "No latest prediction result was provided by the frontend."
    input_lines = []
    inputs = cleaned.get("inputs", {})
    if isinstance(inputs, dict):
        input_lines = [f"- {key}: {value}" for key, value in inputs.items()]
    return (
        "Latest prediction result from the user's actual form submission:\n"
        f"Prediction type: {cleaned.get('type')}\n"
        f"Percentage: {cleaned.get('percentage')}\n"
        f"Risk label: {cleaned.get('risk_label')}\n"
        f"Timestamp: {cleaned.get('timestamp', 'not provided')}\n"
        "Important inputs:\n"
        + ("\n".join(input_lines) if input_lines else "- none provided")
    )


def arabic_context_label(value: Any) -> str:
    labels = {
        "Heart Risk": "مخاطر القلب",
        "Diabetes Risk": "مخاطر السكري",
        "Hypertension Risk": "مخاطر ضغط الدم",
        "Obesity Risk": "مخاطر السمنة",
        "Low Risk": "مخاطر منخفضة",
        "Moderate Risk": "مخاطر متوسطة",
        "High Risk": "مخاطر عالية",
        "Healthy BMI Range": "مؤشر كتلة الجسم في النطاق الصحي",
        "High Obesity Risk": "مخاطر سمنة عالية",
        "Elevated Obesity Risk": "مخاطر سمنة مرتفعة",
        "heart": "مخاطر القلب",
        "diabetes": "مخاطر السكري",
        "hypertension": "مخاطر ضغط الدم",
        "obesity": "مخاطر السمنة",
        "age": "العمر",
        "smoker": "التدخين",
        "cigarettes_per_day": "عدد السجائر يوميًا",
        "high_blood_pressure_or_meds": "ارتفاع الضغط أو أدوية الضغط",
        "blood_pressure": "ضغط الدم",
        "cholesterol": "الكوليسترول",
        "diabetes": "السكري",
        "family_history_with_overweight": "تاريخ عائلي لزيادة الوزن",
        "frequent_high_calorie_food": "أطعمة عالية السعرات بكثرة",
        "physical_activity_score": "درجة النشاط البدني",
        "water_intake_score": "درجة شرب المياه",
        "chest_pain": "ألم الصدر",
        "bmi": "مؤشر كتلة الجسم",
        "glucose": "سكر الدم",
        "yes": "نعم",
        "no": "لا",
    }
    return labels.get(str(value), str(value))


def build_contextual_local_chatbot_reply(message: str, context: dict[str, Any] | None) -> str | None:
    if not message_asks_about_latest_result(message):
        return None
    cleaned = clean_latest_prediction_context(context)
    is_arabic = message_is_arabic(message)
    if not cleaned:
        return (
            "لا أرى نتيجة فحص حديثة محفوظة الآن. أكمل فحصًا صحيًا أولًا، ثم أستطيع إعطاء نصيحة حسب نتيجتك."
            if is_arabic
            else "I do not see a recent test result yet. Please complete a health check first, then I can give advice based on it."
        )

    prediction_type = cleaned.get("type", "health check")
    percentage = cleaned.get("percentage", "")
    risk_label = cleaned.get("risk_label", "")
    inputs = cleaned.get("inputs", {}) if isinstance(cleaned.get("inputs"), dict) else {}
    risk_notes: list[str] = []
    for key, value in inputs.items():
        text = f"{key}: {value}".lower()
        if any(term in text for term in ["smok", "cigarette", "تدخ", "yes", "high", "مرتفع", "cholesterol", "blood pressure", "sysbp", "diabetes", "chest", "stroke"]):
            risk_notes.append(f"{key}: {value}")
    if is_arabic:
        arabic_type = arabic_context_label(prediction_type)
        arabic_risk = arabic_context_label(risk_label)
        arabic_notes = []
        for note in risk_notes[:5]:
            if ":" in note:
                key, value = note.split(":", 1)
                arabic_notes.append(f"{arabic_context_label(key.strip())}: {arabic_context_label(value.strip())}")
            else:
                arabic_notes.append(note)
        notes = " من العوامل المهمة: " + "، ".join(arabic_notes) + "." if arabic_notes else ""
        return (
            f"نتيجتك الأخيرة هي {arabic_type}: {percentage} وتصنيفها {arabic_risk}.{notes} "
            "هذه ليست تشخيصًا. للحفاظ على صحتك، ركز على أكل متوازن، حركة يومية مثل المشي، نوم كاف، تقليل السكر والملح، وتجنب التدخين. "
            "إذا كانت النتيجة مرتفعة أو لديك ألم صدر أو ضيق تنفس أو أعراض قوية، راجع طبيبًا أو الطوارئ حسب شدة الأعراض."
        )
    notes = " Important factors I see: " + ", ".join(risk_notes[:5]) + "." if risk_notes else ""
    return (
        f"Your latest result is {prediction_type}: {percentage}, classified as {risk_label}.{notes} "
        "This is not a diagnosis. To improve or maintain your result, focus on balanced meals, regular walking or movement, good sleep, reducing sugar and salt, and avoiding smoking. "
        "If the result is high or you have symptoms like chest pain or breathing trouble, please see a doctor or seek urgent help."
    )


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    migrate_prediction_history_columns()


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "database": "mysql",
        "models_loaded": sorted(models.keys()),
        "model_features": model_features,
    }


@app.post("/auth/register", response_model=AuthResponse)
def register(payload: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    email = payload.email.lower()
    enforce_auth_rate_limit(request, "register", email, REGISTER_RATE_LIMIT)
    validate_strong_password(payload.password, email)
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(name=payload.name, email=email, password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"access_token": create_access_token(user), "user": {"id": user.id, "name": user.name, "email": user.email}}


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    email = payload.email.lower()
    enforce_auth_rate_limit(request, "login", email, LOGIN_RATE_LIMIT)
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(payload.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"access_token": create_access_token(user), "user": {"id": user.id, "name": user.name, "email": user.email}}


@app.get("/history", response_model=list[HistoryItem])
def get_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(Prediction)
        .filter(Prediction.user_id == user.id)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .all()
    )
    return [history_item(row) for row in rows]


@app.post("/chatbot", response_model=ChatResponse)
async def chatbot(payload: ChatRequest, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    message = payload.message.strip() if isinstance(payload.message, str) else ""
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty.")
    if len(message) > CHATBOT_MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail="Message is too long. Please keep it under 1000 characters.")

    enforce_chatbot_rate_limit(user, request)
    latest_context = clean_latest_prediction_context(payload.latest_result) or clean_latest_prediction_context(latest_history_context(db, user))

    if message_asks_about_latest_result(message):
        contextual_reply = build_contextual_local_chatbot_reply(message, latest_context)
        if contextual_reply:
            return {"reply": contextual_reply}

    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY.startswith("your_"):
        logger.warning("OpenRouter API key is not configured; using local chatbot fallback.")
        contextual_reply = build_contextual_local_chatbot_reply(message, latest_context)
        return {"reply": contextual_reply or build_local_chatbot_reply(message)}

    latest_context_text = format_latest_prediction_context(latest_context)

    body = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are VitaPro Assistant, a safe and friendly health education chatbot. "
                    "Reply in the same language the user writes. If the user writes Arabic, reply in Arabic. "
                    "If the user writes English, reply in English. Do not switch language unless the user asks. "
                    "You can explain VitaPro risk results in simple Arabic or English. "
                    "When a latest prediction result is provided, use that exact result for advice, explanation, and recommendations. "
                    "Mention the actual prediction type, percentage, risk label, and relevant user inputs from the context. "
                    "Do not invent or guess results. If no latest result is provided and the user asks about their result, say they should complete a health check first. "
                    "Make this reusable for heart, diabetes, hypertension, obesity, or any Vita prediction result. "
                    "You can answer general wellness, BMI, diabetes, hypertension, heart disease, and obesity questions. "
                    "When users ask for general health advice, give practical educational wellness tips such as balanced meals, "
                    "drinking water, regular walking or movement, good sleep, reducing sugar, avoiding smoking, checking weight or BMI, "
                    "and monitoring blood pressure when useful. "
                    "Arabic users can receive the same practical wellness tips in Arabic. "
                    "You must not diagnose users. You must not prescribe medicine. You must not replace a doctor. "
                    "You must not give personal treatment plans. "
                    "You must tell users to see a doctor for serious symptoms. "
                    "You must tell users to call emergency services for symptoms like chest pain, severe breathing problems, "
                    "fainting, stroke symptoms, or any life-threatening symptoms. "
                    "Include a short note such as: For personal medical advice, please see a doctor. "
                    "Do not make the disclaimer the whole answer. Keep answers short, practical, friendly, clear, supportive, and educational. "
                    "Reply in plain text only. Do not use Markdown. Do not use bold symbols like **. "
                    "Do not use headings with #. Do not use tables. Do not use code blocks. "
                    "Keep Arabic answers clean, natural, and easy to read."
                ),
            },
            {"role": "system", "content": latest_context_text},
            {"role": "user", "content": message},
        ],
        "temperature": 0.4,
        "max_tokens": 500,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://127.0.0.1:5500",
        "X-Title": "VitaPro",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(OPENROUTER_URL, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            return {"reply": extract_openrouter_reply(data)}
    except httpx.TimeoutException as exc:
        logger.warning("OpenRouter chatbot request timed out: %s", exc)
        return {"reply": build_contextual_local_chatbot_reply(message, latest_context) or build_local_chatbot_reply(message)}
    except httpx.HTTPStatusError as exc:
        logger.warning("OpenRouter chatbot request failed with status %s", exc.response.status_code)
        return {"reply": build_contextual_local_chatbot_reply(message, latest_context) or build_local_chatbot_reply(message)}
    except (httpx.RequestError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("OpenRouter chatbot request failed: %s", exc)
        return {"reply": build_contextual_local_chatbot_reply(message, latest_context) or build_local_chatbot_reply(message)}


@app.post("/predict/diabetes", response_model=PredictionResponse)
def predict_diabetes(payload: DiabetesInput, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    data = payload.model_dump()
    result = predict("diabetes", data)
    save_prediction(db, user, "diabetes", data, result)
    return result


@app.post("/predict/obesity", response_model=PredictionResponse)
def predict_obesity(payload: FlexibleDiseaseInput, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    data = payload.model_dump()
    validate_flexible_prediction_input(data)
    validate_model_ranges("obesity", data)
    result = predict("obesity", data)
    save_prediction(db, user, "obesity", data, result)
    return result


@app.post("/predict/hypertension", response_model=PredictionResponse)
def predict_hypertension(payload: FlexibleDiseaseInput, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    data = payload.model_dump()
    validate_flexible_prediction_input(data)
    validate_model_ranges("hypertension", data)
    result = predict("hypertension", data)
    save_prediction(db, user, "hypertension", data, result)
    return result


@app.post("/predict/heart", response_model=PredictionResponse)
def predict_heart(payload: FlexibleDiseaseInput, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    data = payload.model_dump()
    validate_flexible_prediction_input(data)
    validate_model_ranges("heart", data)
    result = predict("heart", data)
    save_prediction(db, user, "heart", data, result)
    return result
