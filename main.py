from fastapi import FastAPI, HTTPException, Response, Request, Form, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta
import json
import io
from fastapi.responses import StreamingResponse
import csv
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, func, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from uuid import uuid4
import time 
import os
import joblib
import pandas as pd
import numpy as np
import uvicorn
from starlette.middleware.cors import CORSMiddleware
import numpy as np 

# Global variables for model health and stats
STARTUP_TIME = time.time()

# --- NEW: Alert Threshold ---
RISK_THRESHOLD = float(os.environ.get('RISK_THRESHOLD', '0.70')) # 70% probability for a High-Risk Alert

# --- 2. Load Assets (Model, Scaler, Encoders) ---
model: Optional[Any] = None
scaler: Optional[Any] = None
label_encoders: Optional[Dict[str, Any]] = None

# Define the models directory
MODELS_DIR = "models"

# In-memory prediction history (kept small). Each entry: timestamp, input, prediction, probability, contributions
PREDICTIONS_HISTORY: List[Dict[str, Any]] = []
MAX_HISTORY = 200
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "predictions.db")

# SQLAlchemy / Database configuration: allow Postgres via DATABASE_URL env var.
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    SQLALCHEMY_DATABASE_URL = DATABASE_URL
else:
    # Use SQLite file fallback
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine_kwargs = {}
if SQLALCHEMY_DATABASE_URL.startswith('sqlite'):
    # sqlite specific arg to allow usage from multiple threads in dev
    engine_kwargs = {"connect_args": {"check_same_thread": False}}

engine = create_engine(SQLALCHEMY_DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, index=True)
    input_json = Column(Text)
    prediction_status = Column(Integer, index=True)
    probability = Column(Float)
    contributions_json = Column(Text)
    
# --- Placeholder DB Models (Assuming these were in init_db.py) ---
class Alert(Base):
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True, index=True)
    alert_timestamp = Column(DateTime, default=datetime.utcnow)
    entity_id = Column(String, nullable=False)
    risk_signal = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    prediction_score = Column(Float)
    status = Column(String, default="New") # New, In Progress, Closed
    manager_notes = Column(Text, nullable=True)
    resolution_action = Column(Text, nullable=True)
    
class MacroIndicator(Base):
    __tablename__ = 'macro_indicators'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    gdp_growth_rate = Column(Float)
    market_index_yoy = Column(Float)
    unemployment_rate = Column(Float)
    inflation_rate = Column(Float)
    interest_rate = Column(Float)
    
# --- Model Evaluation Table ---
class ModelEvaluation(Base):
    __tablename__ = 'model_evaluations'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_name = Column(String, index=True, nullable=False) # e.g., 'AUC', 'F1-Score'
    metric_value = Column(Float, nullable=False)
    test_size = Column(Integer, nullable=False) # Number of samples in the test set

#Cached SHAP explainer (built at startup for faster explain responses)
SHAP_EXPLAINER = None

#Admin session configuration. Use Redis when REDIS_URL is provided; otherwise fall back to in-memory.
ADMIN_SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL = int(os.environ.get('ADMIN_SESSION_TTL', '3600'))  # seconds
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'password')

# Redis session configuration (optional)
REDIS_URL = os.environ.get('REDIS_URL')
redis_client = None
USE_REDIS = False
if REDIS_URL:
    try:
        import redis as _redis
        redis_client = _redis.from_url(REDIS_URL, decode_responses=True)
        # quick ping to verify connection
        redis_client.ping()
        USE_REDIS = True
        print('Info: Connected to Redis for admin sessions.')
    except Exception as e:
        print('Warning: Could not connect to Redis at REDIS_URL, falling back to in-memory sessions:', e)
        redis_client = None
        USE_REDIS = False

# --- 1. FastAPI App Initialization ---
app = FastAPI(
    title="Bank Early Warning System API",
    description="Predicts the probability of loan default (Status 1) using a Logistic Regression model."
)

#Add CORS middleware to allow the HTML dashboard to communicate with this API
origins = ["*"]  # In production, specify your exact domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _create_admin_session(username: str) -> str:
    token = uuid4().hex
    expires = datetime.utcnow() + timedelta(seconds=SESSION_TTL)
    # Prefer Redis-backed sessions when available
    if USE_REDIS and redis_client:
        try:
            key = f"admin_session:{token}"
            # store username as value, let Redis TTL handle expiry
            redis_client.set(key, username, ex=SESSION_TTL)
            return token
        except Exception as e:
            print('Warning: Redis set failed for admin session, falling back to in-memory:', e)

    # In-memory fallback
    ADMIN_SESSIONS[token] = {'user': username, 'expires': expires}
    return token


def _is_admin_token_valid(token: Optional[str]) -> bool:
    if not token:
        return False
    # Check Redis first when available
    if USE_REDIS and redis_client:
        try:
            val = redis_client.get(f"admin_session:{token}")
            return bool(val)
        except Exception as e:
            print('Warning: Redis get failed for admin session, falling back to in-memory:', e)

    # In-memory fallback check
    info = ADMIN_SESSIONS.get(token)
    if not info:
        return False
    expires = info.get('expires')
    if expires is not None and expires < datetime.utcnow():
        try:
            del ADMIN_SESSIONS[token]
        except Exception:
            pass
        return False
    return True


def _invalidate_admin_token(token: Optional[str]):
    if not token:
        return
    if USE_REDIS and redis_client:
        try:
            redis_client.delete(f"admin_session:{token}")
            return
        except Exception as e:
            print('Warning: Redis delete failed for admin session, falling back to in-memory:', e)

    if token and token in ADMIN_SESSIONS:
        try:
            del ADMIN_SESSIONS[token]
        except Exception:
            pass


def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    # Create tables via SQLAlchemy (Postgres or SQLite depending on DATABASE_URL)
    try:
        Base.metadata.create_all(bind=engine)
        # Seed initial performance data if the table is empty
        seed_initial_performance_data()
        # Seed initial macro indicator data
        seed_initial_macro_indicator()
    except SQLAlchemyError as e:
        print('Warning: Failed to initialize DB with SQLAlchemy:', e)


# --- NEW: Function to seed initial macro indicator data ---
def seed_initial_macro_indicator():
    """Seeds the MacroIndicator table with mock data if it's empty."""
    db = None
    try:
        db = SessionLocal()
        if db.query(MacroIndicator).count() == 0:
            print("Info: Seeding initial macro indicator data.")
            now = datetime.utcnow()
            
            mock_indicators = []
            for i in range(5):
                ts = now - timedelta(days=5 - i)
                # Mock key economic indicators
                mock_indicators.append(MacroIndicator(
                    timestamp=ts,
                    gdp_growth_rate=2.0 + (np.random.rand() - 0.5) * 0.5,  # Example rate between 1.75 and 2.25
                    market_index_yoy=5.0 + (np.random.rand() - 0.5) * 2.0,  # Example rate between 4.0 and 6.0
                    unemployment_rate=4.0 + (np.random.rand() - 0.5) * 1.0,  # Example rate between 3.5 and 4.5
                    inflation_rate=2.5 + (np.random.rand() - 0.5) * 1.0,  # Example rate between 2.0 and 3.0
                    interest_rate=3.5 + (np.random.rand() - 0.5) * 1.0  # Example rate between 3.0 and 4.0
                ))

            db.add_all(mock_indicators)
            db.commit()
            print("Info: Macro indicator data seeded successfully.")
        
    except Exception as e:
        if db:
            try:
                db.rollback()
            except:
                pass
        print(f"Warning: Failed to seed macro indicator data: {e}")
    finally:
        if db:
            try:
                db.close()
            except:
                pass


# --- Function to seed initial performance data ---
def seed_initial_performance_data():
    """Seeds the ModelEvaluation table with mock data if it's empty."""
    db = None
    try:
        db = SessionLocal()
        if db.query(ModelEvaluation).count() == 0:
            print("Info: Seeding initial model performance data.")
            now = datetime.utcnow()
            
            # Generate 5 mock historical entries for each metric
            def generate_mock_data(metric_name, base_value, variance, n=5):
                data = []
                for i in range(n):
                    # Create timestamps at 1-day intervals in the past
                    ts = now - timedelta(days=n - i)
                    val = base_value + (np.random.rand() - 0.5) * variance # Small random change
                    val = max(0.0, min(1.0, val)) # Clamp value between 0 and 1
                    
                    data.append(ModelEvaluation(
                        timestamp=ts,
                        metric_name=metric_name,
                        metric_value=val,
                        test_size=25000 + i * 500
                    ))
                return data

            mock_evals = []
            mock_evals.extend(generate_mock_data('AUC', 0.82, 0.01))
            mock_evals.extend(generate_mock_data('F1-Score', 0.75, 0.015))
            mock_evals.extend(generate_mock_data('Accuracy', 0.79, 0.01))

            db.add_all(mock_evals)
            db.commit()
            print("Info: Model performance data seeded successfully.")
        
    except Exception as e:
        if db:
            try:
                db.rollback()
            except:
                pass
        print(f"Warning: Failed to seed model performance data: {e}")
    finally:
        if db:
            try:
                db.close()
            except:
                pass


def save_prediction_to_db(entry: Dict[str, Any]):
    db = None
    try:
        db = SessionLocal()
        pred = Prediction(
            timestamp=entry.get('timestamp'),
            input_json=json.dumps(entry.get('input', {})),
            prediction_status=entry.get('prediction_status'),
            probability=entry.get('default_probability'),
            contributions_json=json.dumps(entry.get('feature_contributions', {}))
        )
        db.add(pred)
        db.commit()
        db.refresh(pred)
    except Exception as e:
        # Rollback in case of error
        if db:
            try:
                db.rollback()
            except:
                pass
        print('Warning: Failed to save prediction to DB:', e)
    finally:
        # Always close the database connection if it was opened
        if db:
            try:
                db.close()
            except:
                pass


def get_recent_predictions_from_db(limit: int = 20) -> List[Dict[str, Any]]:
    db = None
    try:
        db = SessionLocal()
        rows = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).all()
        result = []
        for r in rows:
            try:
                # Access the column values as attributes of the row object and convert to string
                input_json_str = str(r.input_json) if r.input_json is not None else None
                input_obj = json.loads(input_json_str) if input_json_str else {}
            except Exception:
                input_obj = {}
            try:
                # Access the column values as attributes of the row object and convert to string
                contrib_json_str = str(r.contributions_json) if r.contributions_json is not None else None
                contrib = json.loads(contrib_json_str) if contrib_json_str else {}
            except Exception:
                contrib = {}
            result.append({
                'timestamp': r.timestamp,
                'input': input_obj,
                'prediction_status': r.prediction_status,
                'default_probability': r.probability,
                'feature_contributions': contrib
            })
        return result
    except Exception as e:
        print('Warning: Failed to read predictions from DB:', e)
        return []
    finally:
        # Always close the database connection if it was opened
        if db:
            try:
                db.close()
            except:
                pass


# Define feature lists (must match the training script exactly)
NUMERICAL_COLS = [
    'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'term', 
    'income', 'Credit_Score', 'LTV', 'dtir1'
]
CATEGORICAL_COLS = [
    'loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 
    'Credit_Worthiness', 'open_credit', 'business_or_commercial', 'Neg_ammortization', 
    'interest_only', 'lump_sum_payment', 'construction_type', 'occupancy_type', 
    'Secured_by', 'total_units', 'credit_type', 'co_applicant_credit_type', 
    'age', 'submission_of_application', 'Region', 'Security_Type'
]
EXPECTED_FEATURES = NUMERICAL_COLS + CATEGORICAL_COLS # Order doesn't matter here, Pydantic handles order

def load_models():
    """Loads the joblib assets."""
    global model, scaler, label_encoders
    try:
        # Create full paths to model files
        model_path = os.path.join(MODELS_DIR, 'logistic_regression_model.joblib')
        scaler_path = os.path.join(MODELS_DIR, 'standard_scaler_lr.joblib')
        encoders_path = os.path.join(MODELS_DIR, 'label_encoders_lr.joblib')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(encoders_path)
    except FileNotFoundError:
        print("CRITICAL ERROR: One or more model files were not found.")
        # Raise an exception to prevent the app from starting without assets
        raise RuntimeError("Model assets missing. Ensure all .joblib files are in the models directory.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load assets: {e}")
        raise RuntimeError(f"Failed to load assets: {e}")

#Load models on startup
load_models()


#--- 3. Pydantic Input Schema (Defines the 29 features) ---
class LoanInput(BaseModel):
    # Categorical Features (Strings)
    loan_limit: str = Field(..., examples=["cf"], description="Loan limit type (\"cf\" or \"ncf\").")
    Gender: str = Field(..., examples=["Male"], description="Gender of the borrower.")
    approv_in_adv: str = Field(..., examples=["nopre"], description="Approval in advance status.")
    loan_type: str = Field(..., examples=["type2"], description="Loan type.")
    loan_purpose: str = Field(..., examples=["p1"], description="Loan purpose.")
    Credit_Worthiness: str = Field(..., examples=["l1"], description="Credit worthiness.")
    open_credit: str = Field(..., examples=["nopc"], description="Open credit status.")
    business_or_commercial: str = Field(..., examples=["nob/c"], description="Business or commercial loan flag.")
    Neg_ammortization: str = Field(..., examples=["not_neg"], description="Negative amortization status.")
    interest_only: str = Field(..., examples=["not_int"], description="Interest only payment status.")
    lump_sum_payment: str = Field(..., examples=["not_lpsm"], description="Lump sum payment status.")
    construction_type: str = Field(..., examples=["sb"], description="Construction type.")
    occupancy_type: str = Field(..., examples=["pr"], description="Occupancy type.")
    Secured_by: str = Field(..., examples=["home"], description="What secures the loan.")
    total_units: str = Field(..., examples=["1U"], description="Total units in property.")
    credit_type: str = Field(..., examples=["EXP"], description="Credit type.")
    co_applicant_credit_type: str = Field(..., examples=["CIB"], description="Co-applicant credit type.")
    age: str = Field(..., examples=["25-34"], description="Age bracket of the borrower.")
    submission_of_application: str = Field(..., examples=["to_inst"], description="Submission method.")
    Region: str = Field(..., examples=["south"], description="Region of property.")
    Security_Type: str = Field(..., examples=["direct"], description="Security Type.")
    
    #Numerical Features (Floats)
    rate_of_interest: float = Field(..., examples=[3.75], description="Interest rate.")
    Interest_rate_spread: float = Field(..., examples=[0.25], description="Spread over benchmark rate.")
    Upfront_charges: float = Field(..., examples=[1000.0], description="Upfront charges.")
    term: float = Field(..., examples=[360.0], description="Loan term in months (e.g., 360 for 30 years).")
    income: float = Field(..., examples=[5000.0], description="Monthly income.")
    Credit_Score: float = Field(..., examples=[750.0], description="Credit score (e.g., 300-900).")
    LTV: float = Field(..., examples=[80.0], description="Loan-to-Value ratio (%).")
    dtir1: float = Field(..., examples=[35.0], description="Debt-to-Income Ratio (DTI).")

    
    class Config:
        json_schema_extra = {
            "example": {
                "loan_limit": "cf", "Gender": "Male", "approv_in_adv": "nopre", 
                "loan_type": "type1", "loan_purpose": "p1", "Credit_Worthiness": "l1", 
                "open_credit": "nopc", "business_or_commercial": "nob/c", 
                "Neg_ammortization": "not_neg", "interest_only": "not_int", 
                "lump_sum_payment": "not_lpsm", "construction_type": "sb", 
                "occupancy_type": "pr", "Secured_by": "home", "total_units": "1U", 
                "credit_type": "EXP", "co_applicant_credit_type": "CIB", 
                "age": "25-34", "submission_of_application": "to_inst", 
                "Region": "south", "Security_Type": "direct", 
                "rate_of_interest": 4.5, "Interest_rate_spread": 0.5, 
                "Upfront_charges": 1500.0, "term": 360.0, "income": 7000.0, 
                "Credit_Score": 720.0, "LTV": 75.0, "dtir1": 38.0
            }
        }


# Placeholder PredictionResponse model (implied by /predict endpoint)
class PredictionResponse(BaseModel):
    prediction_status: int = Field(..., description="Predicted status (0=Non-Default, 1=Default).")
    default_probability: float = Field(..., description="Probability of default (class 1).")
    input_data: Dict[str, Any] = Field(..., description="The input data used for prediction.")
    feature_contributions: Dict[str, float] = Field(..., description="Feature importance/contributions.")
    timestamp: str = Field(..., description="Timestamp of the prediction.")

# --- Performance Metric Response Model ---
class PerformanceMetricResponse(BaseModel):
    timestamp: datetime = Field(..., description="Evaluation timestamp.")
    metric_value: float = Field(..., description="Metric value (e.g., 0.85).")
    test_size: int = Field(..., description="Size of the test set.")
    class Config:
        from_attributes = True

#--- 4. Preprocessing Function (Duplicate of the training logic) ---
def preprocess_data(data: LoanInput) -> pd.DataFrame:
    """Applies the same preprocessing steps as the training phase."""
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])

    #1. Imputation (using the saved mode/median)
    # The training code handled 'Sex Not Available' by mode imputation (which was 'Male')
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Sex Not Available', 'Male')
    
    # 2. Encoding Categorical Features
    # Check if label_encoders is not None before using it
    if label_encoders is not None:
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                # The saved encoders may have slightly different column names (e.g. hyphen vs underscore).
                # Try the exact col key first, then try replacing the first underscore with a hyphen.
                encoder_key = None
                if col in label_encoders:
                    encoder_key = col
                else:
                    alt = col.replace('_', '-', 1)
                    if alt in label_encoders:
                        encoder_key = alt

                if encoder_key is not None:
                    le = label_encoders[encoder_key]
                    # Use a try/except for robust transformation against unseen labels
                    try:
                        # Transform using the fitted encoder
                        df[col] = le.transform(df[col])
                    except ValueError:
                        # If an unseen label is found, assign a default value (e.g., the mode's index, usually 0)
                        print(f"Warning: Unseen label in {col}. Setting to 0.")
                        df[col] = 0

    #3. Scaling Numerical Features
    #Check if scaler is not None before using it
    if scaler is not None:
        df[NUMERICAL_COLS] = df[NUMERICAL_COLS].astype(float)
        df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])
    
    #Ensure correct column order is maintained (model expects the order it was trained on)
    #Return as DataFrame in the expected order. The training pipeline used some
    #feature names with hyphens (e.g. 'co-applicant_credit_type'). Our Pydantic
    #fields use underscores. If the loaded model exposes `feature_names_in_`,
    #try to align column names and ordering to what the model expects.


    processed = pd.DataFrame(df[EXPECTED_FEATURES])

    # Align to model feature names if available
    try:
        if model is not None and hasattr(model, 'feature_names_in_'):
            model_features = list(model.feature_names_in_)

            # Build rename map for columns that need underscore->hyphen conversion
            rename_map = {}
            for col in processed.columns:
                if col not in model_features:
                    
                    # Only replace the first underscore (e.g., 'co_applicant_credit_type' -> 'co-applicant_credit_type')

                    alt = col.replace('_', '-', 1)
                    if alt in model_features:
                        rename_map[col] = alt

            if rename_map:
                processed = processed.rename(columns=rename_map)

            # Reindex to model feature order (this will raise if some features are missing)
            processed = processed.reindex(columns=model_features)

    except Exception as e:
        # Surface a clear error if alignment fails
        raise RuntimeError(f"Failed to align features to model: {e}")

    # Ensure we return a DataFrame
    return pd.DataFrame(processed)

def _get_final_estimator(m: Any):
    """Return the final estimator object (sklearn estimator) even if model is a Pipeline."""
    if hasattr(m, 'steps') and isinstance(m.steps, list):
        # Assumes the final step is the estimator
        return m.steps[-1][1]
    return m

def compute_feature_contributions(processed: pd.DataFrame) -> Dict[str, float]:
    """
    Computes feature contributions for a linear model (coef * x) where possible.
    Returns a dict of feature -> contribution. If unavailable, returns empty dict.
    """
    try:
        est = _get_final_estimator(model)
        if hasattr(est, 'coef_'):
            coefs = np.array(est.coef_).ravel()
            
            # Ensure processed columns align to model.feature_names_in_ if available
            feat_names = list(processed.columns)
            vals = processed.iloc[0].astype(float).values
            
            # If lengths mismatch, try to align with estimator.feature_names_in_
            if hasattr(est, 'feature_names_in_') and len(feat_names) != len(est.feature_names_in_):
                feat_names = list(est.feature_names_in_)
                vals = processed[feat_names].iloc[0].astype(float).values

            contributions = {f: float(c) for f, c in zip(feat_names, (coefs * vals))}
            return contributions
    except Exception:
        pass
    return {}

#Build SHAP explainer at startup (safe to do now since helper functions are defined).
try:
    import shap
    est = _get_final_estimator(model)
    try:
        feat_names = list(model.feature_names_in_) if model is not None and hasattr(model, 'feature_names_in_') else EXPECTED_FEATURES
    except Exception:
        feat_names = EXPECTED_FEATURES
    try:
        # Create a background dataset of zeros (mean-centered data) for the explainer
        background = pd.DataFrame([dict((fn, 0) for fn in feat_names)])
        SHAP_EXPLAINER = shap.Explainer(est, background)
        print('Info: Prebuilt SHAP explainer at startup.')
    except Exception as e:
        print('Warning: Failed to prebuild SHAP explainer at startup:', e)
        SHAP_EXPLAINER = None
except Exception:
    SHAP_EXPLAINER = None

#Initialize the DB now that engine and models are available
init_db()


#--- 5. API Endpoints ---
#Mount static files from the `static/` directory so `bank_dashboard.html` and any assets are reachable
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/', summary='Root Redirect')
def root_redirect():
    """Redirects the root to the admin login page."""
    return RedirectResponse(url='/admin-login', status_code=302)

@app.get('/admin-login', summary='Admin Login Page')
def serve_admin_login_redirect():
    """Serves the login page, allowing it to be accessed without the /static prefix."""
    page_path = os.path.join(os.getcwd(), 'static', 'admin_login.html')
    if os.path.exists(page_path):
        return FileResponse(page_path, media_type='text/html')
    raise HTTPException(status_code=404, detail="Page not found")

@app.get('/bank_dashboard.html', summary='Dashboard Page')
def serve_dashboard():
    """Serve the main dashboard page."""
    page_path = os.path.join(os.getcwd(), 'static', 'bank_dashboard.html')
    if os.path.exists(page_path):
        return FileResponse(page_path, media_type='text/html')
    raise HTTPException(status_code=404, detail="Page not found")

@app.get('/risk_management.html', summary='Risk Management Page')
def serve_risk_management():
    """Serve the risk management page."""
    page_path = os.path.join(os.getcwd(), 'static', 'risk_management.html')
    if os.path.exists(page_path):
        return FileResponse(page_path, media_type='text/html')
    raise HTTPException(status_code=404, detail="Page not found")

@app.get('/model_performance.html', summary='Model Performance Page')
def serve_model_performance():
    """Serve the model performance page."""
    page_path = os.path.join(os.getcwd(), 'static', 'model_performance.html')
    if os.path.exists(page_path):
        return FileResponse(page_path, media_type='text/html')
    raise HTTPException(status_code=404, detail="Page not found")


@app.get('/admin.html', summary='Admin Page')
def serve_admin():
    """Serve the admin page."""
    page_path = os.path.join(os.getcwd(), 'static', 'admin.html')
    if os.path.exists(page_path):
        return FileResponse(page_path, media_type='text/html')
    raise HTTPException(status_code=404, detail="Page not found")


@app.get('/admin_login.html', summary='Admin Login Page')
def serve_admin_login():
    """Serve the admin login page."""
    page_path = os.path.join(os.getcwd(), 'static', 'admin_login.html')
    if os.path.exists(page_path):
        return FileResponse(page_path, media_type='text/html')
    raise HTTPException(status_code=404, detail="Page not found")





@app.get('/favicon.ico')
def favicon():
    """Serve `static/favicon.ico` if present, otherwise return 204 No Content to avoid 404 noise."""
    favicon_path = os.path.join(os.getcwd(), 'static', 'favicon.ico')
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type='image/x-icon')
    return Response(status_code=204) # No Content






@app.post('/predict', summary='Predict Loan Default', response_model=PredictionResponse)
def predict_default(data: LoanInput):
    """
    Predicts the loan status (0=Non-Default, 1=Default) and returns the probability
    and feature contributions.
    """
    #Ensure model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model assets are not loaded.")

    try:
        # Preprocess
        X_proc = preprocess_data(data)
        
        # Predict
        prediction = int(model.predict(X_proc)[0])
        # Probability of class 1 (Default)
        probability = float(model.predict_proba(X_proc)[0][1]) 
        contributions = compute_feature_contributions(X_proc)

        # Build response object
        result = {
            "prediction_status": prediction,
            "default_probability": probability,
            "input_data": data.model_dump(),
            "feature_contributions": contributions,
            "timestamp": datetime.now().isoformat()
        }

        # --- Automated Alert Generation ---
        alerts = []
        if probability >= RISK_THRESHOLD:
            # Generate a new Alert for high risk prediction
            alerts.append({
                # Note: entity_id is a placeholder since LoanInput doesn't have an actual ID
                "entity_id": uuid4().hex[:8], 
                "risk_signal": "High Default Probability",
                "severity": "High",
                "prediction_score": probability,
            })
        
        # --- History Management ---
        entry = {
            "timestamp": result.get('timestamp'),
            "input": result.get('input_data', {}),
            "prediction_status": result.get('prediction_status'),
            "default_probability": result.get('default_probability'),
            "feature_contributions": result.get('feature_contributions', {}),
            "alerts": alerts # Add alerts to history
        }

        # In-memory history (optional, preferred to use DB if available)
        PREDICTIONS_HISTORY.append(entry)
        if len(PREDICTIONS_HISTORY) > MAX_HISTORY:
            PREDICTIONS_HISTORY.pop(0)

        # Persist prediction to DB
        save_prediction_to_db(entry)

        # Save alerts to database if alerts:
        if alerts:
            db = None
            try:
                db = SessionLocal()
                for alert_data in alerts:
                    alert = Alert(
                        # Required fields for Alert model (assuming Alert has these fields)
                        entity_id=alert_data['entity_id'],
                        risk_signal=alert_data['risk_signal'],
                        severity=alert_data['severity'],
                        prediction_score=alert_data['prediction_score'],
                        alert_timestamp=datetime.utcnow(),
                        status="New" # Initial status
                    )
                    db.add(alert)
                db.commit()
            except Exception as e:
                # Log the error but don't break the prediction
                print(f"Warning: Failed to save alerts to database: {e}")
                # Try to rollback if db is defined
                if db:
                    try:
                        db.rollback()
                    except:
                        pass
            finally:
                # Always close the database connection if it was opened
                if db:
                    try:
                        db.close()
                    except:
                        pass

        return result # Original return

    except Exception as e:
        # Catch any preprocessing or prediction errors
        raise HTTPException(status_code=500, detail=f"Prediction failed due to internal error: {e}")



@app.get('/predictions', summary='Recent Predictions')
def get_predictions(limit: int = 20):
    """Return the most recent predictions stored in memory (up to `limit`)."""
    #Prefer DB-backed records for persistence
    rows = get_recent_predictions_from_db(limit)
    return {
        "count": len(rows),
        "predictions": rows
    }
    


# --- Model Performance Endpoint (FIXED) ---
@app.get('/api/model_performance', summary='Get Model Performance History')
def get_model_performance(request: Request):
    """
    Retrieves historical model performance metrics (AUC, F1-Score, Accuracy) 
    from the database for drift monitoring.
    """
    # Require admin auth via cookie
    token = request.cookies.get('admin_token')
    if not _is_admin_token_valid(token):
        raise HTTPException(status_code=401, detail='Authentication required')

    db = None
    try:
        db = SessionLocal()
        # Fetch all evaluation records, ordered by time
        evaluations = db.query(ModelEvaluation).order_by(ModelEvaluation.timestamp).all()
        
        # Group results by metric name
        metrics_data = {}
        for metric_name in ['AUC', 'F1-Score', 'Accuracy']:
            metrics_data[metric_name] = []
        
        for eval_record in evaluations:
            if eval_record.metric_name in metrics_data:
                # FIX: Use model_validate to map SQLAlchemy attributes to Pydantic fields correctly
                metrics_data[eval_record.metric_name].append(
                    PerformanceMetricResponse.model_validate(eval_record).model_dump()
                )
        
        return metrics_data

    except Exception as e:
        print(f"Error retrieving model performance data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model performance data: {e}")
    finally:
        if db:
            try:
                db.close()
            except:
                pass





@app.get('/stats', summary='System Statistics and Model Health')
def get_stats():
    """Return key operational statistics and model health metrics."""
    db = None
    try:
        db = SessionLocal()
        total_predictions = db.query(Prediction).count()
        default_count = db.query(Prediction).filter(Prediction.prediction_status == 1).count()
        
        # Check for alerts
        total_alerts = db.query(Alert).count()
        new_alerts = db.query(Alert).filter(Alert.status == "New").count()

        # Check for macro indicators
        latest_indicator = db.query(MacroIndicator).order_by(MacroIndicator.timestamp.desc()).first()
        
        return {
            "uptime_seconds": time.time() - STARTUP_TIME,
            "total_predictions": total_predictions,
            "default_count": default_count,
            "default_rate": default_count / total_predictions if total_predictions else 0,
            "total_alerts": total_alerts,
            "new_alerts": new_alerts,
            "latest_macro_indicator": {
                # Use one of the available indicator values since there's no single indicator_value column
                "timestamp": latest_indicator.timestamp.isoformat(),
                "value": latest_indicator.interest_rate if latest_indicator else None
            } if latest_indicator else None
        }
    except Exception as e:
        # Log error but return partial stats
        print(f"Warning: Failed to retrieve stats: {e}")
        return {
            "uptime_seconds": time.time() - STARTUP_TIME,
            "total_predictions": 0,
            "default_count": 0,
            "default_rate": 0,
            "total_alerts": 0,
            "new_alerts": 0,
            "latest_macro_indicator": None,
            "error": "Database access failed for full stats."
        }
    finally:
        # Always close the database connection if it was opened
        if db:
            try:
                db.close()
            except:
                pass




@app.post('/clear_history', summary='Clear Prediction History')
def clear_history(request: Request):
    """Clear all prediction history from the database."""
    # Require admin auth via cookie
    token = request.cookies.get('admin_token')
    if not _is_admin_token_valid(token):
        raise HTTPException(status_code=401, detail='Authentication required')
    
    db = None
    try:
        db = SessionLocal()
        num_deleted = db.query(Prediction).delete()
        # Optionally clear alerts as well
        db.query(Alert).delete()
        # NOTE: Leaving ModelEvaluation data as it's useful historical context
        # db.query(ModelEvaluation).delete() 
        db.commit()
        
        # Clear in-memory history
        PREDICTIONS_HISTORY.clear()
        
        return {"status": "success", "message": f"Cleared {num_deleted} predictions from history."}
    except Exception as e:
        # Rollback in case of error
        if db:
            try:
                db.rollback()
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {e}")
    finally:
        # Always close the database connection if it was opened
        if db:
            try:
                db.close()
            except:
                pass






@app.get('/export_history_csv', summary='Export prediction history as CSV')
def export_history_csv(request: Request, limit: Optional[int] = None):
    """Export history stored in the DB as CSV. If `limit` is provided, export up to that many most recent rows. Otherwise export all rows. """
    # Require admin auth via cookie
    token = request.cookies.get('admin_token')
    if not _is_admin_token_valid(token):
        raise HTTPException(status_code=401, detail='Authentication required')

    db = None
    try:
        db = SessionLocal()
        query = db.query(Prediction).order_by(Prediction.id.desc())
        if limit:
            rows = query.limit(limit).all()
        else:
            rows = query.all()

        sio = io.StringIO()
        writer = csv.writer(sio)

        #header
        writer.writerow(['id', 'timestamp', 'prediction_status', 'probability', 'input_json', 'contributions_json'])

        for r in rows:
            writer.writerow([r.id, r.timestamp, r.prediction_status, r.probability, r.input_json or '', r.contributions_json or ''])

        sio.seek(0)
        headers = {"Content-Disposition": "attachment; filename=predictions_history.csv"}
        return StreamingResponse(iter([sio.getvalue()]), media_type='text/csv', headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export history: {e}")
    finally:
        if db:
            try:
                db.close()
            except:
                pass





@app.post('/admin-login', summary='Submit admin login form')
async def admin_login(username: str = Form(...), password: str = Form(...)):
    """Handles admin login, sets a session cookie on success."""
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        token = _create_admin_session(username)
        # Redirect to the main admin page on success
        resp = RedirectResponse(url='/static/admin.html', status_code=302)
        # Set the session cookie
        resp.set_cookie(key='admin_token', value=token, httponly=True, samesite='lax', max_age=SESSION_TTL, path='/')
        return resp
    
    # On failure, redirect back to login with a simple message parameter
    resp = RedirectResponse(url='/admin-login?failed=1', status_code=302)
    return resp





@app.post('/admin-logout', summary='Logout admin')
def admin_logout(request: Request):
    token = request.cookies.get('admin_token')
    _invalidate_admin_token(token)
    resp = RedirectResponse(url='/admin-login', status_code=302)
    # Clear cookie
    resp.set_cookie('admin_token', '', expires=0)
    return resp





@app.post('/explain', summary='SHAP explain for a single input')
def explain_input(data: LoanInput):
    """Return SHAP values for the provided input. Uses a model-aware explainer if available. This endpoint requires the `shap` package. If shap is missing or fails, an error is returned. """
    #Ensure model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model assets are not loaded.")
    try:
        #Preprocess
        X_proc = preprocess_data(data)
        
        #Use cached explainer if available
        if SHAP_EXPLAINER is None:
             raise HTTPException(status_code=500, detail="SHAP explainer not available. Ensure 'shap' is installed and models are loaded.")

        # SHAP expects a numpy array, not a DataFrame
        shap_values = SHAP_EXPLAINER(X_proc)
        
        # Extract the values and feature names
        # For a binary classifier, we usually look at the SHAP values for the predicted class (index 1)
        if len(shap_values.values.shape) > 1 and shap_values.values.shape[1] == 2:
            # Multi-output (binary classification) - get values for class 1 (Default)
            values = shap_values.values[0, 1].tolist()
        else:
            # Single output (regression or a simpler binary case)
            values = shap_values.values[0].tolist()

        feature_names = list(X_proc.columns)

        return {
            "base_value": float(SHAP_EXPLAINER.expected_value), # Base value for log-odds
            "shap_values": dict(zip(feature_names, values)) # Map feature names to shap values
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate SHAP explanation: {e}")




@app.post('/export_report', summary='Export Prediction Report as PDF')
def export_prediction_report(data: LoanInput):
    """
    Generates a PDF report for a single prediction, including input, output,
    and feature contributions chart. Requires the 'reportlab' and 'matplotlib' packages.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model assets are not loaded.")

    try:
        # 1. Get prediction, probability, and contributions
        X_proc = preprocess_data(data)
        if model is None:
            raise HTTPException(status_code=503, detail="Model assets are not loaded.")

        prediction = int(model.predict(X_proc)[0])
        probability = float(model.predict_proba(X_proc)[0][1])
        contributions = compute_feature_contributions(X_proc)

        # 2. Build PDF in memory using reportlab and include a chart snapshot (matplotlib)
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.utils import ImageReader
            from reportlab.lib import colors
        except Exception:
            raise HTTPException(status_code=500, detail="reportlab is not installed in the environment.")

        # 3. Create a matplotlib chart for top contributions
        try:
            import matplotlib.pyplot as plt
            from PIL import Image

            items = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
            labels = [k.replace('_',' ').title() for k, _ in items] # Improved labels
            vals = [v for _, v in items]

            fig, ax = plt.subplots(figsize=(6, 3))

            # Use distinct colors for positive (risk-increasing) and negative (risk-reducing) contributions
            bar_colors = ['#ef4444' if v >= 0 else '#10b981' for v in vals]
            ax.barh(range(len(vals)), vals, color=bar_colors)
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_title('Top Feature Contributions (Model Log-Odds)')
            ax.axvline(0, color='gray', linestyle='--') # Add a zero line
            plt.tight_layout()

            imgbuf = io.BytesIO()
            fig.savefig(imgbuf, format='png', dpi=150)
            plt.close(fig) # Close the figure to free memory
            imgbuf.seek(0)
            chart_image = ImageReader(imgbuf)
        except Exception:
            chart_image = None
            print("Warning: Failed to generate matplotlib chart. Skipping chart in PDF.")

        # 4. Create the PDF
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(30, height - 40, "Loan Default Prediction Report")
        c.setFont("Helvetica", 10)
        c.drawString(30, height - 60, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Prediction Summary
        c.setFont("Helvetica-Bold", 14)
        c.drawString(30, height - 90, "Prediction Summary")
        c.setFont("Helvetica", 12)
        status_text = "Default (Risk High)" if prediction == 1 else "Non-Default (Risk Low)"
        status_color = colors.red if prediction == 1 else colors.green
        c.setFillColor(status_color)
        c.drawString(30, height - 110, f"Predicted Status: {status_text}")
        c.setFillColor(colors.black)
        c.drawString(30, height - 130, f"Default Probability: {probability:.4f} ({probability * 100:.2f}%)")

        # Feature Contributions Chart
        y2 = height - 170
        if chart_image:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(30, y2, "Feature Contribution Analysis")
            y2 -= 5
            # Draw the chart image
            img_width = 300 # Adjusted image width
            img_height = img_width * (3/6) # Maintain aspect ratio (3x6)
            c.drawImage(chart_image, 30, y2 - img_height, width=img_width, height=img_height)
            y2 -= (img_height + 20)
        
        # Input Data Section (1st column)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(30, y2, "Input Data (Numerical)")
        c.setFont("Helvetica", 10)
        y2 -= 20
        y_start = y2

        # Numerical Features
        i = 0
        for key, value in data.model_dump().items():
            if key in NUMERICAL_COLS:
                x_pos = 30
                # Check for column break (shouldn't happen with 8 numerical features)
                if i >= 10:
                    x_pos = 300
                    y2 = y_start
                
                c.drawString(x_pos, y2, f"{key.replace('_',' ').title()}: {value}")
                y2 -= 12
                i += 1
        
        # Categorical Features (Start a new column)
        c.setFont("Helvetica-Bold", 14)
        x_pos = 300
        y2 = y_start # Reset Y for the categorical column
        c.drawString(x_pos, y2 + 20, "Input Data (Categorical)")
        c.setFont("Helvetica", 10)
        y2 -= 20
        
        # Categorical Features
        for key, value in data.model_dump().items():
            if key in CATEGORICAL_COLS:
                c.drawString(x_pos, y2, f"{key.replace('_',' ').title()}: {value}")
                y2 -= 12
                # Check for page break if column is too long
                if y2 < 40:
                    c.showPage()
                    y2 = height - 40 # Reset Y for new page
                    c.setFont("Helvetica", 10) # Restore font after showPage

        # Footer
        if y2 < 40:
            c.showPage()
            y2 = height - 40
        
        c.save()
        buffer.seek(0)

        headers = {"Content-Disposition": "attachment; filename=prediction_report.pdf"}
        return StreamingResponse(buffer, media_type='application/pdf', headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build report: {e}")

# Add new Pydantic models for alerts
class AlertUpdate(BaseModel):
    status: str
    manager_notes: Optional[str] = None
    resolution_action: Optional[str] = None

class AlertResponse(BaseModel):
    id: int
    entity_id: str
    alert_timestamp: datetime
    risk_signal: str
    severity: str
    prediction_score: float
    status: str
    manager_notes: Optional[str] = None
    resolution_action: Optional[str] = None
    class Config:
        from_attributes = True # Allow Pydantic to map SQLAlchemy models

# 
# Add new API endpoints for alerts
@app.get("/api/alerts", response_model=List[AlertResponse], summary="Get all active alerts")
def get_alerts(request: Request):
    """Retrieve all non-Closed alerts from the database."""
    # Check for admin session (since this is sensitive management data)
    token = request.cookies.get('admin_token')
    if not _is_admin_token_valid(token):
        # The middleware should catch this for /static/risk_management.html, but API call still needs protection
        raise HTTPException(status_code=401, detail='Authentication required')

    db = None
    try:
        db = SessionLocal()
        # Retrieve alerts that are not 'Closed', ordered by most recent first
        alerts = db.query(Alert).filter(Alert.status != "Closed").order_by(Alert.alert_timestamp.desc()).all()
        return alerts
    except Exception as e:
        print(f"Error retrieving alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve alerts: {e}")
    finally:
        # Always close the database connection if it was opened
        if db:
            try:
                db.close()
            except:
                pass




@app.post("/api/alerts/{alert_id}/update_status", response_model=AlertResponse, summary="Update alert status and notes")
def update_alert_status(alert_id: int, update: AlertUpdate, request: Request):
    """Update the status, manager_notes, and resolution_action for a specific alert."""
    
    # Require admin auth via cookie
    token = request.cookies.get('admin_token')
    if not _is_admin_token_valid(token):
        raise HTTPException(status_code=401, detail='Authentication required')
        
    db = None
    try:
        db = SessionLocal()
        # Retrieve the alert by ID
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
            
        # Update fields
        alert.status = update.status
        if update.manager_notes is not None:
            alert.manager_notes = update.manager_notes
        if update.resolution_action is not None:
            alert.resolution_action = update.resolution_action
            
        db.commit()
        db.refresh(alert)
        
        # Return the updated alert using the Pydantic response model
        return AlertResponse.model_validate(alert)
        
    except HTTPException:
        # Re-raise explicit HTTP exceptions (like 401, 404)
        raise
    except Exception as e:
        if db:
            try:
                db.rollback()
            except:
                pass
        print(f"Error updating alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update alert: {e}")
    finally:
        if db:
            try:
                db.close()
            except:
                pass





@app.get("/api/macro_indicators", summary="Get macroeconomic indicators")
def get_macro_indicators():
    """Retrieve recent macroeconomic indicator data."""
    db = None
    try:
        db = SessionLocal()
        indicators = db.query(MacroIndicator).order_by(MacroIndicator.timestamp.desc()).limit(100).all()
        # NOTE: Assumes MacroIndicator model is properly defined to be serializable
        # If it's not a Pydantic model, it will be serialized as a list of SQLAlchemy objects
        return indicators
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve macroeconomic indicators: {e}")
    finally:
        # Always close the database connection if it was opened
        if db:
            try:
                db.close()
            except:
                pass

# Session Middleware to protect routes
class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Paths that do NOT require authentication
        public_paths = ['/', '/admin-login', '/admin-login.html', '/favicon.ico', '/predict']
        
        # Allow access to static files except protected admin pages
        if request.url.path.startswith("/static/"):
            # Block access to admin pages in static directory
            if (request.url.path.startswith("/static/admin.html") or 
                request.url.path.startswith("/static/risk_management.html") or 
                request.url.path.startswith("/static/model_performance.html") or 
                request.url.path.startswith("/static/admin_login.html")):
                
                # Check for admin session
                token = request.cookies.get('admin_token')
                if not _is_admin_token_valid(token):
                    # Redirect to login page
                    return RedirectResponse(url='/admin-login', status_code=302)
            # Allow all other static files
            response = await call_next(request)
            return response
            
        # Check if the path requires authentication (i.e., not a public path and not covered by static checks)
        if (request.url.path not in public_paths and 
            not request.url.path.startswith('/api/macro_indicators')): # /api/macro_indicators is public
            
            # The following API endpoints are protected internally:
            # /api/alerts
            # /api/model_performance
            # /admin-logout
            # /clear_history
            # /export_history_csv
            # /explain
            # /export_report
            # /predictions
            # /stats
            
            # Check for admin session
            token = request.cookies.get('admin_token')
            if not _is_admin_token_valid(token):
                #Redirect to login page
                return RedirectResponse(url='/admin-login', status_code=302)
        
        response = await call_next(request)
        return response


#Add middleware to app
app.add_middleware(SessionMiddleware)

if __name__ == "__main__":
    #To run the API locally: uvicorn main:app --reload
    #This block is typically commented out when deploying to a container
    uvicorn.run(app, host="0.0.0.0", port=8000)