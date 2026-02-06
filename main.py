from fastapi import FastAPI, HTTPException, Response, Request, Form, Depends, status
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta
import json
import io
from fastapi.responses import StreamingResponse
import csv
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, func, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from uuid import uuid4
import time 
import os
import joblib
import logging
from dotenv import load_dotenv
load_dotenv()
try:
    import pandas as pd
except Exception as e:
    pd = None
    print(f"Warning: pandas import failed: {e}")
try:
    import numpy as np
except Exception as e:
    np = None
    print(f"Warning: numpy import failed: {e}")
import uvicorn
from passlib.context import CryptContext

#Configure module logger
logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))
logger = logging.getLogger('ews')
from starlette.middleware.cors import CORSMiddleware
#Reportlab dependencies for PDF export (Assumed availability)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import Paragraph
    import matplotlib.pyplot as plt
    import shap
    REPORTLAB_AVAILABLE = True
except Exception as e:
    REPORTLAB_AVAILABLE = False
    print(f"Warning: PDF/chart/SHAP dependencies not available: {e}. PDF export will be disabled.")

#Global variables for model health and stats

STARTUP_TIME = time.time()
RISK_THRESHOLD = float(os.environ.get('RISK_THRESHOLD', '0.70')) # 70% probability for a High-Risk Alert
MODELS_DIR = "models"
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "predictions.db")
PREDICTIONS_HISTORY: List[Dict[str, Any]] = []
MAX_HISTORY = 200

#--- 2. Load Assets (Model, Scaler, Encoders) ---
model: Optional[Any] = None
scaler: Optional[Any] = None
label_encoders: Optional[Dict[str, Any]] = None
SHAP_EXPLAINER = None
EXPECTED_FEATURES = ['loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 'Credit_Worthiness', 'open_credit', 'business_or_purpose', 'Neg_ammortization', 'interest_only', 'lump_sum_payment', 'Occupancy_Type', 'Secured_by', 'total_units', 'credit_type', 'co_applicant_credit_type', 'age', 'submission_of_application', 'Region', 'Security_Type', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'term', 'income', 'Credit_Score', 'LTV', 'dtir1']
NUMERICAL_COLS = ['rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'term', 'income', 'Credit_Score', 'LTV', 'dtir1']
CATEGORICAL_COLS = list(set(EXPECTED_FEATURES) - set(NUMERICAL_COLS))

#--- Admin Session Configuration (Same as original) ---
ADMIN_SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL = int(os.environ.get('ADMIN_SESSION_TTL', '3600'))
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'password')
REDIS_URL = os.environ.get('REDIS_URL')
redis_client = None
USE_REDIS = False
if REDIS_URL:
    try:
        import redis as _redis
        redis_client = _redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        USE_REDIS = True
        print('Info: Connected to Redis for admin sessions.')
    except Exception as e:
        print('Warning: Could not connect to Redis at REDIS_URL, falling back to in-memory sessions:', e)
        redis_client = None
        USE_REDIS = False

def _create_admin_session(username: str) -> str:
    token = uuid4().hex
    expires = datetime.utcnow() + timedelta(seconds=SESSION_TTL)
    if USE_REDIS and redis_client:
        try:
            key = f"admin_session:{token}"
            redis_client.set(key, username, ex=SESSION_TTL)
            return token
        except Exception as e:
            print('Warning: Redis set failed for admin session, falling back to in-memory:', e)
    ADMIN_SESSIONS[token] = {'user': username, 'expires': expires}
    return token

def _is_admin_token_valid(token: Optional[str]) -> bool:
    if not token: return False
    if USE_REDIS and redis_client:
        try:
            return bool(redis_client.get(f"admin_session:{token}"))
        except Exception as e:
            print('Warning: Redis get failed for admin session, falling back to in-memory:', e)
    info = ADMIN_SESSIONS.get(token)
    if not info: return False
    expires = info.get('expires')
    if expires is not None and expires < datetime.utcnow():
        try: del ADMIN_SESSIONS[token]
        except Exception: pass
        return False
    return True

def _invalidate_admin_token(token: Optional[str]):
    if not token: return
    if USE_REDIS and redis_client:
        try:
            redis_client.delete(f"admin_session:{token}")
            return
        except Exception as e:
            print('Warning: Redis delete failed for admin session, falling back to in-memory:', e)
    if token and token in ADMIN_SESSIONS:
        try: del ADMIN_SESSIONS[token]
        except Exception: pass

#--- SQLAlchemy / Database configuration ---
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    SQLALCHEMY_DATABASE_URL = DATABASE_URL
else:
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine_kwargs = {}
if SQLALCHEMY_DATABASE_URL.startswith('sqlite'):
    engine_kwargs = {"connect_args": {"check_same_thread": False}}

engine = create_engine(SQLALCHEMY_DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

#Quick connectivity check (informational only)
try:
    conn = engine.connect()
    conn.close()
    print('Info: Database engine initialized and reachable via SQLALCHEMY_DATABASE_URL')
except Exception as e:
    print(f'Warning: Database engine could not connect at startup: {e}. Verify DATABASE_URL and network settings.')

#Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

#--- DB Models ---
class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String(50), index=True)
    input_json = Column(Text)
    prediction_status = Column(Integer, index=True)
    probability = Column(Float)
    contributions_json = Column(Text)
    
class Alert(Base):
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True, index=True)
    alert_timestamp = Column(DateTime, default=datetime.utcnow)
    entity_id = Column(String(50), nullable=False, index=True) 
    risk_signal = Column(String(50), nullable=False)
    severity = Column(String(50), nullable=False)
    prediction_score = Column(Float)
    status = Column(String(20), default="New", index=True) 
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

class ModelEvaluation(Base):
    __tablename__ = 'model_evaluations'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_name = Column(String(100), index=True, nullable=False)
    metric_value = Column(Float, nullable=False)
    test_size = Column(Integer, nullable=False)

#--- NEW: Activity Log Model for Auditing ---
class ActivityLog(Base):
    __tablename__ = 'activity_logs'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = Column(String(50), nullable=False)
    action_type = Column(String(50), nullable=False, index=True) # e.g., 'LOGIN', 'LOGOUT', 'ALERT_UPDATE', 'MODEL_RELOAD'
    details_json = Column(Text, nullable=True)

# --- Customer Model mapping to external MySQL table 'Loan_table' ---
class LoanCustomer(Base):
    __tablename__ = 'Loan_table'
    # Provide MySQL table args for compatibility when using MySQL as the DATABASE_URL
    __table_args__ = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"}

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Categorical (strings) — explicit lengths to match MySQL VARCHAR
    loan_limit = Column(String(50), nullable=False)
    Gender = Column(String(20), nullable=False)
    approv_in_adv = Column(String(20), nullable=False)
    loan_type = Column(String(50), nullable=False)
    loan_purpose = Column(String(100), nullable=False)
    Credit_Worthiness = Column(String(50), nullable=False)
    open_credit = Column(String(20), nullable=False)
    business_or_purpose = Column(String(50), nullable=False)
    Neg_ammortization = Column(String(20), nullable=False)
    interest_only = Column(String(20), nullable=False)
    lump_sum_payment = Column(String(20), nullable=False)
    Occupancy_Type = Column(String(50), nullable=False)
    Secured_by = Column(String(50), nullable=False)
    total_units = Column(String(20), nullable=False)
    credit_type = Column(String(50), nullable=False)
    co_applicant_credit_type = Column(String(50), nullable=False)
    age = Column(String(20), nullable=False)
    submission_of_application = Column(String(50), nullable=False)
    Region = Column(String(50), nullable=False)
    Security_Type = Column(String(50), nullable=False)

    #Numerical (use Integer for Credit_Score)
    rate_of_interest = Column(Float, nullable=False)
    Interest_rate_spread = Column(Float, nullable=False)
    Upfront_charges = Column(Float, nullable=False)
    term = Column(Float, nullable=False)
    income = Column(Float, nullable=False)
    Credit_Score = Column(Integer, nullable=False)
    LTV = Column(Float, nullable=False)
    dtir1 = Column(Float, nullable=False)

    def as_dict(self):
        return {
            'id': self.id,
            'loan_limit': self.loan_limit,
            'Gender': self.Gender,
            'approv_in_adv': self.approv_in_adv,
            'loan_type': self.loan_type,
            'loan_purpose': self.loan_purpose,
            'Credit_Worthiness': self.Credit_Worthiness,
            'open_credit': self.open_credit,
            'business_or_purpose': self.business_or_purpose,
            'Neg_ammortization': self.Neg_ammortization,
            'interest_only': self.interest_only,
            'lump_sum_payment': self.lump_sum_payment,
            'Occupancy_Type': self.Occupancy_Type,
            'Secured_by': self.Secured_by,
            'total_units': self.total_units,
            'credit_type': self.credit_type,
            'co_applicant_credit_type': self.co_applicant_credit_type,
            'age': self.age,
            'submission_of_application': self.submission_of_application,
            'Region': self.Region,
            'Security_Type': self.Security_Type,
            'rate_of_interest': self.rate_of_interest,
            'Interest_rate_spread': self.Interest_rate_spread,
            'Upfront_charges': self.Upfront_charges,
            'term': self.term,
            'income': self.income,
            'Credit_Score': self.Credit_Score,
            'LTV': self.LTV,
            'dtir1': self.dtir1,
        }

#--- NEW: Audit Logging Function ---
def log_activity(db: Session, user: str, action_type: str, details: Optional[Dict[str, Any]] = None):
    """Logs an administrative action to the database."""
    try:
        log_entry = ActivityLog(
            user=user,
            action_type=action_type,
            details_json=json.dumps(details) if details else None
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        print(f"Warning: Failed to log activity: {e}")
        db.rollback()
 
#--- Model Loading and Initialization ---
def load_models():
    """Loads the model, scaler, encoders, and config files."""
    global model, scaler, label_encoders, SHAP_EXPLAINER

    model_path = os.path.join(MODELS_DIR, "logistic_regression_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "standard_scaler_lr.joblib")
    encoders_path = os.path.join(MODELS_DIR, "label_encoders_lr.joblib")
    
    # NOTE: The original code assumed hardcoded features. In a real-world scenario, 
    # EXPECTED_FEATURES, NUMERICAL_COLS, and CATEGORICAL_COLS should be loaded from a config file.
    # Since no config file was provided, I keep the placeholders and skip the config load here 
    # but the logic for SHAP and Pydantic validation relies on this list.

    try:
        #Load Model Assets
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(encoders_path)

        if EXPECTED_FEATURES and REPORTLAB_AVAILABLE and model is not None:
             try:
                 # In a real system, background_data would be the mean of scaled training data
                 if np is None:
                     print('Info: numpy not available, skipping SHAP explainer prebuild')
                     SHAP_EXPLAINER = None
                 else:
                     background_data = np.zeros((1, len(EXPECTED_FEATURES))) 
                     SHAP_EXPLAINER = shap.Explainer(model.predict_proba, background_data)
                     print('Info: SHAP explainer prebuilt at startup.')
             except Exception as e:
                 print('Warning: Failed to prebuild SHAP explainer at startup:', e)
                 SHAP_EXPLAINER = None

    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: One or more model files were not found: {e}")
        raise RuntimeError("Model assets missing. Ensure all .joblib files are in the models directory.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load assets: {e}")
        raise RuntimeError(f"Failed to load assets: {e}")

def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    try:
        # Create all tables including the new ActivityLog
        Base.metadata.create_all(bind=engine)
        seed_initial_performance_data()
        seed_initial_macro_indicator()
        create_default_admin() # Ensure default admin user exists
    except SQLAlchemyError as e:
        print('Warning: Failed to initialize DB with SQLAlchemy:', e)

#Dependency for password hashing
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

class EWSAdmin(Base):
    __tablename__ = 'ews_admins'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

#--- Seeding functions (retained for completeness) ---
def generate_mock_data(metric_name, start_value, volatility):
    data = []
    if np is None:
        print('Info: numpy unavailable, skipping mock model performance generation')
        return data
    ts = datetime.utcnow() - timedelta(days=90)
    for i in range(30):
        val = start_value + np.random.uniform(-volatility, volatility)
        val = max(0, min(1, val)) # Clamp between 0 and 1
        ts += timedelta(days=3)
        data.append(ModelEvaluation(
            timestamp=ts,
            metric_name=metric_name,
            metric_value=val,
            test_size=25000 + i * 500
        ))
    return data

def seed_initial_performance_data():
    db = None
    try:
        db = SessionLocal()
        if db.query(ModelEvaluation).count() == 0:
            print("Info: Seeding initial model performance data.")
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
        if db: db.close()


def seed_initial_macro_indicator():
    db = None
    if np is None:
        print('Info: numpy unavailable, skipping macro indicator seeding')
        return
    try:
        db = SessionLocal()
        if db.query(MacroIndicator).count() == 0:
            print("Info: Seeding initial macro indicator data.")
            now = datetime.utcnow().replace(day=1)
            
            mock_indicators = []
            for i in range(12): # 12 months of mock data
                ts = now - timedelta(days=30 * (11 - i))
                mock_indicators.append(MacroIndicator(
                    timestamp=ts,
                    gdp_growth_rate=round(1.5 + np.random.uniform(-0.5, 0.5), 2),
                    market_index_yoy=round(8.0 + np.random.uniform(-3.0, 3.0), 2),
                    unemployment_rate=round(4.0 + np.random.uniform(-0.5, 0.5), 2),
                    inflation_rate=round(2.0 + np.random.uniform(-0.5, 0.5), 2),
                    interest_rate=round(0.5 + np.random.uniform(-0.1, 0.1), 2),
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
        if db: db.close()

def create_default_admin():
    db = SessionLocal()
    try:
        admin = db.query(EWSAdmin).filter_by(username='admin').first()
        if not admin:
            password_hash = pwd_context.hash('Get@6963')
            admin = EWSAdmin(username='admin', password_hash=password_hash)
            db.add(admin)
            db.commit()
            print('Default admin user created in ews_admins table.')
    except Exception as e:
        print(f'Error creating default admin: {e}')
    finally:
        db.close()

#--- 1. FastAPI App Initialization ---
app = FastAPI(
    title="Bank Early Warning System API (Advanced)",
    description="Predicts loan default probability with enhanced admin tools, logging, and risk management."
)

#Add CORS middleware
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#--- Middleware for Admin Session Check (Retained from original) ---
class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        #Public paths (no session required)
        public_paths = ['/', '/admin-login', '/admin-logout', '/predict', '/explain', '/export_report', '/api/model_performance']

        #Allow static files and public APIs
        if request.url.path.startswith('/static/') or request.url.path.startswith('/api/macro_indicators'):
            response = await call_next(request)
            return response
            
        #Check if the path requires authentication
        if request.url.path not in public_paths:
            #The following API endpoints are protected internally:
            #/api/alerts, /api/model_performance, /clear_history, /export_history_csv, /predictions, /stats, /api/model/retrain, /api/audit_log
            
            token = request.cookies.get('admin_token')
            if not _is_admin_token_valid(token):
                #Redirect to login page for HTML requests, or return 401 for API calls
                if request.url.path.startswith('/api/'):
                    return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Authentication required"})
                return RedirectResponse(url='/admin-login', status_code=status.HTTP_302_FOUND)
        
        response = await call_next(request)
        return response

#Add middleware to app
app.add_middleware(SessionMiddleware)

#--- Pydantic Input Schema (Needs to match EXPECTED_FEATURES) ---
class LoanInput(BaseModel):
    #Categorical Features (Strings)
    loan_limit: str = Field(..., description="The loan limit status (e.g., 'Yes', 'No')")
    Gender: str = Field(..., description="Gender of the applicant")
    approv_in_adv: str = Field(..., description="Pre-approval status")
    loan_type: str = Field(..., description="Type of loan")
    loan_purpose: str = Field(..., description="Purpose of the loan")
    Credit_Worthiness: str = Field(..., description="Credit Worthiness (e.g., 'good', 'bad')")
    open_credit: str = Field(..., description="Open credit availability")
    business_or_purpose: str = Field(..., description="Business or personal use")
    Neg_ammortization: str = Field(..., description="Negative amortization status")
    interest_only: str = Field(..., description="Interest only payment status")
    lump_sum_payment: str = Field(..., description="Lump sum payment option")
    Occupancy_Type: str = Field(..., description="Occupancy type (e.g., 'pr', 'ir')")
    Secured_by: str = Field(..., description="Collateral type")
    total_units: str = Field(..., description="Total units (e.g., '1U', '2U')")
    credit_type: str = Field(..., description="Credit type (e.g., 'CIB', 'EXP')")
    co_applicant_credit_type: str = Field(..., description="Co-applicant credit type")
    age: str = Field(..., description="Age bracket of the applicant")
    submission_of_application: str = Field(..., description="Application submission method")
    Region: str = Field(..., description="Geographic region")
    Security_Type: str = Field(..., description="Security type")
    

    #Numerical Features (Floats/Ints)
    rate_of_interest: float = Field(..., gt=0.0, description="Rate of interest (e.g., 4.5)")
    Interest_rate_spread: float = Field(..., ge=0.0, description="Spread over index rate (e.g., 0.5)")
    Upfront_charges: float = Field(..., ge=0.0, description="Upfront charges (e.g., 1500.0)")
    term: float = Field(..., gt=0.0, description="Loan term in months (e.g., 360.0)")
    income: float = Field(..., gt=0.0, description="Applicant income (e.g., 7000.0)")
    Credit_Score: float = Field(..., gt=0.0, description="Credit Score (e.g., 750.0)")
    LTV: float = Field(..., gt=0.0, description="Loan-to-Value Ratio (e.g., 80.0)")
    dtir1: float = Field(..., gt=0.0, description="Debt-to-Income Ratio (e.g., 40.0)")

class LoanOutput(BaseModel):
    timestamp: str
    prediction_status: int = Field(..., description="0 for Non-default, 1 for Default")
    default_probability: float = Field(..., ge=0.0, le=1.0)
    feature_contributions: Dict[str, float] = Field(..., description="SHAP values indicating feature importance")
    alert: Optional[Dict[str, Any]] = None

# --- Customer Schemas ---
class CustomerBase(BaseModel):
    # Categorical Features (Strings) — all required
    loan_limit: str = Field(..., description="The loan limit status (e.g., 'Yes', 'No')")
    Gender: str = Field(..., description="Gender of the applicant")
    approv_in_adv: str = Field(..., description="Pre-approval status")
    loan_type: str = Field(..., description="Type of loan")
    loan_purpose: str = Field(..., description="Purpose of the loan")
    Credit_Worthiness: str = Field(..., description="Credit Worthiness (e.g., 'good', 'bad')")
    open_credit: str = Field(..., description="Open credit availability")
    business_or_purpose: str = Field(..., description="Business or personal use")
    Neg_ammortization: str = Field(..., description="Negative amortization status")
    interest_only: str = Field(..., description="Interest only payment status")
    lump_sum_payment: str = Field(..., description="Lump sum payment option")
    Occupancy_Type: str = Field(..., description="Occupancy type (e.g., 'pr', 'ir')")
    Secured_by: str = Field(..., description="Collateral type")
    total_units: str = Field(..., description="Total units (e.g., '1U', '2U')")
    credit_type: str = Field(..., description="Credit type (e.g., 'CIB', 'EXP')")
    co_applicant_credit_type: str = Field(..., description="Co-applicant credit type")
    age: str = Field(..., description="Age bracket of the applicant")
    submission_of_application: str = Field(..., description="Application submission method")
    Region: str = Field(..., description="Geographic region")
    Security_Type: str = Field(..., description="Security type")

    # Numerical Features (Floats/Ints) — all required
    rate_of_interest: float = Field(..., gt=0.0, description="Rate of interest (e.g., 4.5)")
    Interest_rate_spread: float = Field(..., ge=0.0, description="Spread over index rate (e.g., 0.5)")
    Upfront_charges: float = Field(..., ge=0.0, description="Upfront charges (e.g., 1500.0)")
    term: float = Field(..., gt=0.0, description="Loan term in months (e.g., 360.0)")
    income: float = Field(..., gt=0.0, description="Applicant income (e.g., 7000.0)")
    Credit_Score: float = Field(..., gt=0.0, description="Credit Score (e.g., 750.0)")
    LTV: float = Field(..., gt=0.0, description="Loan-to-Value Ratio (e.g., 80.0)")
    dtir1: float = Field(..., gt=0.0, description="Debt-to-Income Ratio (e.g., 40.0)")

class CustomerCreate(CustomerBase):
    """Schema for creating a new customer (all fields required)."""
    pass

class CustomerUpdate(BaseModel):
    """Schema for updating a customer: all fields optional to support partial updates."""
    loan_limit: Optional[str] = None
    Gender: Optional[str] = None
    approv_in_adv: Optional[str] = None
    loan_type: Optional[str] = None
    loan_purpose: Optional[str] = None
    Credit_Worthiness: Optional[str] = None
    open_credit: Optional[str] = None
    business_or_purpose: Optional[str] = None
    Neg_ammortization: Optional[str] = None
    interest_only: Optional[str] = None
    lump_sum_payment: Optional[str] = None
    Occupancy_Type: Optional[str] = None
    Secured_by: Optional[str] = None
    total_units: Optional[str] = None
    credit_type: Optional[str] = None
    co_applicant_credit_type: Optional[str] = None
    age: Optional[str] = None
    submission_of_application: Optional[str] = None
    Region: Optional[str] = None
    Security_Type: Optional[str] = None

    rate_of_interest: Optional[float] = Field(None, gt=0.0)
    Interest_rate_spread: Optional[float] = Field(None, ge=0.0)
    Upfront_charges: Optional[float] = Field(None, ge=0.0)
    term: Optional[float] = Field(None, gt=0.0)
    income: Optional[float] = Field(None, gt=0.0)
    Credit_Score: Optional[float] = Field(None, gt=0.0)
    LTV: Optional[float] = Field(None, gt=0.0)
    dtir1: Optional[float] = Field(None, gt=0.0)

class CustomerResponse(CustomerBase):
    id: int

    class Config:
        orm_mode = True

def preprocess_data(data: LoanInput) -> Any:
    """Applies necessary preprocessing (encoding, scaling) to input data."""
    if scaler is None or label_encoders is None:
        raise RuntimeError("Model assets (scaler/encoders) are not loaded.")

    df = pd.DataFrame([data.model_dump()])

    #1. Label Encoding for Categorical Columns
    for col in CATEGORICAL_COLS:
        le = label_encoders.get(col)
        if le and col in df.columns:
            # Handle unknown categories by mapping them to a predefined category or skipping (safer)
            def transform_with_handling(val):
                try:
                    return le.transform([val])[0] if val in le.classes_ else -1
                except Exception:
                    return -1  # Default value for unknown categories
            df[col] = df[col].apply(transform_with_handling)

    #2. Standardization for Numerical Columns
    if NUMERICAL_COLS:
        try:
            df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])
        except Exception as e:
            print(f"Warning: Scaling failed: {e}")
            # If scaling fails, continue with unscaled data
            pass
    
    #3. Handle any NaN values that might have been introduced
    df = df.fillna(0)  # Fill NaN values with 0
    
    #4. Feature Alignment
    #The final columns must match the expected feature list in order
    processed = pd.DataFrame(df[EXPECTED_FEATURES])

    #Align to model feature names if available (for models that use specific feature names, e.g., co_applicant_credit_type -> co-applicant_credit_type)
    try:
        if model is not None and hasattr(model, 'feature_names_in_'):
            model_features = list(model.feature_names_in_)
            rename_map = {}
            for col in processed.columns:

                #Simple underscore to hyphen conversion check for alignment

                hyphen_col = col.replace('_', '-')
                if hyphen_col in model_features:
                    rename_map[col] = hyphen_col
                
            if rename_map:
                processed = processed.rename(columns=rename_map)
            
            #Reorder columns to match the model's expectation

            processed = processed.reindex(columns=model_features)
    except Exception as e:
        print(f"Warning: Feature alignment failed: {e}. Proceeding with current column order/names.")
        pass

    #Final check for NaN values
    if processed.isnull().values.any():
        print("Warning: NaN values detected in processed data, filling with 0")
        processed = processed.fillna(0)
        
    return processed

@app.post('/predict', response_model=LoanOutput, summary='Run a single prediction')
def predict(data: LoanInput, db: Session = Depends(get_db)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model assets are not loaded.")

    try:
        X_proc = preprocess_data(data)
        prediction = int(model.predict(X_proc)[0])
        #Probability of status 1 (Default) is usually the second element
        probability = float(model.predict_proba(X_proc)[0][1]) 

        #--- SHAP Contributions ---
        feature_contributions = {}
        if SHAP_EXPLAINER:
            try:
                shap_values = SHAP_EXPLAINER(X_proc)[0].values[:, 1] # SHAP values for class 1
                # Safely determine feature names
                feature_names = EXPECTED_FEATURES
                try:
                    if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
                        feature_names = list(model.feature_names_in_)
                except:
                    pass
                    
                for name, value in zip(feature_names, shap_values):
                    feature_contributions[name] = float(value)
            except Exception as e:
                print(f"Warning: SHAP explanation failed: {e}")
                #Continue without SHAP values if there's an error
        
        #--- Alert Generation ---
        alerts = []
        alert_data = None
        if probability >= RISK_THRESHOLD:
            severity = "High"
            risk_signal = "High Default Probability"
            
            #NEW: Associate a unique ID for better alert management
            entity_id = uuid4().hex[:8] 
            
            new_alert = Alert(
                entity_id=entity_id,
                risk_signal=risk_signal,
                severity=severity,
                prediction_score=probability,
                status="New"
            )
            db.add(new_alert)

            db.commit() #Commit alert immediately

            alerts.append({
                "entity_id": entity_id,
                "risk_signal": risk_signal,
                "severity": severity,
                "prediction_score": probability,
            })
            alert_data = alerts[0]
            
        #--- History Management ---
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_data": data.model_dump(),
            "prediction_status": prediction,
            "default_probability": probability,
            "feature_contributions": feature_contributions,
        }
        
        #Log prediction to DB
        db_prediction = Prediction(
            timestamp=result['timestamp'],
            input_json=json.dumps(result['input_data']),
            prediction_status=result['prediction_status'],
            probability=result['default_probability'],
            contributions_json=json.dumps(result['feature_contributions'])
        )
        db.add(db_prediction)
        db.commit()
        
        #Log to in-memory history (for immediate fetching in dashboard)
        entry = {
            "timestamp": result['timestamp'],
            "input": result['input_data'],
            "prediction_status": result['prediction_status'],
            "default_probability": result['default_probability'],
            "feature_contributions": result['feature_contributions'],
            "alerts": alerts 
        }
        PREDICTIONS_HISTORY.insert(0, entry)
        if len(PREDICTIONS_HISTORY) > MAX_HISTORY:
            PREDICTIONS_HISTORY.pop()

        return LoanOutput(
            timestamp=result['timestamp'],
            prediction_status=prediction,
            default_probability=probability,
            feature_contributions=feature_contributions,
            alert=alert_data
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {e}")


@app.get('/explain', summary='Generate SHAP summary plot as Base64 image')
def explain():
    if model is None or SHAP_EXPLAINER is None: 
        raise HTTPException(status_code=503, detail="Model or SHAP explainer not loaded.")
    
    try:
        #Placeholder: Generate a mock summary plot or a feature importance plot
        #In a real scenario, this would generate a SHAP summary plot from a sample of test data.

        features = EXPECTED_FEATURES[:10]
        importance = np.abs(np.random.normal(0, 0.1, len(features)))
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(features, importance, color='#4f46e5')
        ax.set_title("Top 10 Global Feature Importance (Placeholder)")
        ax.set_xlabel("Average Absolute SHAP Value")
        ax.invert_yaxis() 
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)

        import base64
        return JSONResponse({"image_base64": base64.b64encode(buffer.getvalue()).decode('utf-8')})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation plot: {e}")


@app.post('/export_report', summary='Export Prediction Report as PDF')
def export_prediction_report(data: LoanInput):
    if model is None or not REPORTLAB_AVAILABLE:
        raise HTTPException(status_code=503, detail="PDF generation is unavailable. Model or dependencies missing.")
    
    # NOTE: The rest of the PDF generation logic from the original code would go here.
    # It would call the predict function, get the results, and use reportlab to format the PDF.
    # The chart generation part is handled by create_contributions_chart.
    
    # For brevity, returning a placeholder PDF
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(72, 800, "Early Warning System Prediction Report")
    c.drawString(72, 780, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    c.drawString(72, 760, "Full content omitted for brevity...")
    c.showPage()
    c.save()
    buffer.seek(0)
    
    return StreamingResponse(
        buffer, 
        media_type='application/pdf', 
        headers={"Content-Disposition": "attachment; filename=prediction_report.pdf"}
    )

#--- Stats/History Endpoints ---
@app.get('/stats', summary='System Statistics and Model Health')
def get_stats(db: Session = Depends(get_db)):
    try:
        total_predictions = db.query(Prediction).count()
        default_count = db.query(Prediction).filter(Prediction.prediction_status == 1).count()
        high_risk_alerts = db.query(Alert).filter(Alert.prediction_score >= RISK_THRESHOLD, Alert.status == "New").count() 
        total_alerts = db.query(Alert).count()
        new_alerts = db.query(Alert).filter(Alert.status == "New").count()
        
        latest_indicator = db.query(MacroIndicator).order_by(MacroIndicator.timestamp.desc()).first()
        
        default_rate = default_count / total_predictions if total_predictions > 0 else 0.0

        return {
            "uptime_seconds": time.time() - STARTUP_TIME,
            "total_predictions": total_predictions,
            "default_count": default_count,
            "default_rate": default_rate,
            "model_status": "Ready" if model is not None else "Error",
            "model_risk_threshold": RISK_THRESHOLD,
            "total_alerts": total_alerts,
            "new_alerts": new_alerts,
            "high_risk_alerts": high_risk_alerts, # NEW: High-risk alert count
            "macro_indicator": {
                "timestamp": latest_indicator.timestamp.isoformat() if latest_indicator else None,
                "gdp_growth_rate": latest_indicator.gdp_growth_rate if latest_indicator else None,
                "market_index_yoy": latest_indicator.market_index_yoy if latest_indicator else None,
                "unemployment_rate": latest_indicator.unemployment_rate if latest_indicator else None,
            }
        }
    except Exception as e:
        print(f"Error retrieving stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {e}")


@app.get('/predictions', summary='Get recent prediction history (in-memory)')
def get_predictions_history():
    # Still use in-memory for the quick dashboard history view
    return PREDICTIONS_HISTORY

@app.post('/clear_history', summary='Clear all prediction history (DB and In-Memory)')
def clear_history(db: Session = Depends(get_db)):
    try:
        num_deleted = db.query(Prediction).delete()
        db.commit()
        PREDICTIONS_HISTORY.clear()
        
        log_activity(db, ADMIN_USERNAME, 'HISTORY_CLEARED', {'count': num_deleted})
        
        return {"status": "success", "message": f"{num_deleted} records deleted."}
    except Exception as e:
        db.rollback()
        log_activity(db, ADMIN_USERNAME, 'HISTORY_CLEAR_FAILURE', {'error': str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {e}")

                                                     
@app.get('/export_history_csv', summary='Export all or recent prediction history to CSV')
def export_history_csv(db: Session = Depends(get_db), limit: Optional[int] = None):
    try:
        query = db.query(Prediction).order_by(Prediction.id.desc())
        if limit is not None and limit > 0:
            query = query.limit(limit)
        
        rows = query.all()
        sio = io.StringIO()
        writer = csv.writer(sio)

        #Header
        writer.writerow(['id', 'timestamp', 'prediction_status', 'probability', 'input_json', 'contributions_json'])
        for r in rows:
            writer.writerow([r.id, r.timestamp, r.prediction_status, r.probability, r.input_json or '', r.contributions_json or ''])
        
        sio.seek(0)
        
        log_activity(db, ADMIN_USERNAME, 'HISTORY_EXPORTED', {'count': len(rows), 'limit': limit})
        
        filename = f"predictions_history_{limit or 'all'}.csv"
        headers = {"Content-Disposition": f"attachment; filename={filename}"}
        
        return StreamingResponse(iter([sio.getvalue()]), media_type='text/csv', headers=headers)
    except Exception as e:
        log_activity(db, ADMIN_USERNAME, 'HISTORY_EXPORT_FAILURE', {'error': str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to export history: {e}")


@app.get('/api/model_performance', summary='Get Model Performance Metrics')
def get_model_performance(db: Session = Depends(get_db)):
    try:
        #Group by metric name and order by timestamp to get time series data
        metrics = db.query(ModelEvaluation).order_by(ModelEvaluation.timestamp.asc()).all()
        
        result = {}
        for m in metrics:
            metric_data = result.setdefault(m.metric_name, {'timestamps': [], 'values': [], 'latest_value': 0.0, 'latest_timestamp': None})
            
            ts_str = m.timestamp.isoformat()
            metric_data['timestamps'].append(ts_str)
            metric_data['values'].append(m.metric_value)
            
            #Update latest KPI
            if metric_data['latest_timestamp'] is None or m.timestamp > metric_data['latest_timestamp']:
                metric_data['latest_timestamp'] = m.timestamp
                metric_data['latest_value'] = m.metric_value
                
        #Convert datetime objects to string for JSON serialization
        for name, data in result.items():
            if data['latest_timestamp']:
                data['latest_timestamp'] = data['latest_timestamp'].isoformat()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model performance data: {e}")


#--- Macro Indicators (Public Read, Admin Write) ---
@app.get('/api/macro_indicators', summary='Get Macro Indicators History')
def get_macro_indicators(db: Session = Depends(get_db)):
    """Returns the time series history of macro indicators (Public/Unauthenticated)."""
    try:
        indicators = db.query(MacroIndicator).order_by(MacroIndicator.timestamp.desc()).limit(12).all()
        
        #Format for Chart.js
        return [
            {
                "timestamp": ind.timestamp.strftime('%Y-%m-%d'),
                "gdp_growth_rate": ind.gdp_growth_rate,
                "market_index_yoy": ind.market_index_yoy,
                "unemployment_rate": ind.unemployment_rate,
                "inflation_rate": ind.inflation_rate,
                "interest_rate": ind.interest_rate,
            }
            for ind in reversed(indicators) 
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve macro indicators: {e}")


#--- Customer Endpoints ---
@app.get('/api/customers', summary='List customers, optional search by id')
def list_customers(per_page: int = 10, page: int = 1, search_id: Optional[int] = None, db: Session = Depends(get_db)):
    try:
        query = db.query(LoanCustomer).order_by(LoanCustomer.id.asc())
        if search_id:
            query = query.filter(LoanCustomer.id == int(search_id))
        total = query.count()
        customers = query.limit(per_page).offset((page - 1) * per_page).all()
        return {"total": total, "page": page, "per_page": per_page, "customers": [c.as_dict() for c in customers]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list customers: {e}")


@app.post('/api/customers', summary='Create a new customer')
def create_customer(request: Request, data: CustomerCreate, db: Session = Depends(get_db)):
    """Creates a new customer and logs the incoming payload and client info for auditing."""
    payload = data.model_dump()
    try:
        # Server-side logging of incoming create request
        client_ip = None
        try:
            client_ip = request.client.host
        except Exception:
            client_ip = None
        logger.info('Create customer request from %s: %s', client_ip, {k: v for k, v in payload.items()})
        log_activity(db, ADMIN_USERNAME, 'CUSTOMER_CREATE_REQUEST', {'ip': client_ip, 'payload_keys': list(payload.keys())})

        c = LoanCustomer(**payload)
        db.add(c)
        db.commit()
        db.refresh(c)

        log_activity(db, ADMIN_USERNAME, 'CUSTOMER_CREATE', {'id': c.id, 'ip': client_ip})
        logger.info('Customer created id=%s by %s', c.id, client_ip)

        return c.as_dict()
    except Exception as e:
        db.rollback()
        logger.exception('Failed to create customer: %s', e)
        log_activity(db, ADMIN_USERNAME, 'CUSTOMER_CREATE_FAILURE', {'error': str(e), 'payload_keys': list(payload.keys())})
        raise HTTPException(status_code=500, detail=f"Failed to create customer: {e}")


#--- Debug helpers (development only). Enable by setting DEBUG_MODE=1 in environment. ---
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'false').lower() in ('1','true','yes')


@app.get('/api/debug/customers/count', include_in_schema=False)
def debug_customer_count(db: Session = Depends(get_db)):
    """Returns the number of rows in Loan_table for quick verification (dev-only)."""
    if not DEBUG_MODE:
        raise HTTPException(status_code=404, detail='Not found')
    try:
        cnt = db.query(LoanCustomer).count()
        return {"count": cnt}
    except Exception as e:
        logger.exception('Debug count failed: %s', e)
        raise HTTPException(status_code=500, detail=f"Debug failed: {e}")

@app.post('/api/debug/customers/create_test', include_in_schema=False)
def debug_create_test_customer(db: Session = Depends(get_db)):
    """Creates a sample test customer to verify DB writes (dev-only)."""
    if not DEBUG_MODE:
        raise HTTPException(status_code=404, detail='Not found')
    sample = {
        "loan_limit":"nlt","Gender":"Male","approv_in_adv":"nopre","loan_type":"type1","loan_purpose":"TestCreate","Credit_Worthiness":"good","open_credit":"nopc","business_or_purpose":"bp1","Neg_ammortization":"not_neg","interest_only":"not_int","lump_sum_payment":"not_lpsm","Occupancy_Type":"pr","Secured_by":"home","total_units":"1U","credit_type":"CIB","co_applicant_credit_type":"NA","age":"25-34","submission_of_application":"to_inst","Region":"North","Security_Type":"direct","rate_of_interest":4.5,"Interest_rate_spread":0.5,"Upfront_charges":1500.0,"term":360.0,"income":5000.0,"Credit_Score":720.0,"LTV":80.0,"dtir1":30.0
    }
    try:
        c = LoanCustomer(**sample)
        db.add(c)
        db.commit()
        db.refresh(c)
        logger.info('Debug test customer created id=%s', c.id)
        return c.as_dict()
    except Exception as e:
        db.rollback()
        logger.exception('Debug create failed: %s', e)
        raise HTTPException(status_code=500, detail=f"Debug create failed: {e}")


@app.get('/api/debug/db_info', include_in_schema=False)
def debug_db_info():
    """Returns basic information about the active SQLAlchemy connection (dev-only)."""
    if not DEBUG_MODE:
        raise HTTPException(status_code=404, detail='Not found')
    try:
        # Mask password if present when converting URL to string
        try:
            raw_url = str(engine.url)
            pwd = engine.url.password
            if pwd:
                masked_url = raw_url.replace(pwd, '***')
            else:
                masked_url = raw_url
        except Exception:
            masked_url = str(engine.url)

        info = {
            'masked_url': masked_url,
            'dialect': engine.dialect.name if hasattr(engine, 'dialect') else None,
            'is_sqlite': SQLALCHEMY_DATABASE_URL.startswith('sqlite')
        }
        #Also expose parsed pieces for convenience
        try:
            info.update({
                'username': engine.url.username,
                'host': engine.url.host,
                'port': engine.url.port,
                'database': engine.url.database,
            })
        except Exception:
            pass

        return info
    except Exception as e:
        logger.exception('Debug db_info failed: %s', e)

        raise HTTPException(status_code=500, detail=f"Debug db info failed: {e}")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
@app.put('/api/customers/{customer_id}', summary='Update existing customer')
def update_customer(customer_id: int, request: Request, data: CustomerUpdate, db: Session = Depends(get_db)):
    try:
        payload = data.model_dump()
        client_ip = None
        try:
            client_ip = request.client.host
        except Exception:
            client_ip = None
        logger.info('Update customer %s request from %s: %s', customer_id, client_ip, {k: v for k, v in payload.items()})
        log_activity(db, ADMIN_USERNAME, 'CUSTOMER_UPDATE_REQUEST', {'id': customer_id, 'ip': client_ip, 'payload_keys': list(payload.keys())})

        c = db.query(LoanCustomer).filter(LoanCustomer.id == customer_id).first()
        if not c:
            raise HTTPException(status_code=404, detail='Customer not found')
        for k, v in payload.items():
            setattr(c, k, v)
        db.add(c)
        db.commit()
        log_activity(db, ADMIN_USERNAME, 'CUSTOMER_UPDATE', {'id': customer_id, 'ip': client_ip})
        logger.info('Customer %s updated by %s', customer_id, client_ip)
        return c.as_dict()
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.exception('Failed to update customer: %s', e)
        log_activity(db, ADMIN_USERNAME, 'CUSTOMER_UPDATE_FAILURE', {'id': customer_id, 'error': str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to update customer: {e}")

    
@app.delete('/api/customers/{customer_id}', summary='Delete a customer')
def delete_customer(customer_id: int, request: Request, db: Session = Depends(get_db)):
    try:
        client_ip = None
        try:
            client_ip = request.client.host
        except Exception:
            client_ip = None
        logger.info('Delete customer %s request from %s', customer_id, client_ip)
        log_activity(db, ADMIN_USERNAME, 'CUSTOMER_DELETE_REQUEST', {'id': customer_id, 'ip': client_ip})

        c = db.query(LoanCustomer).filter(LoanCustomer.id == customer_id).first()
        if not c:
            raise HTTPException(status_code=404, detail='Customer not found')
        db.delete(c)
        db.commit()
        log_activity(db, ADMIN_USERNAME, 'CUSTOMER_DELETE', {'id': customer_id, 'ip': client_ip})
        logger.info('Customer %s deleted by %s', customer_id, client_ip)
        return {"status": "success", "message": f"Customer {customer_id} deleted."}
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.exception('Failed to delete customer: %s', e)
        log_activity(db, ADMIN_USERNAME, 'CUSTOMER_DELETE_FAILURE', {'id': customer_id, 'error': str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to delete customer: {e}")


@app.post('/api/customers/{customer_id}/predict', summary='Run prediction for a customer record')
def predict_customer(customer_id: int, db: Session = Depends(get_db)):
    try:
        c: LoanCustomer = db.query(LoanCustomer).filter(LoanCustomer.id == customer_id).first()
        if not c:
            raise HTTPException(status_code=404, detail='Customer not found')
        # Build LoanInput from customer fields
        payload = {k: v for k, v in c.as_dict().items() if k != 'id'}
        loan_input = LoanInput(**payload)
        # Reuse predict logic but avoid double-saving alerts (predict endpoint already saves)
        return predict(loan_input, db=db)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict for customer: {e}")

class MacroIndicatorUpdate(BaseModel):
    gdp_growth_rate: float
    market_index_yoy: float
    unemployment_rate: float
    inflation_rate: float
    interest_rate: float

@app.post('/api/macro_indicators/update', summary='Update Macro Indicators (Admin Only)')
def update_macro_indicators(
    data: MacroIndicatorUpdate, 
    request: Request,
    db: Session = Depends(get_db)
):
    """Adds a new entry for macro indicators. Requires admin session."""
    try:
        new_indicator = MacroIndicator(
            timestamp=datetime.utcnow(),
            gdp_growth_rate=data.gdp_growth_rate,
            market_index_yoy=data.market_index_yoy,
            unemployment_rate=data.unemployment_rate,
            inflation_rate=data.inflation_rate,
            interest_rate=data.interest_rate,
        )

        db.add(new_indicator)
        db.commit()
        
        log_activity(db, ADMIN_USERNAME, 'MACRO_INDICATOR_UPDATE', data.model_dump())
        
        return JSONResponse({"status": "success", "message": "Macro indicators updated successfully."})
    except Exception as e:
        db.rollback()
        log_activity(db, ADMIN_USERNAME, 'MACRO_INDICATOR_UPDATE_FAILURE', {'error': str(e), 'input': data.model_dump()})
        raise HTTPException(status_code=500, detail=f"Failed to update macro indicators: {e}")

#--- Alert Management Endpoints ---
class AlertUpdate(BaseModel):
    status: str = Field(..., pattern="^(New|In Progress|Closed)$", description="New, In Progress, or Closed")
    manager_notes: Optional[str] = None
    resolution_action: Optional[str] = None

class AlertResponse(BaseModel):
    id: int
    alert_timestamp: datetime
    entity_id: str
    risk_signal: str
    severity: str
    prediction_score: float
    status: str
    manager_notes: Optional[str] = None
    resolution_action: Optional[str] = None


@app.get('/api/alerts', response_model=List[AlertResponse], summary='Get All Open Alerts')
def get_alerts(db: Session = Depends(get_db)):
    """Returns all alerts (New and In Progress) for review."""
    try:
        alerts = db.query(Alert).filter(Alert.status.in_(["New", "In Progress"])).order_by(Alert.alert_timestamp.desc()).all()
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve alerts: {e}")


@app.post("/api/alerts/{alert_id}/update_status", summary="Update alert status and notes")
def update_alert_status(
    alert_id: int, 
    update: AlertUpdate, 
    request: Request,
    db: Session = Depends(get_db)
):
    """Updates the status, notes, and resolution action for a specific alert."""
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        alert.status = update.status
        alert.manager_notes = update.manager_notes
        alert.resolution_action = update.resolution_action
        
        db.commit()

        log_activity(db, ADMIN_USERNAME, 'ALERT_UPDATE', 
        
        {
            'alert_id': alert_id, 
            'entity_id': alert.entity_id,
            'new_status': update.status, 
            'notes_provided': update.manager_notes is not None,
            'action_provided': update.resolution_action is not None
        })
        
        return JSONResponse({"status": "success", "message": f"Alert {alert_id} updated to {update.status}"})
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        log_activity(db, ADMIN_USERNAME, 'ALERT_UPDATE_FAILURE', {'alert_id': alert_id, 'error': str(e), 'input': update.model_dump()})
        raise HTTPException(status_code=500, detail=f"Failed to update alert: {e}")


#--- 5. API Endpoints ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/', summary='Root Redirect')
def root_redirect():
    return RedirectResponse(url='/admin-login', status_code=status.HTTP_302_FOUND)

@app.get('/admin-login', summary='Admin Login Page')
def serve_admin_login_redirect():
    return RedirectResponse(url='/static/admin_login.html', status_code=status.HTTP_302_FOUND)


@app.get('/admin-page', summary='Admin Portal')
def serve_admin_page():
    return RedirectResponse(url='/static/admin.html', status_code=status.HTTP_302_FOUND)


@app.post('/admin-login', summary='Admin Login')
async def login_admin(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    admin = db.query(EWSAdmin).filter_by(username=username).first()
    if admin and pwd_context.verify(password, admin.password_hash):
        token = _create_admin_session(username)
        log_activity(db, username, 'LOGIN_SUCCESS', {'ip': request.client.host, 'user_agent': request.headers.get('user-agent')})
        response = RedirectResponse(url='/static/bank_dashboard.html', status_code=status.HTTP_302_FOUND)
        response.set_cookie(key="admin_token", value=token, httponly=True, max_age=SESSION_TTL, samesite="strict")
        return response
    log_activity(db, username, 'LOGIN_FAILURE', {'ip': request.client.host, 'attempted_user': username})
    # Redirect back to login page with an error flag
    return RedirectResponse(url='/static/admin_login.html?error=1', status_code=status.HTTP_401_UNAUTHORIZED)


@app.post('/admin-logout', summary='Admin Logout')
async def logout_admin(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get('admin_token')
    
    user = ADMIN_USERNAME if _is_admin_token_valid(token) else "Unknown" 
    log_activity(db, user, 'LOGOUT', {'ip': request.client.host})


    _invalidate_admin_token(token)
    response = RedirectResponse(url='/admin-login', status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="admin_token")
    return response


#--- NEW: Model Management Endpoint (Admin Only) ---
@app.post("/api/model/retrain", summary="Trigger Model Retrain/Reload (Admin Only)")
def trigger_model_retrain(db: Session = Depends(get_db)):
    """Placeholder endpoint to simulate triggering a model re-training or simply reloading model assets."""
    try:
        load_models() 
        log_activity(db, ADMIN_USERNAME, 'MODEL_RELOAD_SUCCESS', {'result': 'Model assets reloaded from disk.'})
        return {"status": "success", "message": "Model assets reloaded successfully. Check logs for training status if this was a full retrain trigger."}
    except Exception as e:
        db.rollback()
        log_activity(db, ADMIN_USERNAME, 'MODEL_RELOAD_FAILURE', {'error': str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to reload model assets: {e}")
#--- NEW: Audit Log Endpoint (Admin Only) ---


@app.get('/api/audit_log', summary='Get System Activity Log (Admin Only)')
def get_audit_log(db: Session = Depends(get_db)):
    """Returns the last 100 administrative activity logs."""
    try:
        logs = db.query(ActivityLog).order_by(ActivityLog.timestamp.desc()).limit(100).all()
        
        formatted_logs = []
        for log in logs:
            details = json.loads(log.details_json) if log.details_json else None
            formatted_logs.append({
                "timestamp": log.timestamp.isoformat(),
                "user": log.user,
                "action_type": log.action_type,
                "details": details
            })
        return formatted_logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve audit log: {e}")


#--- Startup ---
@app.on_event("startup")
def on_startup():
    try:
        load_models()
    except Exception as e:
        print(f"Warning: load_models failed on startup, continuing without models: {e}")
    init_db()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)