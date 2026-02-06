#!/usr/bin/env python3
"""
Database initialization script for the Early Warning System.
This script creates all necessary database tables.
"""
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

#Database configuration
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "predictions.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()

#Define models
class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, index=True)
    input_json = Column(Text)
    prediction_status = Column(Integer, index=True)
    probability = Column(Float)
    contributions_json = Column(Text)

class Alert(Base):
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(String, index=True)
    alert_timestamp = Column(DateTime, default=datetime.utcnow)
    risk_signal = Column(String)
    severity = Column(String)
    prediction_score = Column(Float)
    status = Column(String, default='New')
    manager_notes = Column(Text, nullable=True)
    resolution_action = Column(String, nullable=True)

class MacroIndicator(Base):
    __tablename__ = 'macro_indicators'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    gdp_growth_rate = Column(Float)
    market_index_yoy = Column(Float)
    unemployment_rate = Column(Float)
    inflation_rate = Column(Float)
    interest_rate = Column(Float)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1)

class EWSAdmin(Base):
    __tablename__ = 'ews_admins'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    

# Simple DB session helper for utility functions
from sqlalchemy.orm import sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Password utilities using passlib
from passlib.context import CryptContext
# Use pbkdf2_sha256 to avoid bcrypt dependency issues in CI/vagrant where bcrypt may be missing
_pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def get_password_hash(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return _pwd_context.verify(plain_password, hashed_password)


def get_user_by_username(username: str):
    db = SessionLocal()
    try:
        return db.query(User).filter(User.username == username).first()
    finally:
        db.close()


def get_user_by_email(email: str):
    db = SessionLocal()
    try:
        return db.query(User).filter(User.email == email).first()
    finally:
        db.close()


def create_user(username: str, email: str, password: str):
    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == username).first():
            return None
        hashed = get_password_hash(password)
        user = User(username=username, email=email, hashed_password=hashed)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def update_user_password(username: str, new_password: str) -> bool:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return False
        user.hashed_password = get_password_hash(new_password)
        db.add(user)
        db.commit()
        return True
    finally:
        db.close()


def create_default_admin():
    db = SessionLocal()
    try:
        admin = db.query(EWSAdmin).filter_by(username='admin').first()
        if not admin:
            from passlib.context import CryptContext
            pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
            password_hash = pwd_context.hash('Get@6963')
            admin = EWSAdmin(username='admin', password_hash=password_hash)
            db.add(admin)
            db.commit()
            print('Default admin user created in ews_admins table.')
    except Exception as e:
        print(f'Error creating default admin: {e}')
    finally:
        db.close()


def init_database():
    """Initialize the database by creating all tables."""
    print("Initializing database...")
    
    # Create data directory if it doesn't exist
    os.makedirs(DB_DIR, exist_ok=True)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    print("Database initialized successfully!")

if __name__ == "__main__":
    print("Initializing database...")
    Base.metadata.create_all(engine)
    create_default_admin()
    print("Database initialized successfully!")