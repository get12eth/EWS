#!/usr/bin/env python3
"""
Database initialization script for the Early Warning System.
This script creates all necessary database tables.
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

# Database configuration
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "predictions.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()

# Define models
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

def init_database():
    """Initialize the database by creating all tables."""
    print("Initializing database...")
    
    # Create data directory if it doesn't exist
    os.makedirs(DB_DIR, exist_ok=True)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_database()