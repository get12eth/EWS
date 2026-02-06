#!/usr/bin/env python3
"""
Script to test saving alerts directly to the database.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from init_db import Alert, Base
from datetime import datetime

#Database configuration
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "predictions.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_save_alert():
    """Test saving an alert directly to the database."""
    #Create a test alert
    alert_data = {
        "entity_id": "test_loan_123",
        "risk_signal": "Test High Default Probability",
        "severity": "High",
        "prediction_score": 0.5031,
        "status": "New"
    }
    
    try:
        db = SessionLocal()
        alert = Alert(**alert_data)
        db.add(alert)
        db.commit()
        db.refresh(alert)
        db.close()
        
        print(f"Successfully saved alert with ID: {alert.id}")
        
        #Retrieve the alert to verify it was saved
        db = SessionLocal()
        saved_alert = db.query(Alert).filter(Alert.id == alert.id).first()
        db.close()
        
        if saved_alert:
            print(f"Retrieved alert: ID={saved_alert.id}, Entity={saved_alert.entity_id}, "
                  f"Signal={saved_alert.risk_signal}, Severity={saved_alert.severity}, "
                  f"Score={saved_alert.prediction_score}, Status={saved_alert.status}")
        else:
            print("Failed to retrieve the saved alert")
            
    except Exception as e:
        print(f"Error saving alert: {e}")

if __name__ == "__main__":
    test_save_alert()