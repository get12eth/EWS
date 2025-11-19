#!/usr/bin/env python3
"""
Script to check alerts in the database.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from init_db import Alert, Base

# Database configuration
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "predictions.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def check_alerts():
    """Check all alerts in the database."""
    db = SessionLocal()
    alerts = db.query(Alert).all()
    db.close()
    
    print(f"Found {len(alerts)} alerts in the database:")
    for alert in alerts:
        print(f"  ID: {alert.id}, Entity: {alert.entity_id}, Signal: {alert.risk_signal}, "
              f"Severity: {alert.severity}, Score: {alert.prediction_score}, Status: {alert.status}")

if __name__ == "__main__":
    check_alerts()