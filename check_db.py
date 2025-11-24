import sqlite3
import os

# Check if the database file exists
db_path = 'data/predictions.db'
if not os.path.exists(db_path):
    print("Database file does not exist")
else:
    print("Database file exists")
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in the database:")
    for table in tables:
        print(f"  - {table[0]}")
        
    # Check if model_evaluations table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_evaluations';")
    result = cursor.fetchone()
    if result:
        print("\nModel evaluations table exists")
        
        # Check if there's data in the table
        cursor.execute("SELECT COUNT(*) FROM model_evaluations;")
        count = cursor.fetchone()[0]
        print(f"Number of records in model_evaluations: {count}")
        
        if count > 0:
            # Show sample data
            cursor.execute("SELECT * FROM model_evaluations LIMIT 5;")
            rows = cursor.fetchall()
            print("\nSample data from model_evaluations:")
            for row in rows:
                print(row)
    else:
        print("\nModel evaluations table does not exist")
        
    conn.close()