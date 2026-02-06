import sqlite3
import os

# Connect to the database
db_path = 'data/predictions.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

#Check the structure of macro_indicators table
try:
    cursor.execute("PRAGMA table_info(macro_indicators);")
    columns = cursor.fetchall()
    print("Macro indicators table structure:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
except Exception as e:
    print(f"Error checking table structure: {e}")
    

conn.close()