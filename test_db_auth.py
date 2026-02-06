import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from init_db import User, Base, get_password_hash, verify_password, get_user_by_username, create_user, update_user_password, get_user_by_email

#Set up database connection
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "predictions.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_database_auth():
    print("Testing database authentication functionality...")
    
    #Create tables
    Base.metadata.create_all(bind=engine)
    
    #Clean up any existing test user
    db = SessionLocal()
    test_user = db.query(User).filter(User.username == "testuser").first()
    if test_user:
        db.delete(test_user)
        db.commit()
    db.close()
    
    #Test creating a user
    print("\n1. Testing user creation...")
    user = create_user("testuser", "test@example.com", "testpassword")
    if user:
        print("✓ User created successfully")
        print(f"  Username: {user.username}")
        print(f"  Email: {user.email}")
    else:
        print("✗ Failed to create user")
        return
    
    #Test getting user by username
    print("\n2. Testing user retrieval...")
    retrieved_user = get_user_by_username("testuser")
    if retrieved_user:
        print("✓ User retrieved successfully")
        print(f"  Username: {retrieved_user.username}")
        print(f"  Email: {retrieved_user.email}")
    else:
        print("✗ Failed to retrieve user")
        return
    
    #Test password verification
    print("\n3. Testing password verification...")
    if retrieved_user and verify_password("testpassword", retrieved_user.hashed_password):
        print("✓ Password verification successful")
    else:
        print("✗ Password verification failed")
        return
    
    #Test incorrect password
    if retrieved_user and not verify_password("wrongpassword", retrieved_user.hashed_password):
        print("✓ Incorrect password correctly rejected")
    else:
        print("✗ Incorrect password was accepted")
        return
    
    #Test password update
    print("\n4. Testing password update...")
    if update_user_password("testuser", "newpassword"):
        print("✓ Password updated successfully")
        
        #Verify new password
        updated_user = get_user_by_username("testuser")
        if updated_user and verify_password("newpassword", updated_user.hashed_password):
            print("✓ New password verified successfully")
        else:
            print("✗ New password verification failed")
            return
    else:
        print("✗ Failed to update password")
        return
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_database_auth()