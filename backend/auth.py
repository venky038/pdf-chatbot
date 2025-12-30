import os
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from dotenv import load_dotenv
from sqlalchemy.orm import Session
import models

"""
Security & Token Logic (JWT + Hashing)
This file handles the low-level security mechanics of the application.
"""

load_dotenv()

# --- SECURITY CONFIGURATION ---
# SECRET_KEY is used to digitally sign tokens. It must be kept private.
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a_very_bad_default_secret_key_change_me")
ALGORITHM = "HS256"
# Tokens expire after 7 days by default to balance security and convenience.
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 

# Hashing context using 'argon2' (more secure against GPU-based attacks than standard bcrypt)
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# Tells FastAPI where to look for the login endpoint to automate OAuth2 flow
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- CORE SECURITY FUNCTIONS ---

def verify_password(plain_password, hashed_password):
    """Checks if a user-provided password matches the stored hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Converts a plaintext password into a secure, non-reversible cryptographic hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Creates a digitally signed JWT containing user identity and expiration timestamp."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = None):
    """
    Middleware Dependency:
    1. Extracts the token from the request header.
    2. Validates the signature and expiration.
    3. Retrieves the User object from the database based on the 'sub' claim.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode and verify the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Internal DB session management if not provided (fallback)
    if db is None:
        from database import SessionLocal
        db = SessionLocal()
        try:
            user = db.query(models.User).filter(models.User.username == username).first()
        finally:
            db.close()
    else:
        user = db.query(models.User).filter(models.User.username == username).first()
        
    if user is None:
        raise credentials_exception
        
    return user