import logging
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from models import User
from auth import get_password_hash, verify_password, create_access_token

# Dedicated logger for security/authentication auditing
logger = logging.getLogger("QueryMate.Auth")

class AuthService:
    """
    Core authentication service that manages User registration,
    password validation, and JWT token issuance.
    """

    @staticmethod
    def register_user(db: Session, username: str, password: str):
        """
        Creates a new user record after ensuring the username is unique.
        Passwords are never stored as plaintext (they are hashed using bcrypt).
        """
        # 1. Check for existing user
        if db.query(User).filter(User.username == username).first():
            logger.warning(f"Registration failed: Username '{username}' is already taken.")
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # 2. Securely hash the password
        hashed_password = get_password_hash(password)
        user = User(username=username, hashed_password=hashed_password)
        
        # 3. Save to database
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"User created successfully: {username} (ID: {user.id})")
        return user

    @staticmethod
    def authenticate_user(db: Session, username: str, password: str):
        """
        Verifies credentials by comparing the provided password with the stored hash.
        """
        user = db.query(User).filter(User.username == username).first()
        if not user:
            logger.info(f"Auth failed: User '{username}' does not exist.")
            return None
            
        if not verify_password(password, user.hashed_password):
            logger.warning(f"Auth failed: Incorrect password for user '{username}'.")
            return None
            
        return user

    @staticmethod
    def login_for_access_token(db: Session, username: str, password: str):
        """
        Validates credentials and issues a secure JWT access token for the session.
        """
        user = AuthService.authenticate_user(db, username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create a token containing the username (the 'sub' claim)
        access_token = create_access_token(data={"sub": user.username})
        logger.info(f"JWT issued for session: {username}")
        return {"access_token": access_token, "token_type": "bearer"}
