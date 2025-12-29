import logging
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from models import User
from auth import get_password_hash, verify_password, create_access_token

logger = logging.getLogger(__name__)

class AuthService:
    @staticmethod
    def register_user(db: Session, username: str, password: str):
        if db.query(User).filter(User.username == username).first():
            raise HTTPException(status_code=400, detail="Username already exists")
        
        hashed_password = get_password_hash(password)
        user = User(username=username, hashed_password=hashed_password)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def authenticate_user(db: Session, username: str, password: str):
        user = db.query(User).filter(User.username == username).first()
        if not user or not verify_password(password, user.hashed_password):
            return None
        return user

    @staticmethod
    def login_for_access_token(db: Session, username: str, password: str):
        user = AuthService.authenticate_user(db, username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token = create_access_token(data={"sub": user.username})
        return {"access_token": access_token, "token_type": "bearer"}
