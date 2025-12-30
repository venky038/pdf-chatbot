import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

"""
Database Configuration (SQLite)
This file initializes the SQLAlchemy engine and creates the base class for our models.
"""

# 1. Path Calculation: Place the database in the PROJECT ROOT, not inside the backend folder.
# This makes it easier to find and back up.
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
DB_PATH = os.path.join(PROJECT_ROOT, "chat_app.db")

# 2. Database URL: Using SQLite for simplicity and zero-config local development.
DATABASE_URL = f"sqlite:///{DB_PATH}"

# 3. SQLAlchemy Engine: 'check_same_thread' is disabled specifically for SQLite compatibility with FastAPI.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# 4. Session Factory: Used to create individual database sessions for each request.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 5. Base Class: All models in models.py must inherit from this to be recognized by the ORM.
Base = declarative_base()
