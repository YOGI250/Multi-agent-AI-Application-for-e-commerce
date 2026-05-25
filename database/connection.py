# database/connection.py

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool

from config.settings import settings

logger = logging.getLogger(__name__)

# ==========================================
# BASE
# all SQLAlchemy models inherit from this
# ==========================================
Base = declarative_base()

# ==========================================
# ENGINE
# actual connection pool to PostgreSQL
# ==========================================
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)

# ==========================================
# SESSION FACTORY
# ==========================================
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db():
    """
    Creates a database session.
    Closes it automatically when done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """
    Creates all tables in PostgreSQL.
    Safe to call multiple times.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("All database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise


def test_connection():
    """
    Tests PostgreSQL is reachable.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False