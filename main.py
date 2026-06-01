# main.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.routes import router, limiter
from database.connection import create_tables, test_connection
from config.settings import settings
from monitoring.logging_config import setup_logging

# ==========================================
# STRUCTURED LOGGING
# JSON format to stdout + file
# ==========================================
setup_logging(log_level=settings.log_level, log_file="logs/app.log")
logger = logging.getLogger(__name__)


# ==========================================
# LIFESPAN
# ==========================================
@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting E-Commerce AI Support System...")
    if not test_connection():
        logger.error("Database connection failed — aborting startup")
        raise RuntimeError("Database unavailable at startup")
    logger.info("Database connected successfully")
    create_tables()
    logger.info("Database tables verified")
    logger.info(
        "Application started", extra={"environment": settings.environment, "llm_model": settings.llm_model_name}
    )
    yield


# ==========================================
# FASTAPI APP
# ==========================================
app = FastAPI(
    title="E-Commerce AI Support System",
    description="Multi-agent AI customer support powered by LangGraph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ==========================================
# CORS MIDDLEWARE
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Session-ID"],
)

# ==========================================
# PROMETHEUS METRICS
# Exposes /metrics endpoint automatically
# Records: request count, latency, status codes
# ==========================================
Instrumentator().instrument(app).expose(app)

# ==========================================
# RATE LIMITING
# ==========================================
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==========================================
# ROUTES
# ==========================================
app.include_router(router)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/app")
async def serve_frontend():
    return FileResponse("frontend/index.html")


# ==========================================
# ROOT
# ==========================================
@app.get("/")
async def root():
    return {"message": "E-Commerce AI Support System", "version": "1.0.0", "docs": "/docs", "health": "/health"}
