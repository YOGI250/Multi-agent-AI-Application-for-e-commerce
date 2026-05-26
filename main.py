# main.py

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator

from api.routes import router
from database.connection import create_tables, test_connection
from config.settings import settings
from monitoring.logging_config import setup_logging

# ==========================================
# STRUCTURED LOGGING
# JSON format to stdout + file
# ==========================================
setup_logging(
    log_level = settings.log_level,
    log_file  = "logs/app.log"
)
logger = logging.getLogger(__name__)

# ==========================================
# FASTAPI APP
# ==========================================
app = FastAPI(
    title       = "E-Commerce AI Support System",
    description = "Multi-agent AI customer support powered by LangGraph",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc"
)

# ==========================================
# CORS MIDDLEWARE
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:8000",
                         "http://localhost:3000"],
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["Authorization", "Content-Type", "X-Session-ID"]
)

# ==========================================
# PROMETHEUS METRICS
# Exposes /metrics endpoint automatically
# Records: request count, latency, status codes
# ==========================================
Instrumentator().instrument(app).expose(app)

# ==========================================
# ROUTES
# ==========================================
app.include_router(router)
app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)

@app.get("/app")
async def serve_frontend():
    return FileResponse("frontend/index.html")

# ==========================================
# STARTUP
# ==========================================
@app.on_event("startup")
async def startup():
    logger.info("Starting E-Commerce AI Support System...")

    if test_connection():
        logger.info("Database connected successfully")
    else:
        logger.error("Database connection failed")

    create_tables()
    logger.info("Database tables verified")
    logger.info(
        "Application started",
        extra={
            "environment": settings.environment,
            "llm_model":   settings.llm_model_name
        }
    )

# ==========================================
# ROOT
# ==========================================
@app.get("/")
async def root():
    return {
        "message": "E-Commerce AI Support System",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health"
    }