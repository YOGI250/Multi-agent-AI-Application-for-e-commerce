# tests/conftest.py

import os
import pytest

# set dummy env vars so settings loads without real keys
os.environ.setdefault("GROQ_API_KEY",           "test_groq_key")
os.environ.setdefault("DATABASE_URL",
    "postgresql://test:test@localhost:5432/test_db")
os.environ.setdefault("LANGFUSE_SECRET_KEY",    "test_secret")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY",    "test_public")
os.environ.setdefault("LANGFUSE_HOST",          "http://localhost:3000")
os.environ.setdefault("GOOGLE_CLIENT_ID",       "test_client_id")
os.environ.setdefault("GOOGLE_CLIENT_SECRET",   "test_client_secret")
os.environ.setdefault("POSTGRES_USER",          "test")
os.environ.setdefault("POSTGRES_PASSWORD",      "test")
os.environ.setdefault("POSTGRES_DB",            "test_db")
os.environ.setdefault("GRAFANA_ADMIN_PASSWORD", "admin")
os.environ.setdefault("LANGFUSE_NEXTAUTH_SECRET","test_secret")
os.environ.setdefault("LANGFUSE_SALT",          "test_salt")