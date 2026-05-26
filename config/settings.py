# config/settings.py

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):

    # ==========================================
    # LLM
    # ==========================================
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    llm_model_name: str = Field(
        default="llama-3.3-70b-versatile",
        env="LLM_MODEL_NAME"
    )

    # ==========================================
    # DATABASE
    # ==========================================
    database_url: str = Field(..., env="DATABASE_URL")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="ecommerce_agent_db", env="POSTGRES_DB")

    # ==========================================
    # LANGFUSE
    # ==========================================
    langfuse_secret_key: str = Field(
        default="placeholder", env="LANGFUSE_SECRET_KEY"
    )
    langfuse_public_key: str = Field(
        default="placeholder", env="LANGFUSE_PUBLIC_KEY"
    )
    langfuse_host: str = Field(
        default="http://localhost:3000", env="LANGFUSE_HOST"
    )

    # ==========================================
    # GOOGLE OAUTH
    # ==========================================
    google_client_id: str = Field(
        default="placeholder", env="GOOGLE_CLIENT_ID"
    )
    google_client_secret: str = Field(
        default="placeholder", env="GOOGLE_CLIENT_SECRET"
    )
    google_redirect_uri: str = Field(
        default="http://localhost:8000",
        env="GOOGLE_REDIRECT_URI"
    )

    # ==========================================
    # SESSION
    # ==========================================
    session_expiry_minutes: int = Field(
        default=30, env="SESSION_EXPIRY_MINUTES"
    )
    history_window_size: int = Field(
        default=10, env="HISTORY_WINDOW_SIZE"
    )

    # ==========================================
    # APPLICATION
    # ==========================================
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    app_host: str = Field(default="0.0.0.0", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")


    # ==========================================
    # PROMETHEUS
    # ==========================================
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")

    # ==========================================
    # LANGFUSE PROMPT VERSIONS
    # ==========================================
    intent_router_prompt_label: str = Field(
        default="production", env="INTENT_ROUTER_PROMPT_LABEL"
    )
    order_response_prompt_label: str = Field(
        default="production", env="ORDER_RESPONSE_PROMPT_LABEL"
    )
    order_analysis_prompt_label: str = Field(
        default="production", env="ORDER_ANALYSIS_PROMPT_LABEL"
    )

    # ==========================================
    # MONITORING
    # ==========================================
    grafana_admin_password: str = Field(
        default="admin", env="GRAFANA_ADMIN_PASSWORD"
    )
    langfuse_nextauth_secret: str = Field(
        default="mysecret", env="LANGFUSE_NEXTAUTH_SECRET"
    )
    langfuse_salt: str = Field(
        default="mysalt", env="LANGFUSE_SALT"
    )
    # ==========================================
    # GROQ PRICING
    # ==========================================
    groq_input_cost_per_million: float = Field(
        default=0.59, env="GROQ_INPUT_COST_PER_MILLION"
    )
    groq_output_cost_per_million: float = Field(
        default=0.79, env="GROQ_OUTPUT_COST_PER_MILLION"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()