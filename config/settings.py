# config/settings.py

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file          = ".env",
        env_file_encoding = "utf-8",
        case_sensitive    = False,
        extra             = "ignore"
    )

    # ==========================================
    # LLM
    # ==========================================
    groq_api_key: str
    llm_model_name: str = "llama-3.3-70b-versatile"

    # ==========================================
    # DATABASE
    # ==========================================
    database_url: str
    postgres_user: str     = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str       = "ecommerce_agent_db"

    # ==========================================
    # LANGFUSE
    # ==========================================
    langfuse_secret_key: str = "placeholder"
    langfuse_public_key: str = "placeholder"
    langfuse_host: str       = "http://localhost:3000"

    # ==========================================
    # GOOGLE OAUTH
    # ==========================================
    google_client_id: str      = "placeholder"
    google_client_secret: str  = "placeholder"
    google_redirect_uri: str   = "http://localhost:8000"

    # ==========================================
    # SESSION
    # ==========================================
    session_expiry_minutes: int = 30
    history_window_size: int    = 10

    # ==========================================
    # APPLICATION
    # ==========================================
    environment: str = "development"
    log_level: str   = "INFO"
    app_host: str    = "0.0.0.0"
    app_port: int    = 8000

    # ==========================================
    # PROMETHEUS
    # ==========================================
    prometheus_port: int = 9090

    # ==========================================
    # LANGFUSE PROMPT VERSIONS
    # ==========================================
    intent_router_prompt_label: str  = "production"
    order_response_prompt_label: str = "production"
    order_analysis_prompt_label: str = "production"

    # ==========================================
    # MONITORING
    # ==========================================
    grafana_admin_password: str    = "admin"
    langfuse_nextauth_secret: str  = "mysecret"
    langfuse_salt: str             = "mysalt"

    # ==========================================
    # GROQ PRICING
    # ==========================================
    groq_input_cost_per_million: float  = 0.59
    groq_output_cost_per_million: float = 0.79

    # ==========================================
    # INTENT ROUTER
    # ==========================================
    intent_confidence_threshold: float = 0.7

    # ==========================================
    # SUPPORT AGENT
    # ==========================================
    support_high_value_threshold: float = 1500.0

    # ==========================================
    # PRODUCT AGENT
    # ==========================================
    product_search_candidates: int       = 20
    product_recommendation_count: int    = 8
    product_default_max_price: float     = 100000.0
    product_price_broaden_factor: float  = 1.5

    # ==========================================
    # EVALUATION
    # ==========================================
    eval_user_id: str = ""


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
