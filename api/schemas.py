# api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, Union


# ==========================================
# REQUEST SCHEMAS
# ==========================================


class ChatRequest(BaseModel):
    """
    Every message sent from the frontend to FastAPI.
    Authenticated users send JWT in Authorization header.
    Guest users send guest_id in body.
    """

    message: str = Field(..., min_length=1, max_length=2000, description="The user's message")
    guest_id: Optional[str] = Field(default=None, description="Guest user ID for session continuity")


# ==========================================
# RESPONSE SCHEMAS
# ==========================================


class ChatResponse(BaseModel):
    """
    Every response returned from FastAPI to the frontend.
    """

    response: str
    session_id: str
    user_id: str
    agent_used: str
    intent: Optional[str] = None
    intent_confidence: Optional[Union[str, float]]
    is_authenticated: bool = False
    products: Optional[list] = None


class HealthResponse(BaseModel):
    """
    Health check response.
    Used by CI/CD smoke test after deployment.
    """

    status: str
    database: str
    version: str = "1.0.0"


class MessageItem(BaseModel):
    role: str
    content: str
    agent_name: Optional[str] = None


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[MessageItem]
