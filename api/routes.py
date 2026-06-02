# api/routes.py

import uuid
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from langfuse_helpers.tracing import create_trace, flush, calculate_cost
from langfuse_helpers.scoring import score_response
from utils.langfuse_context import set_trace_context
from monitoring.metrics import record_request_metrics, record_error
from utils.memory import merge_context
from api.schemas import ChatRequest, ChatResponse, HealthResponse, HistoryResponse
from agents.intent_router import intent_router_graph
from database.connection import SessionLocal, test_connection
from database.models import User, Session as SessionModel, Message, Order
from config.settings import settings

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

router = APIRouter()


def create_sample_orders_for_user(user_id: str, db: Session):

    today = datetime.today().date()
    sample_orders = [
        {
            "order_id": f"ORD-{user_id[-6:].upper()}-001",
            "status": "delivered",
            "carrier": "DTDC",
            "tracking_number": "DT251092059",
            "order_value": 1299.0,
            "expected_delivery": today - timedelta(days=18),
            "order_date": datetime.combine(today - timedelta(days=27), datetime.min.time()),
            "items": ["Wireless Headphones"],
        },
        {
            "order_id": f"ORD-{user_id[-6:].upper()}-002",
            "status": "shipped",
            "carrier": "BlueDart",
            "tracking_number": "BD123456789",
            "order_value": 2500.0,
            "expected_delivery": today + timedelta(days=3),
            "order_date": datetime.combine(today - timedelta(days=6), datetime.min.time()),
            "items": ["Laptop Stand"],
        },
        {
            "order_id": f"ORD-{user_id[-6:].upper()}-003",
            "status": "processing",
            "carrier": "FedEx",
            "tracking_number": "FX987654321",
            "order_value": 499.0,
            "expected_delivery": today + timedelta(days=5),
            "order_date": datetime.combine(today - timedelta(days=2), datetime.min.time()),
            "items": ["USB Hub"],
        },
        {
            "order_id": f"ORD-{user_id[-6:].upper()}-004",
            "status": "delayed",
            "carrier": "Ekart",
            "tracking_number": "EK112233445",
            "order_value": 899.0,
            "expected_delivery": today - timedelta(days=3),
            "order_date": datetime.combine(today - timedelta(days=13), datetime.min.time()),
            "items": ["Mechanical Keyboard"],
        },
        {
            "order_id": f"ORD-{user_id[-6:].upper()}-005",
            "status": "delivered",
            "carrier": "DHL",
            "tracking_number": "DL941982162",
            "order_value": 350.0,
            "expected_delivery": today - timedelta(days=23),
            "order_date": datetime.combine(today - timedelta(days=27), datetime.min.time()),
            "items": ["Phone Case"],
        },
    ]

    if db.query(Order).filter(Order.user_id == user_id).count() > 0:
        return

    for o in sample_orders:
        order = Order(
            order_id=o["order_id"],
            user_id=user_id,
            status=o["status"],
            carrier=o["carrier"],
            tracking_number=o["tracking_number"],
            order_value=o["order_value"],
            expected_delivery=o["expected_delivery"],
            order_date=o["order_date"],
            items=o["items"],
        )
        db.add(order)

    db.commit()
    logger.info(
        f"Created 5 sample orders for new user: {user_id}",
        extra={"session_id": "unknown", "agent_used": "unknown", "request_id": ""},
    )


# ==========================================
# HELPER — resolve user identity
# ==========================================
def resolve_user(authorization: Optional[str], guest_id: Optional[str], db: Session) -> dict:
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        try:

            id_info = id_token.verify_oauth2_token(token, google_requests.Request(), settings.google_client_id)

            user_id = f"google_{id_info['sub']}"
            email = id_info.get("email")
            name = id_info.get("name")

            user = db.query(User).filter(User.user_id == user_id).first()

            if not user:
                user = User(
                    user_id=user_id,
                    is_authenticated=True,
                    email=email,
                    name=name,
                    auth_provider="google",
                    created_at=datetime.utcnow(),
                    last_login_at=datetime.utcnow(),
                )
                db.add(user)
                db.commit()
                create_sample_orders_for_user(user_id, db)
            else:
                user.last_login_at = datetime.utcnow()

            db.commit()

            return {"user_id": user_id, "is_authenticated": True, "email": email, "name": name}

        except Exception as e:
            logger.warning(
                f"JWT validation failed: {e}",
                extra={"session_id": "unknown", "agent_used": "unknown", "request_id": ""},
            )

    if guest_id:
        user = db.query(User).filter(User.user_id == guest_id).first()

        if not user:
            user = User(
                user_id=guest_id,
                is_authenticated=False,
                auth_provider="anonymous",
                created_at=datetime.utcnow(),
                last_login_at=datetime.utcnow(),
            )
            db.add(user)
            db.commit()

        return {"user_id": guest_id, "is_authenticated": False}

    new_guest_id = str(uuid.uuid4())
    user = User(
        user_id=new_guest_id,
        is_authenticated=False,
        auth_provider="anonymous",
        created_at=datetime.utcnow(),
        last_login_at=datetime.utcnow(),
    )
    db.add(user)
    db.commit()

    return {"user_id": new_guest_id, "is_authenticated": False}


# ==========================================
# HELPER — resolve session
# ==========================================
def resolve_session(session_id: Optional[str], user_id: str, db: Session) -> dict:
    expiry_minutes = settings.session_expiry_minutes
    window_size = settings.history_window_size

    if session_id:
        session = (
            db.query(SessionModel)
            .filter(
                SessionModel.session_id == session_id, SessionModel.user_id == user_id, SessionModel.is_active == True
            )
            .first()
        )

        if session:
            cutoff = datetime.utcnow() - timedelta(minutes=expiry_minutes)
            if session.last_active_at < cutoff:
                session.is_active = False
                session.ended_reason = "expired"
                db.commit()
                logger.info(
                    f"Session expired: {session_id}",
                    extra={"session_id": session_id, "agent_used": "unknown", "request_id": ""},
                )
            else:
                messages = (
                    db.query(Message)
                    .filter(Message.session_id == session_id)
                    .order_by(Message.created_at.desc())
                    .limit(window_size)
                    .all()
                )

                history = [{"role": m.role, "content": m.content} for m in reversed(messages)]

                return {
                    "session_id": session_id,
                    "history": history,
                    "is_new": False,
                    "session_context": session.session_context or {},
                }

    new_session_id = str(uuid.uuid4())
    new_session = SessionModel(
        session_id=new_session_id,
        user_id=user_id,
        created_at=datetime.utcnow(),
        last_active_at=datetime.utcnow(),
        is_active=True,
        message_count=0,
        ended_reason=None,
    )
    db.add(new_session)
    db.commit()

    logger.info(
        f"New session created: {new_session_id}",
        extra={"session_id": new_session_id, "agent_used": "unknown", "request_id": ""},
    )
    return {"session_id": new_session_id, "history": [], "is_new": True, "session_context": {}}


# ==========================================
# HELPER — save messages
# ==========================================
def save_messages(
    session_id: str,
    user_message: str,
    ai_response: str,
    intent: Optional[str],
    confidence: Optional[str],
    agent_name: str,
    latency_ms: int,
    db: Session,
    trace_id: Optional[str] = None,
):
    user_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=user_message,
        intent=intent,
        intent_confidence=confidence,
        created_at=datetime.utcnow(),
    )
    db.add(user_msg)

    ai_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        content=ai_response,
        agent_name=agent_name,
        latency_ms=latency_ms,
        langfuse_trace_id=trace_id,
        created_at=datetime.utcnow(),
    )
    db.add(ai_msg)

    session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()

    if session:
        session.last_active_at = datetime.utcnow()
        session.agent_last_used = agent_name
        session.message_count += 2

    db.commit()
    logger.info(
        f"Messages saved for session: {session_id}",
        extra={"session_id": session_id, "agent_used": agent_name, "request_id": trace_id or ""},
    )


# ==========================================
# HELPER — persist updated session context
# ==========================================
def update_session_context(session_id: str, new_facts: dict, db: Session):
    session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
    if session and new_facts:
        session.session_context = merge_context(session.session_context or {}, new_facts)
        db.commit()


# ==========================================
# POST /chat
# ==========================================
@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
):
    start_time = time.time()
    db = SessionLocal()
    session_id = "unknown"  # pre-init so except block can always reference it
    agent_used = "unknown"  # pre-init so except block uses the resolved value if available

    try:
        # 1. resolve user
        user_info = resolve_user(authorization, body.guest_id, db)
        user_id = user_info["user_id"]
        is_authenticated = user_info["is_authenticated"]

        # 2. resolve session
        session_info = resolve_session(x_session_id, user_id, db)
        session_id = session_info["session_id"]
        history = session_info["history"]
        session_context = session_info.get("session_context", {})

        logger.info(
            f"Request: user={user_id} | session={session_id} | auth={is_authenticated}",
            extra={"session_id": session_id, "agent_used": "unknown", "request_id": ""},
        )

        # 3. create LangFuse trace
        trace = create_trace(
            session_id=session_id, user_id=user_id, is_authenticated=is_authenticated, message=body.message
        )
        set_trace_context(trace.id)

        # 4. run intent router
        result = intent_router_graph.invoke(
            {
                "message": body.message,
                "user_id": user_id,
                "session_id": session_id,
                "history": history,
                "is_authenticated": is_authenticated,
                "session_context": session_context,
                "langfuse_trace_id": trace.id,
                "langfuse_parent_span_id": None,
            }
        )

        response = result.get("response", "")
        agent_used = result.get("agent_used", "unknown")
        intent = result.get("intent")
        confidence = result.get("confidence")
        products = result.get("products", None)
        updated_context = result.get("session_context", {})

        # 5. persist updated session context
        update_session_context(session_id, updated_context, db)

        # 6. score response
        score_response(trace_id=trace.id, agent_used=agent_used, message=body.message, response=response)

        # 7. update trace output + aggregated token/cost totals
        total_input = result.get("total_input_tokens", 0)
        total_output = result.get("total_output_tokens", 0)
        trace.update(
            output={"response": response, "agent_used": agent_used},
            usage_details={"input": total_input, "output": total_output, "total": total_input + total_output},
            cost_details={"total": calculate_cost(total_input, total_output)},
        )

        # 8. flush LangFuse
        flush()

        # 9. calculate latency
        latency_seconds = time.time() - start_time
        latency_ms = int(latency_seconds * 1000)

        # 10. record Prometheus metrics
        record_request_metrics(
            agent_used=agent_used,
            status="success",
            latency_seconds=latency_seconds,
            input_tokens=result.get("total_input_tokens", 0),
            output_tokens=result.get("total_output_tokens", 0),
        )

        # 11. save to database
        save_messages(
            session_id=session_id,
            user_message=body.message,
            ai_response=response,
            intent=intent,
            confidence=confidence,
            agent_name=agent_used,
            latency_ms=latency_ms,
            trace_id=trace.id,
            db=db,
        )

        logger.info(
            f"Response sent: agent={agent_used} | latency={latency_ms}ms",
            extra={"session_id": session_id, "agent_used": agent_used, "request_id": trace.id},
        )

        return ChatResponse(
            response=response,
            session_id=session_id,
            user_id=user_id,
            agent_used=agent_used,
            intent=intent,
            intent_confidence=confidence,
            is_authenticated=is_authenticated,
            products=products,
        )

    except Exception as e:
        latency_seconds = time.time() - start_time
        logger.error(
            f"Chat endpoint error: {e}", extra={"session_id": session_id, "agent_used": agent_used, "request_id": ""}
        )
        record_request_metrics(agent_used=agent_used, status="error", latency_seconds=latency_seconds)
        record_error(error_type=type(e).__name__, agent_used=agent_used)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()


# ==========================================
# POST /session/close
# Marks the caller's session as closed.
# Called when user clicks "Clear Chat".
# ==========================================
@router.post("/session/close")
async def close_session(
    guest_id: Optional[str] = None,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
):
    db = SessionLocal()
    try:
        user_info = resolve_user(authorization, guest_id, db)
        user_id = user_info["user_id"]

        if not x_session_id:
            return {"closed": False}

        session = (
            db.query(SessionModel)
            .filter(
                SessionModel.session_id == x_session_id, SessionModel.user_id == user_id, SessionModel.is_active == True
            )
            .first()
        )

        if not session:
            return {"closed": False}

        session.is_active = False
        session.ended_reason = "user_cleared"
        db.commit()

        logger.info(
            f"Session closed by user: {x_session_id}",
            extra={"session_id": x_session_id, "agent_used": "unknown", "request_id": ""},
        )
        return {"closed": True}
    finally:
        db.close()


# ==========================================
# GET /history
# Returns all messages for the caller's active session.
# Authenticated users: Authorization header.
# Guest users: ?guest_id=<id> query param.
# ==========================================
@router.get("/history", response_model=HistoryResponse)
async def get_history(
    guest_id: Optional[str] = None,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
):
    db = SessionLocal()
    try:
        user_info = resolve_user(authorization, guest_id, db)
        user_id = user_info["user_id"]

        if not x_session_id:
            return HistoryResponse(session_id="", messages=[])

        session = (
            db.query(SessionModel)
            .filter(SessionModel.session_id == x_session_id, SessionModel.user_id == user_id)
            .first()
        )

        if not session:
            return HistoryResponse(session_id=x_session_id, messages=[])

        rows = db.query(Message).filter(Message.session_id == x_session_id).order_by(Message.created_at.asc()).all()

        return HistoryResponse(
            session_id=x_session_id,
            messages=[{"role": m.role, "content": m.content, "agent_name": m.agent_name} for m in rows],
        )
    finally:
        db.close()


# ==========================================
# GET /health
# ==========================================
@router.get("/health", response_model=HealthResponse)
async def health():
    db_status = "connected" if test_connection() else "disconnected"
    return HealthResponse(status="healthy", database=db_status)


# ==========================================
# GET /config
# Serves public client-side config so secrets
# never need to be hardcoded in static HTML.
# ==========================================
@router.get("/config")
async def get_config():
    return {"google_client_id": settings.google_client_id}
