# api/routes.py

import uuid
import time
import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, Header, HTTPException
from typing import Optional
from sqlalchemy.orm import Session

from langfuse_helpers.tracing import create_trace, flush
from langfuse_helpers.scoring import score_response
from monitoring.metrics import record_request_metrics, record_error

from api.schemas import ChatRequest, ChatResponse, HealthResponse
from agents.intent_router import intent_router_graph
from database.connection import SessionLocal, test_connection
from database.models import User, Session as SessionModel, Message
from config.settings import settings
from fastapi import BackgroundTasks
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from langfuse_helpers.tracing import get_trace_token_usage
from monitoring.metrics import TOKEN_COUNTER

logger = logging.getLogger(__name__)

router = APIRouter()


# ==========================================
# HELPER — resolve user identity
# ==========================================
def create_sample_orders_for_user(user_id: str, db: Session):
    

    sample_orders = [
        {
            "order_id":          f"ORD-{user_id[-6:].upper()}-001",
            "status":            "delivered",
            "carrier":           "DTDC",
            "tracking_number":   "DT251092059",
            "order_value":       1299.0,
            "expected_delivery": "2026-05-10",
            "order_date":        "2026-05-01",
            "items":             "Wireless Headphones"
        },
        {
            "order_id":          f"ORD-{user_id[-6:].upper()}-002",
            "status":            "shipped",
            "carrier":           "BlueDart",
            "tracking_number":   "BD123456789",
            "order_value":       2500.0,
            "expected_delivery": "2026-05-28",
            "order_date":        "2026-05-22",
            "items":             "Laptop Stand"
        },
        {
            "order_id":          f"ORD-{user_id[-6:].upper()}-003",
            "status":            "processing",
            "carrier":           "FedEx",
            "tracking_number":   "FX987654321",
            "order_value":       499.0,
            "expected_delivery": "2026-05-30",
            "order_date":        "2026-05-23",
            "items":             "USB Hub"
        },
        {
            "order_id":          f"ORD-{user_id[-6:].upper()}-004",
            "status":            "delayed",
            "carrier":           "Ekart",
            "tracking_number":   "EK112233445",
            "order_value":       899.0,
            "expected_delivery": "2026-05-20",
            "order_date":        "2026-05-15",
            "items":             "Mechanical Keyboard"
        },
        {
            "order_id":          f"ORD-{user_id[-6:].upper()}-005",
            "status":            "delivered",
            "carrier":           "DHL",
            "tracking_number":   "DL941982162",
            "order_value":       350.0,
            "expected_delivery": "2026-05-05",
            "order_date":        "2026-05-01",
            "items":             "Phone Case"
        }
    ]

    for o in sample_orders:
        order = Order(
            order_id          = o["order_id"],
            user_id           = user_id,
            status            = o["status"],
            carrier           = o["carrier"],
            tracking_number   = o["tracking_number"],
            order_value       = o["order_value"],
            expected_delivery = o["expected_delivery"],
            order_date        = o["order_date"],
            items             = o["items"]
        )
        db.add(order)

    db.commit()
    logger.info(
        f"Created 5 sample orders for new user: {user_id}",
        extra={"session_id": "unknown", "agent_used": "unknown", "request_id": ""}
    )


def resolve_user(
    authorization: Optional[str],
    guest_id:      Optional[str],
    db:            Session
) -> dict:
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")
        try:
            
            id_info = id_token.verify_oauth2_token(
                token,
                google_requests.Request(),
                settings.google_client_id
            )

            user_id = f"google_{id_info['sub']}"
            email   = id_info.get("email")
            name    = id_info.get("name")

            user = db.query(User).filter(
                User.user_id == user_id
            ).first()

            if not user:
                user = User(
                    user_id          = user_id,
                    is_authenticated = True,
                    email            = email,
                    name             = name,
                    auth_provider    = "google",
                    created_at       = datetime.utcnow(),
                    last_login_at    = datetime.utcnow()
                )
                db.add(user)
                db.commit()
                create_sample_orders_for_user(user_id, db)
            else:
                user.last_login_at = datetime.utcnow()

            db.commit()

            return {
                "user_id":          user_id,
                "is_authenticated": True,
                "email":            email,
                "name":             name
            }

        except Exception as e:
            logger.warning(
                f"JWT validation failed: {e}",
                extra={"session_id": "unknown", "agent_used": "unknown", "request_id": ""}
            )

    if guest_id:
        user = db.query(User).filter(
            User.user_id == guest_id
        ).first()

        if not user:
            user = User(
                user_id          = guest_id,
                is_authenticated = False,
                auth_provider    = "anonymous",
                created_at       = datetime.utcnow(),
                last_login_at    = datetime.utcnow()
            )
            db.add(user)
            db.commit()

        return {
            "user_id":          guest_id,
            "is_authenticated": False
        }

    new_guest_id = str(uuid.uuid4())
    user = User(
        user_id          = new_guest_id,
        is_authenticated = False,
        auth_provider    = "anonymous",
        created_at       = datetime.utcnow(),
        last_login_at    = datetime.utcnow()
    )
    db.add(user)
    db.commit()

    return {
        "user_id":          new_guest_id,
        "is_authenticated": False
    }


# ==========================================
# HELPER — resolve session
# ==========================================
def resolve_session(
    session_id: Optional[str],
    user_id:    str,
    db:         Session
) -> dict:
    expiry_minutes = settings.session_expiry_minutes
    window_size    = settings.history_window_size

    if session_id:
        session = db.query(SessionModel).filter(
            SessionModel.session_id == session_id,
            SessionModel.user_id    == user_id,
            SessionModel.is_active  == True
        ).first()

        if session:
            cutoff = datetime.utcnow() - timedelta(minutes=expiry_minutes)
            if session.last_active_at < cutoff:
                session.is_active    = False
                session.ended_reason = "expired"
                db.commit()
                logger.info(
                    f"Session expired: {session_id}",
                    extra={"session_id": session_id, "agent_used": "unknown", "request_id": ""}
                )
            else:
                messages = db.query(Message).filter(
                    Message.session_id == session_id
                ).order_by(
                    Message.created_at.desc()
                ).limit(window_size).all()

                history = [
                    {"role": m.role, "content": m.content}
                    for m in reversed(messages)
                ]

                return {
                    "session_id": session_id,
                    "history":    history,
                    "is_new":     False
                }

    new_session_id = str(uuid.uuid4())
    new_session    = SessionModel(
        session_id     = new_session_id,
        user_id        = user_id,
        created_at     = datetime.utcnow(),
        last_active_at = datetime.utcnow(),
        is_active      = True,
        message_count  = 0,
        ended_reason   = None
    )
    db.add(new_session)
    db.commit()

    logger.info(
        f"New session created: {new_session_id}",
        extra={"session_id": new_session_id, "agent_used": "unknown", "request_id": ""}
    )
    return {
        "session_id": new_session_id,
        "history":    [],
        "is_new":     True
    }


# ==========================================
# HELPER — save messages
# ==========================================
def save_messages(
    session_id:   str,
    user_message: str,
    ai_response:  str,
    intent:       Optional[str],
    confidence:   Optional[str],
    agent_name:   str,
    latency_ms:   int,
    db:           Session,
    trace_id:     Optional[str] = None
):
    user_msg = Message(
        message_id        = str(uuid.uuid4()),
        session_id        = session_id,
        role              = "user",
        content           = user_message,
        intent            = intent,
        intent_confidence = confidence,
        created_at        = datetime.utcnow()
    )
    db.add(user_msg)

    ai_msg = Message(
        message_id        = str(uuid.uuid4()),
        session_id        = session_id,
        role              = "assistant",
        content           = ai_response,
        agent_name        = agent_name,
        latency_ms        = latency_ms,
        langfuse_trace_id = trace_id,
        created_at        = datetime.utcnow()
    )
    db.add(ai_msg)

    session = db.query(SessionModel).filter(
        SessionModel.session_id == session_id
    ).first()

    if session:
        session.last_active_at  = datetime.utcnow()
        session.agent_last_used = agent_name
        session.message_count  += 2

    db.commit()
    logger.info(
        f"Messages saved for session: {session_id}",
        extra={"session_id": session_id, "agent_used": agent_name, "request_id": trace_id or ""}
    )


# ==========================================
# POST /chat
# ==========================================
@router.post("/chat", response_model=ChatResponse)
async def chat(
    request:          ChatRequest,
    background_tasks: BackgroundTasks,
    authorization:    Optional[str] = Header(
        default=None, alias="Authorization"
    ),
    x_session_id:     Optional[str] = Header(
        default=None, alias="X-Session-ID"
    )
):
    start_time = time.time()
    db         = SessionLocal()
    session_id = "unknown"          # pre-init so except block can always reference it

    try:
        # 1. resolve user
        user_info = resolve_user(authorization, request.guest_id, db)
        user_id          = user_info["user_id"]
        is_authenticated = user_info["is_authenticated"]

        # 2. resolve session
        session_info = resolve_session(x_session_id, user_id, db)
        session_id   = session_info["session_id"]
        history      = session_info["history"]

        logger.info(
            f"Request: user={user_id} | session={session_id} | auth={is_authenticated}",
            extra={
                "session_id": session_id,
                "agent_used": "unknown",
                "request_id": ""
            }
        )

        # 3. create LangFuse trace
        trace = create_trace(
            session_id       = session_id,
            user_id          = user_id,
            is_authenticated = is_authenticated,
            message          = request.message
        )

        # 4. run intent router
        result = intent_router_graph.invoke({
            "message":                 request.message,
            "user_id":                 user_id,
            "session_id":              session_id,
            "history":                 history,
            "is_authenticated":        is_authenticated,
            "langfuse_trace_id":       trace.id,
            "langfuse_parent_span_id": None
        })

        response   = result.get("response", "")
        agent_used = result.get("agent_used", "unknown")
        intent     = result.get("intent")
        confidence = result.get("confidence")

        # 5. score response
        score_response(
            trace_id   = trace.id,
            agent_used = agent_used,
            message    = request.message,
            response   = response
        )

        # 6. update trace output
        trace.update(output={"response": response, "agent_used": agent_used})

        # 7. flush LangFuse
        flush()

        # 8. calculate latency
        latency_seconds = time.time() - start_time
        latency_ms      = int(latency_seconds * 1000)

        # 9. record Prometheus metrics
        record_request_metrics(
            agent_used      = agent_used,
            status          = "success",
            latency_seconds = latency_seconds
        )

        # 10. record token usage in background — non-blocking
        def record_tokens_background(tid: str, aused: str):
            import time as _time
            _time.sleep(3)
            usage = get_trace_token_usage(tid)
            if usage["input"] > 0:
                TOKEN_COUNTER.labels(agent_used=aused, token_type="input").inc(usage["input"])
                TOKEN_COUNTER.labels(agent_used=aused, token_type="output").inc(usage["output"])

        background_tasks.add_task(record_tokens_background, trace.id, agent_used)

        # 11. save to database
        save_messages(
            session_id   = session_id,
            user_message = request.message,
            ai_response  = response,
            intent       = intent,
            confidence   = confidence,
            agent_name   = agent_used,
            latency_ms   = latency_ms,
            trace_id     = trace.id,
            db           = db
        )

        logger.info(
            f"Response sent: agent={agent_used} | latency={latency_ms}ms",
            extra={
                "session_id": session_id,
                "agent_used": agent_used,
                "request_id": trace.id
            }
        )

        return ChatResponse(
            response          = response,
            session_id        = session_id,
            user_id           = user_id,
            agent_used        = agent_used,
            intent            = intent,
            intent_confidence = confidence,
            is_authenticated  = is_authenticated
        )

    except Exception as e:
        latency_seconds = time.time() - start_time
        logger.error(
            f"Chat endpoint error: {e}",
            extra={
                "session_id": session_id,
                "agent_used": "unknown",
                "request_id": ""
            }
        )
        record_error(error_type=type(e).__name__, agent_used="unknown")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()


# ==========================================
# GET /health
# ==========================================
@router.get("/health", response_model=HealthResponse)
async def health():
    db_status = "connected" if test_connection() else "disconnected"
    return HealthResponse(
        status   = "healthy",
        database = db_status
    )