# database/models.py

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Text, Numeric, Date, ForeignKey, Integer, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from database.connection import Base


def generate_uuid():
    return str(uuid.uuid4())


# ==========================================
# SYSTEM TABLE 1 — users
# ==========================================
class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, default=generate_uuid)
    is_authenticated = Column(Boolean, default=False, nullable=False)
    email = Column(String, nullable=True)
    name = Column(String, nullable=True)
    auth_provider = Column(String, default="anonymous", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata_ = Column("metadata", JSONB, nullable=True, default=dict)

    sessions = relationship("Session", back_populates="user")

    def __repr__(self):
        return f"<User user_id={self.user_id} auth={self.is_authenticated}>"


# ==========================================
# SYSTEM TABLE 2 — sessions
# ==========================================
class Session(Base):
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_active_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    agent_last_used = Column(String, nullable=True)
    message_count = Column(Integer, default=0, nullable=False)
    ended_reason = Column(String, default=None, nullable=True)
    session_context = Column(JSONB, nullable=True, default=dict)

    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session")

    __table_args__ = (
        Index("idx_sessions_user_id", "user_id"),
        Index("idx_sessions_user_active", "user_id", "is_active"),
    )

    def __repr__(self):
        return f"<Session session_id={self.session_id} active={self.is_active}>"


# ==========================================
# SYSTEM TABLE 3 — messages
# ==========================================
class Message(Base):
    __tablename__ = "messages"

    message_id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("sessions.session_id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    agent_name = Column(String, nullable=True)
    intent = Column(String, nullable=True)
    intent_confidence = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    token_usage = Column(JSONB, nullable=True, default={})
    latency_ms = Column(Integer, nullable=True)
    langfuse_trace_id = Column(String, nullable=True)
    error = Column(Text, nullable=True)

    session = relationship("Session", back_populates="messages")

    def __repr__(self):
        return f"<Message message_id={self.message_id} role={self.role}>"


# ==========================================
# DOMAIN TABLE 1 — orders
# ==========================================
class Order(Base):
    __tablename__ = "orders"

    order_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    status = Column(String, nullable=False)
    items = Column(JSONB, nullable=False, default=list)
    carrier = Column(String, nullable=True)
    tracking_number = Column(String, nullable=True)
    order_date = Column(DateTime, nullable=False)
    expected_delivery = Column(Date, nullable=True)
    order_value = Column(Numeric(10, 2), nullable=False)

    __table_args__ = (Index("idx_orders_user_id", "user_id"),)

    def __repr__(self):
        return f"<Order order_id={self.order_id} status={self.status}>"


# ==========================================
# DOMAIN TABLE 2 — products
# ==========================================
class Product(Base):
    __tablename__ = "products"

    product_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    price = Column(Numeric(10, 2), nullable=False)
    actual_price = Column(Numeric(10, 2), nullable=True)
    discount_percent = Column(Numeric(5, 2), nullable=True)
    brand = Column(String, nullable=True)
    rating = Column(Numeric(3, 1), nullable=True)
    rating_count = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    features = Column(JSONB, nullable=True, default=list)
    in_stock = Column(Boolean, default=True, nullable=False)
    product_type = Column(String, nullable=True)

    def __repr__(self):
        return f"<Product name={self.name} category={self.category}>"


# ==========================================
# DOMAIN TABLE 3 — policies
# ==========================================
class Policy(Base):
    __tablename__ = "policies"

    policy_id = Column(String, primary_key=True, default=generate_uuid)
    issue_type = Column(String, nullable=False, unique=True)
    policy_text = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Policy issue_type={self.issue_type}>"


# ==========================================
# DOMAIN TABLE 4 — carrier_tracking
# ==========================================
class CarrierTracking(Base):
    __tablename__ = "carrier_tracking"

    tracking_id = Column(String, primary_key=True, default=generate_uuid)
    tracking_number = Column(String, nullable=False, unique=True)
    carrier_name = Column(String, nullable=False)
    current_status = Column(String, nullable=False)
    current_location = Column(String, nullable=True)
    events = Column(JSONB, nullable=False, default=list)
    estimated_delivery = Column(Date, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<CarrierTracking tracking_number={self.tracking_number}>"


# ==========================================
# RUNTIME TABLE 1 — support_tickets
# ==========================================
class SupportTicket(Base):
    __tablename__ = "support_tickets"

    ticket_id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    order_id = Column(String, ForeignKey("orders.order_id"), nullable=True)
    issue_type = Column(String, nullable=False)
    priority = Column(String, nullable=False)
    status = Column(String, default="open", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("idx_tickets_user_id", "user_id"),)

    def __repr__(self):
        return f"<SupportTicket ticket_id={self.ticket_id} priority={self.priority}>"


# ==========================================
# RUNTIME TABLE 2 — agent_runs
# ==========================================
class AgentRun(Base):
    __tablename__ = "agent_runs"

    run_id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, nullable=False)
    message_id = Column(String, nullable=True)
    agent_name = Column(String, nullable=False)
    intent = Column(String, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    success = Column(Boolean, default=True, nullable=False)
    total_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Numeric(10, 6), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<AgentRun agent={self.agent_name} duration={self.duration_ms}ms>"
