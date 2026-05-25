# tests/test_api/test_routes.py

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture
def client():
    from main import app
    return TestClient(app)

@pytest.fixture
def mock_db_session():
    with patch("api.routes.SessionLocal") as mock_session:
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        yield mock_db

@pytest.fixture
def mock_agent_result():
    return {
        "response":   "Here are your results.",
        "agent_used": "product_agent",
        "intent":     "product_query",
        "confidence": "high"
    }


# ==========================================
# TESTS — health endpoint
# ==========================================
class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        with patch("api.routes.test_connection", return_value=True):
            response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_healthy(self, client):
        with patch("api.routes.test_connection", return_value=True):
            response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_returns_database_connected(self, client):
        with patch("api.routes.test_connection", return_value=True):
            response = client.get("/health")
        data = response.json()
        assert data["database"] == "connected"

    def test_health_returns_database_disconnected(self, client):
        with patch("api.routes.test_connection", return_value=False):
            response = client.get("/health")
        data = response.json()
        assert data["database"] == "disconnected"


# ==========================================
# TESTS — chat endpoint
# ==========================================
class TestChatEndpoint:

    def test_chat_returns_200_for_guest(
        self, client, mock_db_session, mock_agent_result
    ):
        mock_user    = MagicMock()
        mock_session = MagicMock()
        mock_session.is_active      = True
        mock_session.last_active_at = MagicMock()

        mock_db_session.query.return_value \
            .filter.return_value.first.side_effect = [
                None,         # user not found → new guest
                None,         # session not found → new session
                mock_session  # session update
            ]

        with patch("api.routes.intent_router_graph") as mock_graph, \
             patch("api.routes.create_trace")         as mock_trace, \
             patch("api.routes.score_response"), \
             patch("api.routes.flush"), \
             patch("api.routes.record_request_metrics"), \
             patch("api.routes.save_messages"):

            mock_trace.return_value = MagicMock(id="trace_123")
            mock_graph.invoke.return_value = mock_agent_result

            response = client.post(
                "/chat",
                json={"message": "show me laptops"}
            )

        assert response.status_code == 200

    def test_chat_returns_response_field(
        self, client, mock_db_session, mock_agent_result
    ):
        mock_session             = MagicMock()
        mock_session.is_active   = True
        mock_session.last_active_at = MagicMock()

        mock_db_session.query.return_value \
            .filter.return_value.first.side_effect = [
                None, None, mock_session
            ]

        with patch("api.routes.intent_router_graph") as mock_graph, \
             patch("api.routes.create_trace")         as mock_trace, \
             patch("api.routes.score_response"), \
             patch("api.routes.flush"), \
             patch("api.routes.record_request_metrics"), \
             patch("api.routes.save_messages"):

            mock_trace.return_value = MagicMock(id="trace_123")
            mock_graph.invoke.return_value = mock_agent_result

            response = client.post(
                "/chat",
                json={"message": "show me laptops"}
            )

        data = response.json()
        assert "response"   in data
        assert "session_id" in data
        assert "agent_used" in data

    def test_chat_requires_message_field(self, client):
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_chat_rejects_empty_message(self, client):
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422


# ==========================================
# TESTS — session management
# ==========================================
class TestSessionManagement:

    def test_resolve_user_creates_guest(self, mock_db_session):
        from api.routes import resolve_user

        mock_db_session.query.return_value \
            .filter.return_value.first.return_value = None

        result = resolve_user(None, None, mock_db_session)

        assert result["is_authenticated"] == False
        assert result["user_id"] is not None

    def test_resolve_user_returns_existing_guest(self, mock_db_session):
        from api.routes import resolve_user

        mock_user         = MagicMock()
        mock_user.user_id = "guest_123"
        mock_db_session.query.return_value \
            .filter.return_value.first.return_value = mock_user

        result = resolve_user(None, "guest_123", mock_db_session)

        assert result["user_id"]          == "guest_123"
        assert result["is_authenticated"] == False

    def test_resolve_session_creates_new_session(self, mock_db_session):
        from api.routes import resolve_session

        mock_db_session.query.return_value \
            .filter.return_value.first.return_value = None

        result = resolve_session(None, "user_123", mock_db_session)

        assert result["is_new"]     == True
        assert result["session_id"] is not None
        assert result["history"]    == []

    def test_resolve_session_loads_existing_session(self, mock_db_session):
        from api.routes import resolve_session
        from datetime import datetime, timedelta

        mock_session                = MagicMock()
        mock_session.session_id     = "session_abc"
        mock_session.user_id        = "user_123"
        mock_session.is_active      = True
        mock_session.last_active_at = datetime.utcnow()

        mock_db_session.query.return_value \
            .filter.return_value.first.return_value = mock_session

        mock_db_session.query.return_value \
            .filter.return_value \
            .order_by.return_value \
            .limit.return_value.all.return_value = []

        result = resolve_session(
            "session_abc", "user_123", mock_db_session
        )

        assert result["session_id"] == "session_abc"
        assert result["is_new"]     == False