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
        from datetime import datetime, timedelta, timezone

        mock_session                = MagicMock()
        mock_session.session_id     = "session_abc"
        mock_session.user_id        = "user_123"
        mock_session.is_active      = True
        mock_session.last_active_at = datetime.now(timezone.utc)

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

    def test_resolve_session_expires_stale_session(self, mock_db_session):
        from api.routes import resolve_session
        from datetime import datetime, timedelta, timezone

        mock_session                = MagicMock()
        mock_session.session_id     = "stale_session"
        mock_session.is_active      = True
        # last_active_at is very old — will be expired
        mock_session.last_active_at = datetime.now(timezone.utc) - timedelta(hours=48)

        mock_db_session.query.return_value \
            .filter.return_value.first.return_value = mock_session

        result = resolve_session("stale_session", "user_123", mock_db_session)

        # Should create a new session since the existing one expired
        assert result["is_new"] == True


class TestHelperFunctions:

    def test_create_sample_orders_for_user(self, mock_db_session):
        from api.routes import create_sample_orders_for_user
        mock_db_session.query.return_value.filter.return_value.count.return_value = 0
        create_sample_orders_for_user("user_abc123", mock_db_session)
        assert mock_db_session.add.call_count == 5
        mock_db_session.commit.assert_called()

    def test_create_sample_orders_skips_existing_user(self, mock_db_session):
        from api.routes import create_sample_orders_for_user
        mock_db_session.query.return_value.filter.return_value.count.return_value = 5
        create_sample_orders_for_user("user_abc123", mock_db_session)
        mock_db_session.add.assert_not_called()

    def test_save_messages(self, mock_db_session):
        from api.routes import save_messages

        mock_session = MagicMock()
        mock_session.message_count = 0
        mock_db_session.query.return_value \
            .filter.return_value.first.return_value = mock_session

        save_messages(
            session_id   = "sess_1",
            user_message = "hello",
            ai_response  = "hi there",
            intent       = "greeting",
            confidence   = "high",
            agent_name   = "support_agent",
            latency_ms   = 150,
            db           = mock_db_session,
            trace_id     = "trace_xyz"
        )

        assert mock_db_session.add.call_count == 2
        mock_db_session.commit.assert_called()

    def test_save_messages_without_session(self, mock_db_session):
        from api.routes import save_messages

        mock_db_session.query.return_value \
            .filter.return_value.first.return_value = None

        save_messages(
            session_id   = "sess_missing",
            user_message = "hello",
            ai_response  = "hi",
            intent       = None,
            confidence   = None,
            agent_name   = "product_agent",
            latency_ms   = 100,
            db           = mock_db_session
        )

        assert mock_db_session.add.call_count == 2

    def test_update_session_context(self, mock_db_session):
        from api.routes import update_session_context

        mock_session                 = MagicMock()
        mock_session.session_context = {"existing": "value"}
        mock_db_session.query.return_value \
            .filter.return_value.first.return_value = mock_session

        with patch("api.routes.merge_context", return_value={"existing": "value", "new": "fact"}):
            update_session_context("sess_1", {"new": "fact"}, mock_db_session)

        mock_db_session.commit.assert_called()

    def test_update_session_context_no_new_facts(self, mock_db_session):
        from api.routes import update_session_context

        mock_session = MagicMock()
        mock_db_session.query.return_value \
            .filter.return_value.first.return_value = mock_session

        update_session_context("sess_1", {}, mock_db_session)

        # No commit when new_facts is empty
        mock_db_session.commit.assert_not_called()


class TestAdditionalEndpoints:

    def test_config_endpoint_returns_google_client_id(self, client):
        response = client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert "google_client_id" in data

    def test_session_close_without_session_id(self, client, mock_db_session):
        mock_db_session.query.return_value \
            .filter.return_value.first.return_value = None

        response = client.post("/session/close")
        assert response.status_code == 200
        assert response.json()["closed"] == False

    def test_session_close_when_session_not_found(self, client, mock_db_session):
        mock_db_session.query.return_value \
            .filter.return_value.first.return_value = None

        response = client.post(
            "/session/close",
            headers={"X-Session-ID": "non_existent_session"}
        )
        assert response.status_code == 200
        assert response.json()["closed"] == False

    def test_session_close_when_session_found(self, client, mock_db_session):
        mock_session           = MagicMock()
        mock_session.is_active = True
        # resolve_user with no auth/guest_id creates a new guest without a DB read,
        # so the first DB .first() is the session query itself.
        mock_db_session.query.return_value \
            .filter.return_value.first.return_value = mock_session

        response = client.post(
            "/session/close",
            headers={"X-Session-ID": "active_session_123"}
        )
        assert response.status_code == 200
        assert response.json()["closed"] == True

    def test_history_without_session_id(self, client, mock_db_session):
        mock_db_session.query.return_value \
            .filter.return_value.first.return_value = None

        response = client.get("/history")
        assert response.status_code == 200
        data = response.json()
        assert data["messages"] == []

    def test_history_session_not_found(self, client, mock_db_session):
        mock_db_session.query.return_value \
            .filter.return_value.first.return_value = None

        response = client.get(
            "/history",
            headers={"X-Session-ID": "missing_session"}
        )
        assert response.status_code == 200
        assert response.json()["messages"] == []

    def test_history_returns_messages(self, client, mock_db_session):
        mock_session = MagicMock()
        mock_msg     = MagicMock()
        mock_msg.role       = "user"
        mock_msg.content    = "hello"
        mock_msg.agent_name = None

        mock_db_session.query.return_value \
            .filter.return_value.first.side_effect = [
                None,          # resolve_user
                mock_session   # session found
            ]
        mock_db_session.query.return_value \
            .filter.return_value \
            .order_by.return_value.all.return_value = [mock_msg]

        response = client.get(
            "/history",
            headers={"X-Session-ID": "session_with_messages"}
        )
        assert response.status_code == 200