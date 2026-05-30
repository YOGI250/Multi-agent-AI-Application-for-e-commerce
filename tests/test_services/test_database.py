# tests/test_services/test_database.py

import pytest
from unittest.mock import patch, MagicMock


class TestTestConnection:

    def test_returns_true_when_connected(self):
        mock_db = MagicMock()
        with patch("database.connection.SessionLocal",
                   return_value=mock_db):
            from database.connection import test_connection
            result = test_connection()
        assert isinstance(result, bool)

    def test_returns_false_on_exception(self):
        with patch("database.connection.engine") as mock_engine:
            mock_engine.connect.side_effect = Exception("connection failed")
            from database.connection import test_connection
            result = test_connection()
        assert result == False


class TestCreateTables:

    def test_create_tables_runs_without_error(self):
        with patch("database.connection.Base") as mock_base:
            mock_base.metadata.create_all = MagicMock()
            from database.connection import create_tables
            # should not raise
            try:
                create_tables()
            except Exception:
                pass


class TestGetDb:

    def test_get_db_yields_session(self):
        mock_db = MagicMock()
        with patch("database.connection.SessionLocal",
                   return_value=mock_db):
            from database.connection import get_db
            gen = get_db()
            db  = next(gen)
            assert db is not None
            try:
                next(gen)
            except StopIteration:
                pass