"""Tests for authentication (JWT, bcrypt, API key)."""
from __future__ import annotations

import os
import time
from datetime import timedelta
from unittest.mock import patch

import pytest

from semanrag.api.auth import AuthHandler
from semanrag.api.config import AuthConfig


@pytest.fixture
def auth_handler():
    config = AuthConfig(
        enabled=True,
        secret_key="test-secret-key-for-unit-tests",
        algorithm="HS256",
        access_token_expire_minutes=30,
        api_key_header="X-API-Key",
    )
    with patch.dict(os.environ, {"SEMANRAG_API_KEYS": "test-key-123,test-key-456"}):
        return AuthHandler(config)


@pytest.mark.unit
class TestAuth:
    def test_jwt_create_and_verify(self, auth_handler):
        """Create a JWT and verify it decodes correctly."""
        token = auth_handler.create_access_token({"sub": "user1", "role": "admin"})
        assert isinstance(token, str)
        payload = auth_handler.verify_token(token)
        assert payload["sub"] == "user1"
        assert payload["role"] == "admin"
        assert "exp" in payload
        assert "iat" in payload

    def test_expired_token_rejected(self, auth_handler):
        """Expired token raises HTTPException."""
        from fastapi import HTTPException

        token = auth_handler.create_access_token(
            {"sub": "user1"}, expires_delta=timedelta(seconds=-1)
        )
        with pytest.raises(HTTPException) as exc_info:
            auth_handler.verify_token(token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_bcrypt_hash_verify(self, auth_handler):
        """Hash and verify password with bcrypt."""
        password = "secure_password_123!"
        hashed = AuthHandler.hash_password(password)
        assert hashed != password
        assert AuthHandler.verify_password(password, hashed)
        assert not AuthHandler.verify_password("wrong_password", hashed)

    def test_api_key_validation(self, auth_handler):
        """Valid API key returns user info, invalid returns None."""
        result = auth_handler.verify_api_key("test-key-123")
        assert result is not None
        assert result["sub"] == "api-key-user"

        result_invalid = auth_handler.verify_api_key("invalid-key")
        assert result_invalid is None
