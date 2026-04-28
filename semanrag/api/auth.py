"""SemanRAG API authentication — JWT, API key, and optional OIDC support."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from semanrag.api.config import AuthConfig

# Graceful optional imports
try:
    import jwt as pyjwt
except ImportError:
    pyjwt = None  # type: ignore[assignment]

try:
    import bcrypt as _bcrypt
except ImportError:
    _bcrypt = None  # type: ignore[assignment]

_bearer_scheme = HTTPBearer(auto_error=False)


class AuthHandler:
    def __init__(self, config: AuthConfig) -> None:
        self.config = config
        # API keys loaded from env as comma-separated list
        raw = os.environ.get("SEMANRAG_API_KEYS", "")
        self._api_keys: dict[str, dict] = {}
        for key in (k.strip() for k in raw.split(",") if k.strip()):
            self._api_keys[key] = {"sub": "api-key-user", "scope": "full"}

    # ── JWT ───────────────────────────────────────────────────────────

    def create_access_token(
        self, data: dict, expires_delta: timedelta | None = None
    ) -> str:
        if pyjwt is None:
            raise ImportError("PyJWT is required. Install with: pip install PyJWT")
        payload = data.copy()
        expire = datetime.now(UTC) + (
            expires_delta or timedelta(minutes=self.config.access_token_expire_minutes)
        )
        payload["exp"] = expire
        payload["iat"] = datetime.now(UTC)
        return pyjwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)

    def verify_token(self, token: str) -> dict:
        if pyjwt is None:
            raise ImportError("PyJWT is required. Install with: pip install PyJWT")
        try:
            return pyjwt.decode(
                token, self.config.secret_key, algorithms=[self.config.algorithm]
            )
        except pyjwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired") from None
        except pyjwt.InvalidTokenError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {exc}"
            ) from exc

    # ── API key ───────────────────────────────────────────────────────

    def verify_api_key(self, api_key: str) -> dict | None:
        return self._api_keys.get(api_key)

    # ── Password hashing ─────────────────────────────────────────────

    @staticmethod
    def hash_password(password: str) -> str:
        if _bcrypt is None:
            raise ImportError("bcrypt is required. Install with: pip install bcrypt")
        return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        if _bcrypt is None:
            raise ImportError("bcrypt is required. Install with: pip install bcrypt")
        return _bcrypt.checkpw(plain.encode(), hashed.encode())

    # ── OIDC stub ─────────────────────────────────────────────────────

    async def verify_oidc_token(self, token: str) -> dict:
        """Validate an OIDC token against the configured issuer.

        This is a stub — in production, fetch the JWKS from
        ``self.config.oidc_issuer + "/.well-known/openid-configuration"``
        and verify the token signature and claims.
        """
        if not self.config.oidc_issuer:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="OIDC issuer not configured",
            )
        if pyjwt is None:
            raise ImportError("PyJWT is required for OIDC verification")
        try:
            # In production: fetch JWKS, verify audience, issuer, etc.
            payload = pyjwt.decode(token, options={"verify_signature": False})
            if payload.get("iss") != self.config.oidc_issuer:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid OIDC issuer"
                )
            return payload
        except pyjwt.InvalidTokenError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid OIDC token: {exc}"
            ) from exc

    # ── FastAPI dependency ────────────────────────────────────────────

    def get_current_user(self):
        """Return a FastAPI dependency that authenticates via Bearer JWT or API key."""
        handler = self

        async def _dependency(
            request: Request,
            credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
        ) -> dict:
            # Try Bearer token first
            if credentials and credentials.credentials:
                return handler.verify_token(credentials.credentials)

            # Try API key header
            api_key = request.headers.get(handler.config.api_key_header, "")
            if api_key:
                user = handler.verify_api_key(api_key)
                if user is not None:
                    return user
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
                )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication credentials",
            )

        return _dependency
