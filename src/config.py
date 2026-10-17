"""Centralized configuration for PackItUp Smart Shopping Assistant.

Supports loading from environment variables (.env) as well as Streamlit Cloud
`st.secrets` when running directly on Streamlit Community Cloud.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get_secret_or_env(key: str, default: str = "") -> str:
    """Reads from environment variable first, then falls back to st.secrets."""
    val = os.getenv(key)
    if val is not None and str(val).strip() != "":
        return str(val).strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "packitup"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
DEFAULT_LEADS_CSV = PROJECT_ROOT / "data" / "leads.csv"


@dataclass
class Settings:
    """Application-wide configuration settings."""

    google_api_key: str = field(
        default_factory=lambda: _get_secret_or_env("GOOGLE_API_KEY", "") or _get_secret_or_env("GEMINI_API_KEY", "")
    )
    gemini_model: str = field(default_factory=lambda: _get_secret_or_env("GEMINI_MODEL", "gemini-3.1-flash-lite"))
    gemini_embedding_model: str = field(
        default_factory=lambda: _get_secret_or_env("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    )
    gemini_temperature: float = field(
        default_factory=lambda: float(_get_secret_or_env("GEMINI_TEMPERATURE", "0.1"))
    )

    # Paths & Storage
    data_dir: Path = field(
        default_factory=lambda: Path(_get_secret_or_env("PACKITUP_DATA_DIR", str(DEFAULT_DATA_DIR)))
    )
    chroma_persist_dir: Path = field(
        default_factory=lambda: Path(_get_secret_or_env("CHROMA_PERSIST_DIR", str(DEFAULT_CHROMA_DIR)))
    )
    chroma_collection_name: str = field(
        default_factory=lambda: _get_secret_or_env("CHROMA_COLLECTION_NAME", "packitup_knowledge")
    )
    leads_csv_path: Path = field(
        default_factory=lambda: Path(_get_secret_or_env("LEADS_CSV_PATH", str(DEFAULT_LEADS_CSV)))
    )

    # Execution & Backend Mode ("direct" runs LangGraph in-process; "fastapi" calls FastAPI server)
    agent_execution_mode: str = field(
        default_factory=lambda: _get_secret_or_env("AGENT_EXECUTION_MODE", "direct")
    )
    fastapi_url: str = field(
        default_factory=lambda: _get_secret_or_env("FASTAPI_URL", "http://localhost:8000")
    )

    # Shopify Storefront / Admin / MCP Configuration (for live cutover)
    shopify_store_domain: str = field(
        default_factory=lambda: _get_secret_or_env("SHOPIFY_STORE_DOMAIN", "packitup.com.pk")
    )
    shopify_storefront_access_token: str = field(
        default_factory=lambda: _get_secret_or_env("SHOPIFY_STOREFRONT_ACCESS_TOKEN", "")
    )
    shopify_admin_access_token: str = field(
        default_factory=lambda: _get_secret_or_env("SHOPIFY_ADMIN_ACCESS_TOKEN", "")
    )
    shopify_api_version: str = field(
        default_factory=lambda: _get_secret_or_env("SHOPIFY_API_VERSION", "2026-01")
    )
    shopify_mcp_endpoint: str = field(
        default_factory=lambda: _get_secret_or_env(
            "SHOPIFY_MCP_ENDPOINT", "https://packitup.com.pk/api/ucp/mcp"
        )
    )

    # WhatsApp Business Cloud API Configuration
    whatsapp_verify_token: str = field(
        default_factory=lambda: _get_secret_or_env("WHATSAPP_VERIFY_TOKEN", "packitup_verify_secret_2026")
    )
    whatsapp_access_token: str = field(
        default_factory=lambda: _get_secret_or_env("WHATSAPP_ACCESS_TOKEN", "")
    )
    whatsapp_phone_number_id: str = field(
        default_factory=lambda: _get_secret_or_env("WHATSAPP_PHONE_NUMBER_ID", "")
    )
    whatsapp_app_secret: str = field(
        default_factory=lambda: _get_secret_or_env("WHATSAPP_APP_SECRET", "")
    )
    whatsapp_sales_alert_phone: str = field(
        default_factory=lambda: _get_secret_or_env("WHATSAPP_SALES_ALERT_PHONE", "+923222620169")
    )

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.google_api_key and not self.google_api_key.startswith("YOUR_"))

    @property
    def has_shopify_token(self) -> bool:
        return bool(
            self.shopify_storefront_access_token
            and not self.shopify_storefront_access_token.startswith("YOUR_")
        )


def get_settings() -> Settings:
    """Returns a fresh Settings instance reflecting current environment / secrets."""
    return Settings()
