"""Pydantic v2 API Request & Response Schemas for FastAPI Backend."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Customer natural language message.")
    session_id: str = Field(default="default", description="Unique customer or thread session ID.")
    channel: str = Field(default="fastapi", description="Originating channel ('streamlit', 'fastapi', 'whatsapp').")
    category_filter: Optional[str] = Field(default=None, description="Optional category filter.")
    max_price_pkr: Optional[float] = Field(default=None, description="Optional max pack price in PKR.")
    in_stock_only: bool = Field(default=False, description="Only return in-stock products.")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    router_decision: Optional[Dict[str, Any]] = None
    executed_agents: List[str] = Field(default_factory=list)
    product_cards: List[Dict[str, Any]] = Field(default_factory=list)
    cart: Dict[str, Any] = Field(default_factory=dict)
    cart_notification: Optional[str] = None
    captured_lead: Optional[Dict[str, Any]] = None
    citations: List[str] = Field(default_factory=list)
    suggested_prompts: List[str] = Field(default_factory=list)


class CartAddRequest(BaseModel):
    session_id: str = Field(default="default")
    product_identifier: str = Field(..., description="Product handle, title, or numeric ID.")
    variant_id: Optional[int] = Field(default=None, description="Optional Shopify variant ID.")
    quantity_packs: int = Field(default=1, ge=1)
    pack_size_hint: Optional[str] = Field(default=None)
    color_hint: Optional[str] = Field(default=None)


class CartUpdateRequest(BaseModel):
    session_id: str = Field(default="default")
    variant_id: int
    quantity_packs: int = Field(..., ge=0)


class LeadCreateRequest(BaseModel):
    customer_name: str
    phone: str
    email: str = ""
    company_name: str = ""
    city: str = "Karachi"
    inquiry_type: str = "custom_printing"
    product_interest: str
    estimated_quantity: str = "500-1000 pcs"
    customization_details: str = ""
    budget_pkr: str = ""
    source_channel: str = "api"
    notes: str = ""
