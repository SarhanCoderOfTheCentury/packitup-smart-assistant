"""LangGraph State & Structured Routing Schemas for PackItUp Multi-Agent Architecture."""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from src import BaseMessage, add_messages
from pydantic import BaseModel, Field


AgentName = Literal[
    "A1_product_discovery",
    "A2_pricing_and_cart",
    "A3_collection_navigator",
    "A4_policy_compliance",
    "A5_b2b_custom_lead",
    "A6_order_support",
]


class ExtractedEntities(BaseModel):
    """Structured entities extracted by A0 Orchestrator from the user's natural language message."""

    search_query: str = Field(default="", description="Core product search query or packaging need.")
    product_identifier: Optional[str] = Field(
        default=None, description="Specific product handle or title mentioned by the user."
    )
    category_or_collection: Optional[str] = Field(
        default=None, description="Category or collection name if browsing (e.g. 'bakery', 'burger', 'pouches')."
    )
    pack_size_hint: Optional[str] = Field(
        default=None, description="Pack MOQ size mentioned (e.g. '20', '50', '100', '200')."
    )
    color_or_size_hint: Optional[str] = Field(
        default=None, description="Color or dimension/volume option (e.g. 'White', 'Brown', '12oz', '10x10x4')."
    )
    quantity_packs: int = Field(
        default=1, ge=1, description="Number of packs to add to cart or quote."
    )
    desired_pieces: Optional[int] = Field(
        default=None, description="Total pieces requested when user asks for piece count (e.g. '500 cups')."
    )
    max_price_pkr: Optional[float] = Field(
        default=None, description="Maximum budget in PKR if specified."
    )
    cart_action: Optional[Literal["add", "remove", "view", "clear", "checkout"]] = Field(
        default=None, description="Cart operation requested by the user in natural language."
    )
    policy_topic: Optional[str] = Field(
        default=None, description="Policy topic if asking about shipping, refund, return, privacy, payment, or terms."
    )
    # Lead capture fields
    customer_name: Optional[str] = Field(default=None)
    phone: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)
    company_name: Optional[str] = Field(default=None)
    city: Optional[str] = Field(default=None)
    customization_details: Optional[str] = Field(default=None)


class RouterDecision(BaseModel):
    """Output schema for A0 Supervisor / Router Node."""

    primary_intent: Literal[
        "product_lookup",
        "price_quote",
        "cart_operation",
        "browse_category",
        "policy_question",
        "custom_order_lead",
        "order_status_or_checkout",
        "multi_domain",
        "consultative_advisory",
    ]
    target_agents: List[AgentName] = Field(
        ...,
        description="Ordered list of specialist agents (A1–A6) to invoke for this turn.",
    )
    reasoning: str = Field(..., description="Brief explanation of why these agents were selected.")
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)


class AgentState(TypedDict):
    """Shared state flowing through the LangGraph StateGraph."""

    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    user_query: str
    channel: str  # "streamlit", "fastapi", "whatsapp"
    active_category_filter: Optional[str]
    active_max_price_pkr: Optional[float]
    in_stock_only: bool

    # Routing & Agent Execution
    router_decision: Optional[Dict[str, Any]]
    pending_agents: List[str]
    executed_agents: List[str]
    specialist_outputs: List[Dict[str, Any]]

    # Rich UI Artifacts & Mutations
    product_cards: List[Dict[str, Any]]
    cart_state: Dict[str, Any]
    cart_notification: Optional[str]
    captured_lead: Optional[Dict[str, Any]]
    citations: List[str]
    suggested_prompts: List[str]
    final_response: str
