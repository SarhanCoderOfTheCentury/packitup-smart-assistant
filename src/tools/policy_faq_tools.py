"""Policy & Compliance Guardrail Tools (Agent A4) and Order Support Tools (Agent A6)."""
from __future__ import annotations

from typing import Any, Dict
from src import tool
from pydantic import BaseModel, Field

from src.data_loader import get_catalog
from src.rag_chroma import get_rag_engine


class PolicyLookupInput(BaseModel):
    topic: str = Field(
        ...,
        description="Policy topic to query: 'shipping', 'delivery', 'refund', 'return', 'exchange', 'damaged', 'privacy', 'terms', 'payment', or 'contact'.",
    )


@tool("lookup_store_policy", args_schema=PolicyLookupInput)
def lookup_store_policy(topic: str) -> Dict[str, Any]:
    """Queries ONLY the locked `type='policy'` namespace (Agent A4 Guardrail).

    Enforces strict legal compliance:
    - Returns verbatim clauses and `source_url` for Shipping, Refund/Return, and Privacy policies.
    - Explicitly flags Terms of Service and Payment Policy as unpublished (HTTP 404 on live site)
      so the assistant never hallucinates unverified policy rules.
    """
    catalog = get_catalog()
    rag = get_rag_engine()
    hits = rag.retrieve(query=topic, doc_type="policy", k=4)

    # Also match directly against policies.json for 404 markers
    t_low = topic.lower()
    direct_matches = []
    for p in catalog.policies:
        if any(w in p["id"].lower() or w in p["title"].lower() for w in t_low.split()):
            direct_matches.append(p)

    return {
        "topic": topic,
        "policy_chunks": [
            {
                "title": h["metadata"].get("title"),
                "source_url": h["metadata"].get("source_url"),
                "text": h["text"],
            }
            for h in hits
        ],
        "direct_policy_records": direct_matches or catalog.policies,
        "store_contact": catalog.business_facts,
    }


@tool("get_business_facts")
def get_business_facts() -> Dict[str, Any]:
    """Returns official PackItUp store contact details, Karachi addresses, WhatsApp phone, email, social links, and Shopify UCP/MCP endpoint."""
    catalog = get_catalog()
    return {
        "business_facts": catalog.business_facts,
        "head_office_address_privacy_policy": "M22 mezzanine floor, Kohinoor square, Kemball Road, Aram Bagh, Karachi, 74200, PK",
        "warehouse_pickup_address": catalog.business_facts.get("address", ""),
        "whatsapp_and_phone": catalog.business_facts.get("phone", "+92 322 2620169"),
        "email": catalog.business_facts.get("email", "packitup.khi@gmail.com"),
    }


class EscalationInput(BaseModel):
    reason: str = Field(..., description="Reason for escalating to a human specialist.")
    customer_summary: str = Field(..., description="Brief summary of what the customer needs.")


@tool("escalate_to_human", args_schema=EscalationInput)
def escalate_to_human(reason: str, customer_summary: str) -> Dict[str, Any]:
    """First-class human handoff tool for missing policies, damaged-item video claims, or complex B2B negotiations."""
    catalog = get_catalog()
    bf = catalog.business_facts
    return {
        "escalated": True,
        "reason": reason,
        "customer_summary": customer_summary,
        "human_channels": {
            "whatsapp_direct": "https://wa.me/923222620169",
            "phone": bf.get("phone", "+92 322 2620169"),
            "email": bf.get("email", "packitup.khi@gmail.com"),
            "address": bf.get("address", "Nazimabad Block 2, Karachi, Pakistan"),
        },
    }
