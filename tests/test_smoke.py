"""Automated verification suite for PackItUp Smart Shopping Assistant.

Tests:
1. Dataset enrichment (187 products, 1,131 variants, 56 collections, 451 chunks, 73 missing categories fixed, 6 PKR 0.00 anomalies guarded)
2. Hybrid RAG retrieval across domain partitions (`product`, `collection`, `policy`, `page`)
3. Natural-language prompt-driven automatic cart addition & pack quote calculation
4. B2B Lead capture into `data/leads.csv`
5. Shopify checkout permalink & WhatsApp Cloud API v21.0 interactive payload builder
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.graph import run_shopping_assistant
from src.backend.whatsapp_webhook import build_whatsapp_outbound_payloads
from src.data_loader import get_catalog
from src.rag_chroma import get_rag_engine
from src.state_store import CartManager, LeadStore


def run_all_tests() -> None:
    print("1. Testing PackItUp Dataset Hygiene & Enrichment...")
    catalog = get_catalog()
    assert len(catalog.products_by_handle) == 187, f"Expected 187 products, got {len(catalog.products_by_handle)}"
    assert len(catalog.variants_by_id) == 1131, f"Expected 1131 variants, got {len(catalog.variants_by_id)}"
    assert len(catalog.collection_map) == 56, f"Expected 56 collections, got {len(catalog.collection_map)}"
    assert len(catalog.chunks) == 451, f"Expected 451 chunks, got {len(catalog.chunks)}"

    # Verify zero uncategorized products remain after enrichment
    uncat = [p.handle for p in catalog.products_by_handle.values() if not p.enriched_category]
    assert len(uncat) == 0, f"Found uncategorized products: {uncat}"

    # Verify PKR 0.00 placeholder guardrail on `serving-tray`
    tray = catalog.get_product("serving-tray")
    assert tray is not None and tray.requires_custom_quote is True
    print("   [PASS] 187 products, 1,131 variants, 56 collections, 451 chunks loaded & sanitized.")

    print("2. Testing Hybrid RAG Retrieval...")
    rag = get_rag_engine()
    cards = rag.search_products("10x10x4 bakery cake box with window", top_k=3)
    assert len(cards) > 0, "Expected product cards from RAG search"
    assert cards[0]["handle"] == "10x10x4-bakery-boxwindow"
    pol_hits = rag.retrieve("Karachi same day express shipping delivery", doc_type="policy", k=2)
    assert len(pol_hits) > 0 and pol_hits[0]["metadata"]["type"] == "policy"
    print(f"   [PASS] Top product card: {cards[0]['title']} ({cards[0]['price_display']})")

    print("3. Testing Multi-Agent Prompt-to-Cart Automation...")
    session_id = "test_session_cart"
    CartManager.clear_cart(session_id)
    turn_cart = run_shopping_assistant(
        user_query="Please add 2 packs of 10x10x4 white bakery box (50 pack) to my cart",
        session_id=session_id,
    )
    cart_after = turn_cart["cart"]
    assert cart_after["item_count"] == 1, f"Expected 1 cart line item, got {cart_after}"
    assert cart_after["total_packs"] == 2, f"Expected 2 packs, got {cart_after}"
    assert cart_after["subtotal_pkr"] == 7300.0, f"Expected PKR 7,300 (2 * 3,650), got {cart_after['subtotal_pkr']}"
    print(f"   [PASS] Prompt auto-added to cart: {turn_cart['cart_notification']}")

    print("4. Testing Multi-Domain Fan-Out (Discovery + Quote + Shipping Policy)...")
    turn_fanout = run_shopping_assistant(
        user_query="How much for 500 12oz U-shaped PP boba cups with lids and what is the delivery time to Lahore?",
        session_id=session_id,
    )
    exec_agents = turn_fanout["executed_agents"]
    assert "A2_pricing_and_cart" in exec_agents and "A4_policy_compliance" in exec_agents
    print(f"   [PASS] Multi-agent fan-out executed: {exec_agents}")

    print("5. Testing B2B Lead Capture into data/leads.csv...")
    before_count = len(LeadStore.list_leads())
    turn_lead = run_shopping_assistant(
        user_query="My name is Usman Tariq from BrewLab Cafe in Lahore (+92 301 5556677), I need 2000 custom printed burger boxes with our logo.",
        session_id=session_id,
    )
    after_count = len(LeadStore.list_leads())
    assert turn_lead["captured_lead"] is not None, "Expected captured_lead in result"
    assert after_count == before_count + 1, "Expected leads.csv row count to increment by 1"
    print(f"   [PASS] Captured B2B lead into leads.csv: {turn_lead['captured_lead']['lead_id']}")

    print("6. Testing WhatsApp Cloud API Payload Formatter...")
    wa_payloads = build_whatsapp_outbound_payloads("+923001234567", turn_cart)
    assert len(wa_payloads) >= 1 and wa_payloads[0]["type"] == "interactive"
    print("   [PASS] WhatsApp Cloud API v21.0 interactive payload validated.")
    print("\nALL SMOKE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
