"""Multi-Agent LangGraph Orchestration for PackItUp Smart Shopping Assistant.

Implements the complete A0 -> (A1..A6) -> Synthesizer `StateGraph` using `langgraph`
and `ChatGoogleGenerativeAI` (`gemini-2.5-flash`). Includes a seamless deterministic
rule+RAG intelligence layer so that every node functions reliably both with a live
`GOOGLE_API_KEY` and in offline/CI test environments.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from src import AIMessage, END, HumanMessage, START, StateGraph, SystemMessage

from src.agents.prompts import (
    A0_SUPERVISOR_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)
from src.agents.state import AgentState, ExtractedEntities, RouterDecision
from src.config import get_settings
from src.data_loader import get_catalog
from src.rag_chroma import get_rag_engine
from src.state_store import CartManager, LeadStore
from src.tools.shopify_adapter import ShopifyStorefrontClient


def _get_llm():
    """Returns a configured ChatGoogleGenerativeAI instance if GOOGLE_API_KEY is available."""
    settings = get_settings()
    if not settings.has_gemini_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=settings.gemini_temperature,
            timeout=25.0,
            max_retries=1,
        )
    except Exception:
        return None


def _clean_llm_response(content: Any) -> str:
    """Extracts clean markdown text from strings, list of dicts, or serialized langchain chunks."""
    if not content:
        return ""
    if hasattr(content, "content"):
        content = content.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif "content" in block:
                    parts.append(str(block["content"]))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "".join(parts).strip()
    s = str(content).strip()
    if (s.startswith("[{") or s.startswith("['") or s.startswith('["')) and "type" in s and "text" in s:
        import ast
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                parts = []
                for b in parsed:
                    if isinstance(b, dict) and "text" in b:
                        parts.append(str(b["text"]))
                    elif isinstance(b, str):
                        parts.append(b)
                if parts:
                    return "".join(parts).strip()
        except Exception:
            import re
            m = re.findall(r"'text':\s*['\"](.*?)['\"](?:\s*,\s*'extras'|\s*})", s, re.DOTALL)
            if m:
                return "\n\n".join(m).replace("\\n", "\n").replace('\\"', '"').strip()
    return s


def _heuristic_route_and_extract(user_query: str) -> RouterDecision:
    """Fast, high-precision intent router & slot extractor tailored to PackItUp's corpus."""
    q = (user_query or "").strip()
    q_low = q.lower()
    catalog = get_catalog()

    entities = ExtractedEntities(search_query=q)
    target_agents: List[str] = []

    # 1. Detect explicit pack size / MOQ or quantity
    # Check "add X packs" or "X packs of"
    m_packs = re.search(r"\b(\d+)\s*(?:packs?|boxes|sets|qty)\b", q_low)
    if m_packs:
        entities.quantity_packs = max(1, int(m_packs.group(1)))

    # Check piece count like "500 cups" or "1000 boxes"
    m_pieces = re.search(r"\b(100|200|250|300|400|500|1000|2000|5000)\s+(?:pcs|pieces|cups|boxes|bags|pouches|straws|bowls|trays)\b", q_low)
    if m_pieces:
        entities.desired_pieces = int(m_pieces.group(1))

    # Check pack MOQ hints like "50 pack", "pack of 20", "moq 100"
    m_moq = re.search(r"(?:pack\s*of\s*|moq\s*|size\s*)?\b(20|25|50|100|200)\b(?:\s*pack|\s*pcs)?", q_low)
    if m_moq:
        entities.pack_size_hint = m_moq.group(1)

    # Check color / size options
    for col in ("white", "brown", "black", "gold", "clear", "transparent", "kraft", "12oz", "16oz", "8mm", "12mm", "200ml", "500ml", "750ml", "1000ml"):
        if re.search(rf"\b{col}\b", q_low):
            entities.color_or_size_hint = col.capitalize() if col in ("white", "brown", "black", "gold") else col
            break

    # Check budget in PKR
    m_budget = re.search(r"(?:under|below|max|budget|less than)\s*(?:pkr|rs\.?\s*)?([0-9,]+)", q_low)
    if m_budget:
        try:
            entities.max_price_pkr = float(m_budget.group(1).replace(",", ""))
        except ValueError:
            pass

    # Check phone / email / lead contact info
    m_phone = re.search(r"(\+?92[\s\-]?\d{3}[\s\-]?\d{7}|03\d{2}[\s\-]?\d{7})", q)
    if m_phone:
        entities.phone = m_phone.group(1)
    m_email = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", q)
    if m_email:
        entities.email = m_email.group(1)
    m_name = re.search(r"(?:my name is|i am|name:)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", q, re.IGNORECASE)
    if m_name:
        entities.customer_name = m_name.group(1).strip()
    m_comp = re.search(r"(?:company|brand|bakery|cafe|restaurant|from)\s*(?:is|:)?\s*([A-Z][A-Za-z0-9 &']+)", q)
    if m_comp:
        entities.company_name = m_comp.group(1).strip()

    for city_name in ("Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Multan", "Peshawar", "Hyderabad", "Quetta", "Sialkot"):
        if city_name.lower() in q_low:
            entities.city = city_name
            break

    # 2. Detect Cart Operations (check 'add' and 'remove' before 'view')
    if any(k in q_low for k in ("clear cart", "empty cart", "empty my cart", "reset cart")):
        entities.cart_action = "clear"
        target_agents.append("A2_pricing_and_cart")
    elif any(k in q_low for k in ("remove from cart", "delete from cart", "take out of cart")) or re.search(r"\bremove\b.*\bcart\b", q_low):
        entities.cart_action = "remove"
        target_agents.append("A2_pricing_and_cart")
    elif any(k in q_low for k in ("add to cart", "add to my cart", "put in cart", "add 1", "add 2", "add 3", "add 4", "add 5", "add the", "add this", "order 1", "order 2")) or re.search(r"\badd\b.*\b(?:to\s+(?:my\s+)?cart|pack|box|cup|pouch|bowl|straw)", q_low):
        entities.cart_action = "add"
        target_agents.append("A2_pricing_and_cart")
    elif any(k in q_low for k in ("checkout", "place order", "buy now", "payment link", "proceed to checkout")):
        entities.cart_action = "checkout"
        target_agents.append("A6_order_support")
    elif any(k in q_low for k in ("view cart", "view my cart", "show cart", "show my cart", "what is in my cart", "what's in my cart")):
        entities.cart_action = "view"
        target_agents.append("A2_pricing_and_cart")

    # Extract clean product identifier when adding/quoting
    cleaned_prod = q_low
    for noise in (
        "please", "can you", "automatically", "add to my cart", "add to cart", "put in my cart",
        "add", "packs of", "pack of", "packs", "pack", "to my cart", "to cart", "how much is",
        "how much for", "price of", "quote for", "in white", "in brown",
    ):
        cleaned_prod = cleaned_prod.replace(noise, " ")
    cleaned_prod = re.sub(r"\b\d+\s*(?:packs?|pcs|pieces)\b", " ", cleaned_prod).strip()
    matched_prod = catalog.get_product(cleaned_prod) or catalog.get_product(q)
    if matched_prod:
        entities.product_identifier = matched_prod.handle

    # 3. Detect Policy / Shipping / Refund / Privacy / Contact questions
    if any(k in q_low for k in ("shipping", "delivery", "deliver", "express", "same-day", "same day", "karachi delivery", "lahore delivery", "refund", "return", "exchange", "damaged", "broken", "privacy", "terms of service", "payment policy", "cod", "cash on delivery", "address", "location", "contact", "phone", "whatsapp")):
        if "A4_policy_compliance" not in target_agents:
            target_agents.append("A4_policy_compliance")
        entities.policy_topic = q

    # 4. Detect Custom Order / B2B / Logo / Lead Capture
    if any(k in q_low for k in ("custom", "logo", "branded", "printing", "printed", "wholesale", "bulk", "b2b", "lead", "callback", "call me", "contact me", "serving tray", "hang tag", "visiting card", "compliment card")) or entities.phone:
        if "A5_b2b_custom_lead" not in target_agents:
            target_agents.append("A5_b2b_custom_lead")
        entities.customization_details = q

    # 5. Detect Price Quote calculation
    if any(k in q_low for k in ("price", "cost", "how much", "quote", "calculate", "total for")) or entities.desired_pieces:
        if "A2_pricing_and_cart" not in target_agents:
            target_agents.append("A2_pricing_and_cart")

    # 6. Detect Collection / Category exploration
    if any(k in q_low for k in ("collection", "categories", "category", "what do you have for", "catalog", "all products", "types of")):
        if "A3_collection_navigator" not in target_agents:
            target_agents.append("A3_collection_navigator")

    # 7. Detect Consultative / Advisory Queries (e.g., "help me think through", "ask relevant questions", "what packaging do i need")
    is_consultative = any(k in q_low for k in (
        "think through", "ask me questions", "ask questions", "ask relevant questions",
        "help me choose", "help me decide", "guide me", "advice", "advise",
        "starting a", "opening a", "new business", "what do i need for my",
        "packaging needs for my", "what do you suggest for a", "consultation",
        "help me think", "help me plan", "packaging strategy"
    ))

    # 8. Default or supplementary Product Discovery
    if not is_consultative:
        if not target_agents or any(k in q_low for k in ("show", "find", "looking for", "need", "recommend", "suggest", "box", "cup", "pouch", "bag", "bowl", "straw", "cutlery", "napkin", "sticker", "ribbon", "flyer", "tray", "bakery", "burger", "boba", "coffee", "cake", "cupcake", "brownie", "donut", "fries")):
            if "A1_product_discovery" not in target_agents and entities.cart_action not in ("clear", "view", "remove"):
                target_agents.insert(0, "A1_product_discovery")
    else:
        # For consultative dialog, provide discovery context for relevant business category
        if not target_agents:
            target_agents.append("A1_product_discovery")

    primary_intent = "multi_domain" if len(target_agents) > 1 else "product_lookup"
    if is_consultative:
        primary_intent = "consultative_advisory"
    elif entities.cart_action:
        primary_intent = "cart_operation"
    elif target_agents == ["A4_policy_compliance"]:
        primary_intent = "policy_question"
    elif target_agents == ["A5_b2b_custom_lead"]:
        primary_intent = "custom_order_lead"
    elif target_agents == ["A3_collection_navigator"]:
        primary_intent = "browse_category"

    return RouterDecision(
        primary_intent=primary_intent,
        target_agents=target_agents,  # type: ignore[arg-type]
        reasoning=f"Routed query across {target_agents} based on intent={primary_intent} and extracted slots.",
        entities=entities,
    )


# =====================================================================
# LangGraph Node Implementations
# =====================================================================

def node_a0_supervisor(state: AgentState) -> Dict[str, Any]:
    """A0 Orchestrator & Intent Router Node — Instant slot-aware deterministic router."""
    user_query = state["user_query"]
    decision = _heuristic_route_and_extract(user_query)

    # Only fall back to LLM structured router if heuristic matched absolutely no target agents
    if not decision.target_agents:
        llm = _get_llm()
        if llm is not None:
            try:
                structured_llm = llm.with_structured_output(RouterDecision)
                llm_decision = structured_llm.invoke([
                    SystemMessage(content=A0_SUPERVISOR_SYSTEM_PROMPT),
                    HumanMessage(content=user_query),
                ])
                if llm_decision and llm_decision.target_agents:
                    decision = llm_decision
            except Exception:
                pass

    return {
        "router_decision": (decision.model_dump() if hasattr(decision, 'model_dump') else decision.dict()),
        "pending_agents": list(decision.target_agents),
        "executed_agents": [],
        "specialist_outputs": [],
    }


def node_a1_product_discovery(state: AgentState) -> Dict[str, Any]:
    """A1 Product Discovery Specialist Node — Queries ChromaDB + Lexical RRF."""
    rag = get_rag_engine()
    decision = state.get("router_decision") or {}
    entities = decision.get("entities") or {}
    query = entities.get("search_query") or state["user_query"]
    cat_filter = state.get("active_category_filter") or entities.get("category_or_collection")
    max_price = state.get("active_max_price_pkr") or entities.get("max_price_pkr")
    in_stock = state.get("in_stock_only", False)

    cards = rag.search_products(
        query=query,
        category=cat_filter,
        max_price_pkr=max_price,
        in_stock_only=in_stock,
        top_k=6,
    )
    # Fallback without category filter if 0 results
    if not cards and cat_filter:
        cards = rag.search_products(query=query, max_price_pkr=max_price, in_stock_only=in_stock, top_k=6)

    citations = [c["url"] for c in cards if c.get("url")]
    pending = [a for a in state.get("pending_agents", []) if a != "A1_product_discovery"]
    executed = state.get("executed_agents", []) + ["A1_product_discovery"]
    outputs = state.get("specialist_outputs", []) + [
        {
            "agent": "A1_product_discovery",
            "summary": f"Found {len(cards)} matching PackItUp products for '{query}'.",
            "products": cards,
        }
    ]
    existing_cards = state.get("product_cards", [])
    seen_ids = {c["id"] for c in existing_cards}
    merged_cards = existing_cards + [c for c in cards if c["id"] not in seen_ids]

    return {
        "pending_agents": pending,
        "executed_agents": executed,
        "specialist_outputs": outputs,
        "product_cards": merged_cards[:8],
        "citations": list(dict.fromkeys(state.get("citations", []) + citations)),
    }


def node_a2_pricing_and_cart(state: AgentState) -> Dict[str, Any]:
    """A2 Pricing, Variant & Cart Manager Node — Pack quotes & automatic cart mutations."""
    catalog = get_catalog()
    rag = get_rag_engine()
    session_id = state.get("session_id", "default")
    decision = state.get("router_decision") or {}
    entities = decision.get("entities") or {}

    cart_action = entities.get("cart_action")
    prod_ident = entities.get("product_identifier")
    qty_packs = int(entities.get("quantity_packs") or 1)
    desired_pieces = entities.get("desired_pieces")
    pack_hint = entities.get("pack_size_hint")
    color_hint = entities.get("color_or_size_hint")

    # If no explicit product_identifier yet, pick top product from A1 cards or RAG search
    if not prod_ident:
        if state.get("product_cards"):
            prod_ident = state["product_cards"][0]["handle"]
        else:
            hits = rag.search_products(query=state["user_query"], top_k=1)
            if hits:
                prod_ident = hits[0]["handle"]

    cart_notification = state.get("cart_notification")
    a2_result: Dict[str, Any] = {"agent": "A2_pricing_and_cart"}
    new_cards = list(state.get("product_cards", []))
    citations = list(state.get("citations", []))

    if cart_action == "clear":
        res = CartManager.clear_cart(session_id)
        cart_notification = res["message"]
        a2_result["cart_operation"] = res
    elif cart_action == "view":
        res = CartManager.get_cart(session_id)
        a2_result["cart_operation"] = res
    elif cart_action == "remove":
        target = prod_ident or state["user_query"]
        res = CartManager.remove_item(session_id, target)
        cart_notification = res.get("message") or res.get("error")
        a2_result["cart_operation"] = res
    elif cart_action == "add" and prod_ident:
        res = CartManager.add_item(
            session_id=session_id,
            product_identifier=prod_ident,
            quantity_packs=qty_packs,
            pack_size_hint=pack_hint,
            color_hint=color_hint,
            natural_query=state["user_query"],
        )
        cart_notification = res.get("message") or res.get("error")
        a2_result["cart_operation"] = res
        prod_obj = catalog.get_product(prod_ident)
        if prod_obj:
            citations.append(prod_obj.url)
            if not any(c["id"] == prod_obj.id for c in new_cards):
                new_cards.insert(0, prod_obj.to_card_dict())

    # Always compute a deterministic variant pack quote if a product is identified
    if prod_ident:
        prod_obj = catalog.get_product(prod_ident)
        if prod_obj:
            citations.append(prod_obj.url)
            variant = catalog.resolve_variant(
                product=prod_obj,
                pack_size=pack_hint,
                color_or_option=color_hint,
                natural_query=state["user_query"],
            )
            pack_sz = variant.pack_size_int or 1
            if desired_pieces and pack_sz > 1:
                import math
                packs_needed = max(1, math.ceil(desired_pieces / pack_sz))
            else:
                packs_needed = max(1, qty_packs)

            a2_result["quote"] = {
                "product_title": prod_obj.title,
                "product_handle": prod_obj.handle,
                "source_url": prod_obj.url,
                "selected_variant": variant.to_dict(),
                "requires_custom_quote": variant.requires_custom_quote or prod_obj.requires_custom_quote,
                "packs_needed": packs_needed,
                "pack_size_units": pack_sz,
                "total_pieces": packs_needed * pack_sz,
                "unit_pack_price_pkr": variant.price_pkr,
                "total_price_pkr": round(packs_needed * variant.price_pkr, 2),
                "all_variants": [v.to_dict() for v in prod_obj.variants[:8]],
            }
            if not any(c["id"] == prod_obj.id for c in new_cards):
                new_cards.append(prod_obj.to_card_dict(selected_variant=variant))

    pending = [a for a in state.get("pending_agents", []) if a != "A2_pricing_and_cart"]
    executed = state.get("executed_agents", []) + ["A2_pricing_and_cart"]
    outputs = state.get("specialist_outputs", []) + [a2_result]

    return {
        "pending_agents": pending,
        "executed_agents": executed,
        "specialist_outputs": outputs,
        "product_cards": new_cards[:8],
        "cart_state": CartManager.get_cart(session_id),
        "cart_notification": cart_notification,
        "citations": list(dict.fromkeys(citations)),
    }


def node_a3_collection_navigator(state: AgentState) -> Dict[str, Any]:
    """A3 Collection & Category Navigator Node."""
    catalog = get_catalog()
    rag = get_rag_engine()
    query = state["user_query"]
    chunks = rag.retrieve(query=query, doc_type="collection", k=5)

    matched_cols = []
    cards = list(state.get("product_cards", []))
    seen_ids = {c["id"] for c in cards}
    citations = list(state.get("citations", []))

    for ch in chunks:
        url = ch["metadata"].get("source_url", "")
        col_handle = url.split("/collections/")[-1].strip("/") if "/collections/" in url else ""
        if col_handle in catalog.collection_map:
            cinfo = catalog.collection_map[col_handle]
            matched_cols.append({
                "handle": col_handle,
                "title": cinfo["title"],
                "url": cinfo["url"],
                "product_count": len(cinfo.get("product_handles", [])),
            })
            citations.append(cinfo["url"])
            for ph in cinfo.get("product_handles", [])[:2]:
                p = catalog.get_product(ph)
                if p and p.id not in seen_ids:
                    seen_ids.add(p.id)
                    cards.append(p.to_card_dict())
                    citations.append(p.url)

    pending = [a for a in state.get("pending_agents", []) if a != "A3_collection_navigator"]
    executed = state.get("executed_agents", []) + ["A3_collection_navigator"]
    outputs = state.get("specialist_outputs", []) + [
        {
            "agent": "A3_collection_navigator",
            "collections": matched_cols,
        }
    ]
    return {
        "pending_agents": pending,
        "executed_agents": executed,
        "specialist_outputs": outputs,
        "product_cards": cards[:8],
        "citations": list(dict.fromkeys(citations)),
    }


def node_a4_policy_compliance(state: AgentState) -> Dict[str, Any]:
    """A4 Policy & Compliance Guardrail Node — Hard-locked to `policies` namespace."""
    catalog = get_catalog()
    rag = get_rag_engine()
    query = state["user_query"]
    hits = rag.retrieve(query=query, doc_type="policy", k=4)

    q_low = query.lower()
    unpublished_404 = []
    for p in catalog.policies:
        if p.get("status") and (
            ("term" in q_low and "terms" in p["id"])
            or ("payment" in q_low and "payment" in p["id"])
        ):
            unpublished_404.append(p)

    citations = list(state.get("citations", []))
    for h in hits:
        if h["metadata"].get("source_url"):
            citations.append(h["metadata"]["source_url"])

    pending = [a for a in state.get("pending_agents", []) if a != "A4_policy_compliance"]
    executed = state.get("executed_agents", []) + ["A4_policy_compliance"]
    outputs = state.get("specialist_outputs", []) + [
        {
            "agent": "A4_policy_compliance",
            "policy_chunks": [
                {
                    "title": h["metadata"].get("title"),
                    "url": h["metadata"].get("source_url"),
                    "text": h["text"],
                }
                for h in hits
            ],
            "unpublished_404_policies": unpublished_404,
            "business_facts": catalog.business_facts,
        }
    ]
    return {
        "pending_agents": pending,
        "executed_agents": executed,
        "specialist_outputs": outputs,
        "citations": list(dict.fromkeys(citations)),
    }


def node_a5_b2b_custom_lead(state: AgentState) -> Dict[str, Any]:
    """A5 Custom Order & B2B Lead Capture Node — Writes leads to `data/leads.csv`."""
    rag = get_rag_engine()
    decision = state.get("router_decision") or {}
    entities = decision.get("entities") or {}
    query = state["user_query"]

    page_hits = rag.retrieve(query=query, doc_type="page", k=3)
    citations = list(state.get("citations", []))
    for h in page_hits:
        if h["metadata"].get("source_url"):
            citations.append(h["metadata"]["source_url"])

    captured_lead = state.get("captured_lead")
    # Automatically save lead if phone or email is provided in the prompt
    if entities.get("phone") or entities.get("email"):
        captured_lead = LeadStore.add_lead(
            customer_name=entities.get("customer_name") or "Chat Customer",
            phone=entities.get("phone") or "Provided via Email",
            email=entities.get("email") or "",
            company_name=entities.get("company_name") or "",
            city=entities.get("city") or "Karachi",
            inquiry_type="custom_printing",
            product_interest=entities.get("product_identifier") or entities.get("search_query") or "Custom Packaging",
            estimated_quantity=str(entities.get("desired_pieces") or entities.get("pack_size_hint") or "500-1000 pcs"),
            customization_details=query,
            source_channel=state.get("channel", "streamlit_chat"),
            notes="Auto-captured by Agent A5 from natural language conversation",
        )

    pending = [a for a in state.get("pending_agents", []) if a != "A5_b2b_custom_lead"]
    executed = state.get("executed_agents", []) + ["A5_b2b_custom_lead"]
    outputs = state.get("specialist_outputs", []) + [
        {
            "agent": "A5_b2b_custom_lead",
            "b2b_pages": [
                {
                    "title": h["metadata"].get("title"),
                    "url": h["metadata"].get("source_url"),
                    "text": h["text"],
                }
                for h in page_hits
            ],
            "lead_captured": captured_lead,
        }
    ]
    return {
        "pending_agents": pending,
        "executed_agents": executed,
        "specialist_outputs": outputs,
        "captured_lead": captured_lead,
        "citations": list(dict.fromkeys(citations)),
    }


def node_a6_order_support(state: AgentState) -> Dict[str, Any]:
    """A6 Order & Fulfillment Support Node — Generates Shopify checkout URL & escalation."""
    session_id = state.get("session_id", "default")
    catalog = get_catalog()
    shopify_client = ShopifyStorefrontClient()
    checkout_info = shopify_client.create_checkout_from_session(session_id=session_id)

    pending = [a for a in state.get("pending_agents", []) if a != "A6_order_support"]
    executed = state.get("executed_agents", []) + ["A6_order_support"]
    outputs = state.get("specialist_outputs", []) + [
        {
            "agent": "A6_order_support",
            "checkout": checkout_info,
            "support_contacts": catalog.business_facts,
        }
    ]
    return {
        "pending_agents": pending,
        "executed_agents": executed,
        "specialist_outputs": outputs,
    }


def _generate_smart_suggestions(state: AgentState) -> List[str]:
    """Generates contextual follow-up prompt pills for the Streamlit UI."""
    decision = state.get("router_decision") or {}
    if decision.get("primary_intent") == "consultative_advisory":
        return [
            "We serve smash burgers & loaded fries primarily for delivery",
            "We are a takeaway café serving coffee, boba & desserts",
            "We serve fried chicken, broast & signature dips",
            "I need custom logo printed packaging (MOQ 1,000 pcs)",
        ]

    cards = state.get("product_cards", [])
    cart = state.get("cart_state", {})
    suggestions: List[str] = []

    if cards:
        top = cards[0]
        suggestions.append(f"Add 2 packs of {top['title']} to my cart")
        suggestions.append(f"Quote price for 500 pieces of {top['title']}")
    if cart.get("item_count", 0) > 0:
        suggestions.append("Generate my Shopify checkout link")
    suggestions.extend([
        "Show eco-friendly bagasse bowls & wooden cutlery",
        "What are your Karachi & nationwide delivery timelines?",
        "I need custom logo printed burger boxes (1000 pcs)",
    ])
    return suggestions[:4]


def _deterministic_synthesize(state: AgentState) -> str:
    """Builds a grounded, citation-rich Markdown response when LLM is offline or rate-limited."""
    parts: List[str] = []
    outputs = state.get("specialist_outputs", [])
    cards = state.get("product_cards", [])
    cart = state.get("cart_state", {})
    cart_note = state.get("cart_notification")
    lead = state.get("captured_lead")
    decision = state.get("router_decision") or {}
    is_consult = decision.get("primary_intent") == "consultative_advisory"

    if is_consult:
        consult_header = (
            "Hello! I'm your Lead Packaging Consultant at **PackItUp.com.pk**. It's great to connect with an entrepreneur in the food & retail scene! "
            "Choosing the right packaging is all about balancing food presentation, heat/grease retention, and unit cost.\n\n"
            "To help build the ideal packaging checklist for your business, could you tell me a little more about your setup?\n\n"
            "1. **What are your signature menu items?** (e.g., Smash burgers, loaded fries, fried chicken, wraps, or rice bowls? This determines if we need corrugated clamshells vs greaseproof butter paper vs leak-proof meal boxes).\n"
            "2. **What is your primary service model?** (Dine-in, takeaway counter, or third-party delivery via Foodpanda/Careem where steam ventilation and tamper-evident bags are vital?)\n"
            "3. **What beverages or sides do you serve?** (e.g., Cold drinks/milkshakes in 12oz/16oz U-shaped PET cups with dome/flat lids, cup carriers, paper straws?)\n"
            "4. **Do you need portion dip cups?** (e.g., 1oz / 2oz dip cups with snap-on lids for sauces and dips?)\n"
            "5. **Are you starting with standard plain stock or custom logo branding?** (Standard plain kraft packaging has small MOQ pack sizes of 20–50 packs ready to ship, while custom logo printing starts around 500–1,000 pcs).\n\n"
            "Share your thoughts on any of these points, and I'll curate your tailored Starter Packaging Checklist with exact pack sizes and prices!"
        )
        parts.append(consult_header)

    if cart_note:
        parts.append(f"### 🛒 Cart Update\n**{cart_note}**\n- **Active Cart**: {cart.get('item_count', 0)} line item(s) ({cart.get('total_packs', 0)} packs) • **Subtotal: {cart.get('subtotal_display', 'PKR 0')}**")

    if lead:
        parts.append(
            f"### ✅ B2B Custom Lead Captured (`{lead['lead_id']}`)\n"
            f"Your inquiry for **{lead['product_interest']}** ({lead['estimated_quantity']}) has been logged to `leads.csv` under **{lead['customer_name']}** (`{lead['phone']}`). "
            f"Our Karachi packaging team (`+92 322 2620169` / `packitup.khi@gmail.com`) will reach out with a tailored dieline and wholesale quote."
        )

    for out in outputs:
        agent = out.get("agent")
        if agent == "A2_pricing_and_cart" and out.get("quote"):
            q = out["quote"]
            if q.get("requires_custom_quote"):
                parts.append(
                    f"### 📋 Custom Quote Required: [{q['product_title']}]({q['source_url']})\n"
                    f"- **Selected Option**: `{q['selected_variant']['title']}`\n"
                    f"- **Pricing Notice**: This item is listed at `PKR 0.00` on the storefront because it is made-to-order / custom-printed (or currently out of stock in that size). Share your **Name, WhatsApp number, and target quantity** in chat (or via the sidebar form) to log a custom quote lead!"
                )
            else:
                parts.append(
                    f"### 💰 Pack-Based Price Quote: [{q['product_title']}]({q['source_url']})\n"
                    f"- **Selected Variant**: `{q['selected_variant']['title']}` (`{q['selected_variant']['sku']}`)\n"
                    f"- **Pack Price**: **PKR {q['unit_pack_price_pkr']:,.0f}** per pack ({q['pack_size_units']} pcs/pack)\n"
                    f"- **Recommended Quantity**: **{q['packs_needed']} pack(s)** ({q['total_pieces']} total pieces) = **PKR {q['total_price_pkr']:,.0f}**\n"
                    f"- *Tip: Say **\"Add {q['packs_needed']} packs of {q['product_title']} to my cart\"** or click **Add to Cart** on the product card below!*"
                )

        elif agent == "A3_collection_navigator" and out.get("collections"):
            col_lines = [
                f"- **[{c['title']}]({c['url']})** — {c['product_count']} products (`{c['handle']}`)"
                for c in out["collections"]
            ]
            parts.append("### 🗂️ Matching PackItUp Collections\n" + "\n".join(col_lines))

        elif agent == "A4_policy_compliance":
            pol_lines = []
            for p in out.get("unpublished_404_policies", []):
                pol_lines.append(
                    f"- ⚠️ **{p['title']}**: Not published on the live storefront (`{p['url']}` returns HTTP 404). Please contact PackItUp directly at **+92 322 2620169** or **packitup.khi@gmail.com**."
                )
            for pc in out.get("policy_chunks", []):
                snippet = pc["text"].replace("\n", " ").strip()
                if len(snippet) > 420:
                    snippet = snippet[:417] + "..."
                pol_lines.append(f"- **[{pc['title']}]({pc['url']})**: {snippet}")
            bf = out.get("business_facts", {})
            if bf:
                pol_lines.append(
                    f"- **Official Karachi Store Contact**: {bf.get('address')} | Phone/WhatsApp: **{bf.get('phone')}** | Email: `{bf.get('email')}`"
                )
            parts.append("### 🛡️ Official Store Policy & Compliance Details\n" + "\n".join(pol_lines))

        elif agent == "A5_b2b_custom_lead" and not lead:
            b2b_lines = [
                f"- **[{bp['title']}]({bp['url']})**: Custom branded packaging & bulk B2B supply."
                for bp in out.get("b2b_pages", [])
            ]
            b2b_lines.append(
                "- **How to Lock in a Custom B2B Quote**: Reply with your **Name, Phone/WhatsApp (+92...), Brand Name, Product, and Quantity (e.g., 500 or 1000 pcs)**—or use the **B2B Lead Capture** tab—and I will log your request directly to `leads.csv`!"
            )
            parts.append("### 🏭 Custom Logo Printing & B2B Wholesale\n" + "\n".join(b2b_lines))

        elif agent == "A6_order_support":
            chk = out.get("checkout", {})
            if chk.get("success"):
                parts.append(
                    f"### ⚡ Instant Shopify Checkout Ready\n"
                    f"- **Cart Summary**: {chk.get('item_count')} item(s) • **Subtotal: {chk.get('subtotal_display', 'PKR ' + str(chk.get('subtotal_pkr')))}**\n"
                    f"- **Direct Shopify Checkout Link**: **[Click Here to Complete Checkout on PackItUp.com.pk]({chk['checkout_url']})**\n"
                    f"- **Need Order Support?**: Message our Karachi fulfillment desk on WhatsApp at **+92 322 2620169**."
                )
            else:
                parts.append(f"### 🛍️ Checkout & Order Support\n- {chk.get('error')}\n- For existing order tracking or damaged delivery claims (within 24h with photo/video proof), contact **+92 322 2620169**.")

    if cards and not is_consult:
        card_bullets = []
        for c in cards[:6]:
            stock_tag = "In Stock" if c["in_stock"] else ("Custom Quote Only" if c["requires_custom_quote"] else "Out of Stock")
            card_bullets.append(
                f"- **[{c['title']}]({c['url']})** — **{c['price_display']}** ({stock_tag} • Category: *{c['category']}*)"
            )
        parts.append(
            "### 📦 Recommended PackItUp Products\n"
            + "\n".join(card_bullets)
            + "\n\n*Browse the interactive product cards below to inspect variants, choose pack quantities, or click **Add to Cart**.*"
        )

    if not parts:
        parts.append(
            "I'm your **PackItUp Smart Shopping Assistant**! Ask me to find food-grade boxes, boba cups, kraft pouches, or eco-friendly bagasse bowls, calculate pack quotes in PKR, automatically add items to your cart, check Karachi/nationwide shipping policies, or log a custom logo printing lead."
        )

    return "\n\n".join(parts)


def node_synthesizer(state: AgentState) -> Dict[str, Any]:
    """Final Response Synthesizer Node — Combines specialist outputs, citations, and UI payloads."""
    fallback_md = _deterministic_synthesize(state)
    final_md = fallback_md

    # Fast-path instant response for cart mutations (deterministic & sub-millisecond)
    if state.get("cart_notification"):
        final_md = fallback_md
    else:
        llm = _get_llm()
        if llm is not None:
            try:
                decision = state.get("router_decision") or {}
                is_consult = decision.get("primary_intent") == "consultative_advisory"

                context_summary = {
                    "user_query": state["user_query"],
                    "primary_intent": decision.get("primary_intent"),
                    "executed_agents": state.get("executed_agents", []),
                    "cart_notification": state.get("cart_notification"),
                    "cart_state": state.get("cart_state"),
                    "captured_lead": state.get("captured_lead"),
                    "specialist_outputs": state.get("specialist_outputs", []),
                    "top_products": [
                        {
                            "title": c["title"],
                            "handle": c["handle"],
                            "url": c["url"],
                            "price_display": c["price_display"],
                            "category": c["category"],
                            "in_stock": c["in_stock"],
                            "requires_custom_quote": c["requires_custom_quote"],
                        }
                        for c in state.get("product_cards", [])[:5]
                    ],
                }

                instruction = (
                    "The customer is asking for packaging consultation and needs discovery. "
                    "Engage warmly in natural language, explain what factors to consider, and ask 3-5 sharp diagnostic questions "
                    "about their menu, service model (dine-in vs delivery), and drinks before recommending an exact packaging checklist. "
                    "Do NOT just dump a list of product cards or prices."
                    if is_consult
                    else "Write a concise, helpful, citation-grounded response in Markdown."
                )

                resp = llm.invoke([
                    SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"Customer message: {state['user_query']}\n\n"
                            f"Grounded Specialist Data:\n{context_summary}\n\n"
                            f"Instructions:\n{instruction}"
                        )
                    ),
                ])
                cleaned = _clean_llm_response(resp)
                if cleaned:
                    final_md = cleaned
            except Exception:
                final_md = fallback_md

    final_md = _clean_llm_response(final_md)
    suggestions = _generate_smart_suggestions(state)
    decision = state.get("router_decision") or {}
    is_consult = decision.get("primary_intent") == "consultative_advisory"

    ret: Dict[str, Any] = {
        "final_response": final_md,
        "suggested_prompts": suggestions,
        "messages": [AIMessage(content=final_md)],
    }
    if is_consult:
        ret["product_cards"] = []
    return ret


def _route_next_specialist(state: AgentState) -> str:
    """Conditional edge router that dispatches sequentially across all pending specialist agents."""
    pending = state.get("pending_agents", [])
    if not pending:
        return "synthesizer"
    next_agent = pending[0]
    valid_nodes = {
        "A1_product_discovery",
        "A2_pricing_and_cart",
        "A3_collection_navigator",
        "A4_policy_compliance",
        "A5_b2b_custom_lead",
        "A6_order_support",
    }
    return next_agent if next_agent in valid_nodes else "synthesizer"


@lru_cache(maxsize=1)
def build_packitup_graph():
    """Compiles and returns the Multi-Agent LangGraph `StateGraph`."""
    workflow = StateGraph(AgentState)

    workflow.add_node("A0_supervisor", node_a0_supervisor)
    workflow.add_node("A1_product_discovery", node_a1_product_discovery)
    workflow.add_node("A2_pricing_and_cart", node_a2_pricing_and_cart)
    workflow.add_node("A3_collection_navigator", node_a3_collection_navigator)
    workflow.add_node("A4_policy_compliance", node_a4_policy_compliance)
    workflow.add_node("A5_b2b_custom_lead", node_a5_b2b_custom_lead)
    workflow.add_node("A6_order_support", node_a6_order_support)
    workflow.add_node("synthesizer", node_synthesizer)

    workflow.add_edge(START, "A0_supervisor")

    routing_map = {
        "A1_product_discovery": "A1_product_discovery",
        "A2_pricing_and_cart": "A2_pricing_and_cart",
        "A3_collection_navigator": "A3_collection_navigator",
        "A4_policy_compliance": "A4_policy_compliance",
        "A5_b2b_custom_lead": "A5_b2b_custom_lead",
        "A6_order_support": "A6_order_support",
        "synthesizer": "synthesizer",
    }

    for source_node in [
        "A0_supervisor",
        "A1_product_discovery",
        "A2_pricing_and_cart",
        "A3_collection_navigator",
        "A4_policy_compliance",
        "A5_b2b_custom_lead",
        "A6_order_support",
    ]:
        workflow.add_conditional_edges(source_node, _route_next_specialist, routing_map)

    workflow.add_edge("synthesizer", END)
    return workflow.compile()


def run_shopping_assistant(
    user_query: str,
    session_id: str = "default",
    channel: str = "streamlit",
    category_filter: Optional[str] = None,
    max_price_pkr: Optional[float] = None,
    in_stock_only: bool = False,
) -> Dict[str, Any]:
    """High-level entry point to invoke a complete turn of the Multi-Agent LangGraph."""
    graph = build_packitup_graph()
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "session_id": session_id,
        "user_query": user_query,
        "channel": channel,
        "active_category_filter": category_filter,
        "active_max_price_pkr": max_price_pkr,
        "in_stock_only": in_stock_only,
        "router_decision": None,
        "pending_agents": [],
        "executed_agents": [],
        "specialist_outputs": [],
        "product_cards": [],
        "cart_state": CartManager.get_cart(session_id),
        "cart_notification": None,
        "captured_lead": None,
        "citations": [],
        "suggested_prompts": [],
        "final_response": "",
    }
    result = graph.invoke(initial_state)
    return {
        "session_id": session_id,
        "response": result.get("final_response", ""),
        "router_decision": result.get("router_decision"),
        "executed_agents": result.get("executed_agents", []),
        "product_cards": result.get("product_cards", []),
        "cart": result.get("cart_state", CartManager.get_cart(session_id)),
        "cart_notification": result.get("cart_notification"),
        "captured_lead": result.get("captured_lead"),
        "citations": result.get("citations", []),
        "suggested_prompts": result.get("suggested_prompts", []),
    }
