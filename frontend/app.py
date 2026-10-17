"""PackItUp Smart Shopping Assistant — Interactive Streamlit Storefront & Multi-Agent Console.

Run locally:
    streamlit run frontend/app.py
"""
from __future__ import annotations

# Streamlit Community Cloud (Debian) sqlite3 compatibility fix for ChromaDB
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path whether invoked from root or `frontend/`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.graph import run_shopping_assistant
from src.config import get_settings
from src.data_loader import get_catalog
from src.rag_chroma import get_rag_engine
from src.state_store import CartManager, LeadStore
from src.tools.shopify_adapter import ShopifyStorefrontClient

st.set_page_config(
    page_title="PackItUp.com.pk • AI Smart Shopping Assistant",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished e-commerce product cards and agent badges
st.markdown(
    """
    <style>
    .piu-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
        padding: 16px 22px;
        border-radius: 12px;
        border-left: 6px solid #D97706;
        margin-bottom: 14px;
    }
    .piu-banner h2 {
        margin: 0 0 4px 0;
        font-size: 20px;
        font-weight: 800;
        color: #FFFFFF;
    }
    .piu-banner p {
        margin: 0;
        font-size: 13px;
        color: #CBD5E1;
    }
    .agent-pill {
        display: inline-block;
        background: #FEF3C7;
        color: #B45309;
        border: 1px solid #F59E0B;
        padding: 2px 9px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .price-badge {
        font-size: 15px;
        font-weight: 800;
        color: #047857;
    }
    .custom-quote-badge {
        font-size: 11.5px;
        font-weight: 700;
        color: #B45309;
        background: #FEF3C7;
        border: 1px dashed #F59E0B;
        border-radius: 6px;
        padding: 4px 6px;
        text-align: center;
        margin-bottom: 6px;
    }

    /* Compact, Uniform Product Cards */
    .piu-card-img-wrap {
        width: 100%;
        height: 130px;
        background-color: #F8FAFC;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        margin-bottom: 6px;
        border: 1px solid #F1F5F9;
    }
    .piu-card-img-wrap img {
        max-height: 125px;
        max-width: 100%;
        width: auto;
        object-fit: contain;
    }
    .piu-card-tag {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.4px;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .piu-card-title {
        font-size: 12.5px;
        font-weight: 700;
        line-height: 1.3;
        height: 33px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        margin-bottom: 3px;
    }
    .piu-card-title a {
        color: #0F172A;
        text-decoration: none;
    }
    .piu-card-title a:hover {
        color: #D97706;
    }
    .piu-card-desc {
        font-size: 11px;
        line-height: 1.35;
        height: 30px;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        color: #64748B;
        margin-bottom: 6px;
    }

    /* Snug, Modern Chat Layout without Dead Space */
    .stMainBlockContainer, [data-testid="stMainBlockContainer"], .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 1400px;
    }
    [data-testid="stBottom"] {
        padding-bottom: 0px !important;
    }
    [data-testid="stChatInput"] {
        margin-top: 6px !important;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    [data-testid="stChatInput"] > div {
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06) !important;
        border: 1.5px solid #CBD5E1 !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #D97706 !important;
        box-shadow: 0 0 0 2px rgba(217, 119, 6, 0.2) !important;
    }
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading PackItUp 187-Product Catalog & Local ChromaDB Vector Index...")
def init_cached_resources():
    """Singleton initialization of Catalog and ChromaDB Hybrid RAG for Streamlit Cloud."""
    catalog = get_catalog()
    rag = get_rag_engine()
    return catalog, rag


catalog, rag_engine = init_cached_resources()
settings = get_settings()

# Initialize Streamlit Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = f"st_{uuid.uuid4().hex[:8]}"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "Welcome to **PackItUp.com.pk** (Karachi's premier food-service, bakery, café & retail packaging store)! 📦\n\n"
                "I can help you:\n"
                "- **Discover Products**: Browse 187 products & 1,131 variants across 56 collections with live PKR pricing\n"
                "- **Auto-Add to Cart by Prompt**: Just type *\"Add 2 packs of 10x10x4 white bakery box (50 pack) to my cart\"*\n"
                "- **Calculate Bulk Pack Quotes**: Get exact pack multiples for 100, 500, or 1,000+ pieces\n"
                "- **Custom Logo Printing & B2B Leads**: Request custom branded packaging and log directly to `leads.csv`"
            ),
            "product_cards": rag_engine.search_products("10x10x4 bakery box window kraft pouch boba cup", top_k=3),
            "executed_agents": ["A0_supervisor", "A1_product_discovery"],
            "suggested_prompts": [
                "Show me cake & bakery boxes with window",
                "Add 2 packs of 10x10x4 white bakery box (50 pack) to my cart",
                "Quote price for 500 12oz U-shaped boba cups with lids",
                "What are your Karachi & nationwide delivery timelines?",
            ],
        }
    ]
if "selected_Modal_Product" not in st.session_state:
    st.session_state.selected_Modal_Product = None


def call_agent_turn(
    query: str,
    category_filter: Optional[str] = None,
    max_price_pkr: Optional[float] = None,
    in_stock_only: bool = False,
) -> Dict[str, Any]:
    """Routes turn either through FastAPI (`AGENT_EXECUTION_MODE=fastapi`) or in-process LangGraph."""
    mode = st.session_state.get("exec_mode_override") or settings.agent_execution_mode
    if mode == "fastapi":
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(
                    f"{settings.fastapi_url.rstrip('/')}/api/v1/chat",
                    json={
                        "query": query,
                        "session_id": st.session_state.session_id,
                        "channel": "streamlit",
                        "category_filter": None if category_filter == "All Categories" else category_filter,
                        "max_price_pkr": max_price_pkr,
                        "in_stock_only": in_stock_only,
                    },
                )
                if r.status_code == 200:
                    return r.json()
        except Exception as exc:
            st.warning(f"FastAPI endpoint unreachable ({exc}); falling back to direct in-process LangGraph.")

    return run_shopping_assistant(
        user_query=query,
        session_id=st.session_state.session_id,
        channel="streamlit",
        category_filter=None if category_filter == "All Categories" else category_filter,
        max_price_pkr=max_price_pkr,
        in_stock_only=in_stock_only,
    )


@st.dialog("📋 Request Custom Logo Quote / B2B Lead")
def open_custom_quote_dialog(prefill_product: str = "Custom Printed Packaging"):
    """Modal dialog for capturing B2B or Custom Packaging leads into `data/leads.csv`."""
    st.markdown(
        f"Submit your customization specs for **{prefill_product}**. "
        "This record is immediately saved to `data/leads.csv` and routed to our Karachi B2B team."
    )
    with st.form("modal_lead_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Full Name *", placeholder="e.g., Bilal Ahmed")
            phone = st.text_input("Phone / WhatsApp *", placeholder="e.g., +92 300 1234567")
            email = st.text_input("Business Email", placeholder="e.g., bilal@brand.pk")
        with c2:
            company = st.text_input("Brand / Bakery / Café Name", placeholder="e.g., Sweet Crumb Studio")
            city = st.selectbox(
                "Delivery City",
                ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Multan", "Other"],
            )
            qty = st.selectbox(
                "Estimated Order Quantity (MOQ)",
                ["200–500 pcs", "500–1,000 pcs", "1,000–5,000 pcs", "5,000+ pcs"],
                index=1,
            )
        prod_int = st.text_input("Product / Packaging Type *", value=prefill_product)
        specs = st.text_area(
            "Customization Details (Logo colors, dimensions, finish)",
            placeholder="e.g., 1-color black logo printed on top lid, matte kraft finish",
        )
        budget = st.text_input("Target Budget (PKR, optional)", placeholder="e.g., 45000")
        submitted = st.form_submit_button("✅ Save Lead & Request Quote", type="primary", use_container_width=True)
        if submitted:
            if not name.strip() or not phone.strip():
                st.error("Please provide both your Name and Phone/WhatsApp number.")
            else:
                lead = LeadStore.add_lead(
                    customer_name=name,
                    phone=phone,
                    email=email,
                    company_name=company,
                    city=city,
                    inquiry_type="custom_printing",
                    product_interest=prod_int,
                    estimated_quantity=qty,
                    customization_details=specs,
                    budget_pkr=budget,
                    source_channel="streamlit_modal",
                )
                st.success(f"Lead `{lead['lead_id']}` saved to `leads.csv`! Our team (+92 322 2620169) will contact you shortly.")
                st.toast(f"✅ Saved Lead {lead['lead_id']} to leads.csv!", icon="📋")


def _clean_message_content(content: Any) -> str:
    """Ensures message content is clean formatted markdown without raw python dict/list artifacts."""
    if not content:
        return ""
    if hasattr(content, "content"):
        content = content.content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and "text" in b:
                parts.append(str(b["text"]))
            elif isinstance(b, str):
                parts.append(b)
            elif hasattr(b, "text"):
                parts.append(str(b.text))
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


def render_product_cards_grid(cards: List[Dict[str, Any]], key_prefix: str) -> None:
    """Renders a 4-column compact, consistently sized grid of Product Cards with variant dropdowns & Add to Cart buttons."""
    if not cards:
        return
    cols_per_row = 4
    for row_start in range(0, len(cards), cols_per_row):
        row_cards = cards[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for idx, card in enumerate(row_cards):
            with cols[idx]:
                with st.container(border=True):
                    # Fixed-height image wrapper to keep card sizes completely uniform
                    card_title = card.get("title", "")
                    card_url = card.get("url", "#")
                    cat_name = card.get("category", "Packaging")
                    img_url = card.get("image_url")
                    if img_url:
                        st.markdown(
                            f'<div class="piu-card-img-wrap"><img src="{img_url}" alt="{card_title}" loading="lazy"/></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="piu-card-img-wrap" style="color: #94a3b8; font-size: 32px;">📦</div>',
                            unsafe_allow_html=True,
                        )

                    # Category Badge & Clamped Title
                    st.markdown(f'<div class="piu-card-tag">🏷️ {cat_name}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="piu-card-title"><a href="{card_url}" target="_blank">{card_title}</a></div>', unsafe_allow_html=True)

                    # Consistent Short Description
                    desc = card.get("short_description", "")
                    short_desc = (desc[:90] + "...") if len(desc) > 90 else (desc or "Premium food-grade packaging from PackItUp.")
                    st.markdown(f'<div class="piu-card-desc">{short_desc}</div>', unsafe_allow_html=True)

                    # Variant Selector
                    variants = card.get("variants", [])
                    priced_vars = [v for v in variants if not v.get("requires_custom_quote") and v.get("price_pkr", 0) > 0]

                    unique_key = f"{key_prefix}_{card['id']}_{row_start}_{idx}"
                    if priced_vars:
                        var_labels = [
                            f"{v['title']} — PKR {v['price_pkr']:,.0f}" + ("" if v.get("available", True) else " (OOS)")
                            for v in priced_vars
                        ]
                        selected_idx = st.selectbox(
                            "Pack / Option (MOQ)",
                            options=range(len(priced_vars)),
                            format_func=lambda i, lbls=var_labels: lbls[i],
                            key=f"var_sel_{unique_key}",
                            label_visibility="collapsed",
                        )
                        chosen_var = priced_vars[selected_idx]
                        c_qty, c_btn = st.columns([1, 1.8])
                        with c_qty:
                            qty_packs = st.number_input(
                                "Packs",
                                min_value=1,
                                max_value=100,
                                value=1,
                                step=1,
                                key=f"qty_{unique_key}",
                                label_visibility="collapsed",
                            )
                        with c_btn:
                            line_pkr = chosen_var["price_pkr"] * qty_packs
                            if st.button(
                                f"🛒 Add ({line_pkr:,.0f})",
                                key=f"add_btn_{unique_key}",
                                type="primary",
                                use_container_width=True,
                            ):
                                res = CartManager.add_item(
                                    session_id=st.session_state.session_id,
                                    product_identifier=card["handle"],
                                    variant_id=chosen_var["id"],
                                    quantity_packs=int(qty_packs),
                                )
                                if res.get("success"):
                                    st.toast(res["message"], icon="🛒")
                                    st.rerun()
                                else:
                                    st.warning(res.get("error", "Could not add item."))
                    else:
                        st.markdown(
                            '<div class="custom-quote-badge">📋 Custom Quote (PKR 0)</div>',
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            "📋 Request Quote",
                            key=f"quote_btn_{unique_key}",
                            use_container_width=True,
                        ):
                            open_custom_quote_dialog(prefill_product=card["title"])


# =====================================================================
# Sidebar: Live Shopping Cart Drawer, Filters & Configuration
# =====================================================================
with st.sidebar:
    st.markdown("## 🛒 Live Shopping Cart")
    cart_data = CartManager.get_cart(st.session_state.session_id)
    m1, m2 = st.columns(2)
    m1.metric("Items / Packs", f"{cart_data['item_count']} / {cart_data['total_packs']}")
    m2.metric("Subtotal", cart_data["subtotal_display"])

    if cart_data["items"]:
        for item in cart_data["items"]:
            with st.container(border=True):
                st.markdown(f"**{item['product_title']}**")
                st.caption(
                    f"Variant: `{item['variant_title']}` • PKR {item['unit_pack_price_pkr']:,.0f}/pack"
                )
                sc1, sc2, sc3 = st.columns([1.3, 1, 0.8])
                with sc1:
                    new_q = st.number_input(
                        "Packs",
                        min_value=0,
                        max_value=200,
                        value=int(item["quantity_packs"]),
                        key=f"side_qty_{item['variant_id']}",
                        label_visibility="collapsed",
                    )
                    if new_q != int(item["quantity_packs"]):
                        CartManager.update_quantity(
                            st.session_state.session_id,
                            int(item["variant_id"]),
                            int(new_q),
                        )
                        st.rerun()
                with sc2:
                    st.markdown(f"**PKR {item['line_total_pkr']:,.0f}**")
                with sc3:
                    if st.button("🗑️", key=f"del_{item['variant_id']}", help="Remove item"):
                        CartManager.update_quantity(st.session_state.session_id, int(item["variant_id"]), 0)
                        st.rerun()

        b_chk, b_clr = st.columns([1.6, 1])
        with b_chk:
            if st.button("⚡ Shopify Checkout", type="primary", use_container_width=True):
                s_client = ShopifyStorefrontClient()
                chk = s_client.create_checkout_from_session(st.session_state.session_id)
                if chk.get("success"):
                    st.success("Checkout link ready!")
                    st.markdown(f"👉 **[Open PackItUp Shopify Checkout]({chk['checkout_url']})**")
        with b_clr:
            if st.button("Clear Cart", use_container_width=True):
                CartManager.clear_cart(st.session_state.session_id)
                st.rerun()
    else:
        st.info(
            "Your cart is empty. Click **🛒 Add** on any product card or tell the assistant: "
            "*\"Add 2 packs of 10x10x4 white bakery box to my cart\"*!"
        )

    st.divider()
    st.markdown("### 🔍 Smart Search Filters")
    all_categories = ["All Categories"] + catalog.list_categories()
    sidebar_category = st.selectbox("Filter by Category", options=all_categories, index=0)
    sidebar_max_price = st.slider(
        "Max Starting Pack Price (PKR)",
        min_value=0,
        max_value=30000,
        value=0,
        step=500,
        help="Set above 0 to pre-filter ChromaDB product chunks by maximum starting pack price.",
    )
    sidebar_in_stock = st.toggle("In-Stock Items Only", value=True)

    st.divider()
    st.markdown("### ⚙️ Runtime & Gemini API")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Enter your GOOGLE_API_KEY to activate live Gemini 2.5 Flash synthesis & embeddings.",
    )
    if api_key_input and api_key_input != os.getenv("GOOGLE_API_KEY", ""):
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.toast("Updated GOOGLE_API_KEY in environment!", icon="🔑")

    st.session_state.exec_mode_override = st.radio(
        "Agent Execution Mode",
        options=["direct", "fastapi"],
        index=0,
        horizontal=True,
        help="`direct` runs LangGraph inside Streamlit (best for Streamlit Cloud); `fastapi` calls the external FastAPI server.",
    )
    if st.button("📋 Open Quick B2B Lead Modal", use_container_width=True):
        open_custom_quote_dialog()


# =====================================================================
# Main Header Banner
# =====================================================================
st.markdown(
    """
    <div class="piu-banner">
        <h2>📦 PackItUp.com.pk — Multi-Agent AI Smart Shopping Assistant</h2>
        <p>Powered by <b>LangGraph StateGraph</b> • <b>Local ChromaDB Hybrid RAG (451 Chunks)</b> • <b>Google Gemini 2.5 Flash</b> • <b>Shopify &amp; WhatsApp Business Automation</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top Navigation Tabs
tab_chat, tab_catalog, tab_leads, tab_arch = st.tabs([
    "💬 AI Shopping Assistant & Auto-Cart",
    "🛍️ Visual Product Catalog (187 Products)",
    "📋 B2B Custom Leads CRM (`leads.csv`)",
    "🏗️ Architecture, Hosting & WhatsApp Blueprint",
])

# ---------------------------------------------------------------------
# TAB 1: Conversational AI Shopping Assistant
# ---------------------------------------------------------------------
with tab_chat:
    # Dedicated scrollable chat conversation window to keep input anchored and eliminate dead space
    chat_box = st.container(height=520, border=False)
    with chat_box:
        for msg_idx, msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg["role"], avatar="📦" if msg["role"] == "assistant" else "🧑‍💼"):
                # Show multi-agent execution trace badges for assistant messages
                if msg["role"] == "assistant" and msg.get("executed_agents"):
                    badges_html = "".join(
                        f'<span class="agent-pill">⚡ {ag}</span>' for ag in msg["executed_agents"]
                    )
                    st.markdown(badges_html, unsafe_allow_html=True)

                st.markdown(_clean_message_content(msg["content"]))

                # Render interactive Product Cards inline below the assistant's reply
                if msg.get("product_cards"):
                    with st.expander(
                        f"📦 View {len(msg['product_cards'])} Recommended Product Cards (Click to Add to Cart)",
                        expanded=(msg_idx == len(st.session_state.chat_history) - 1),
                    ):
                        render_product_cards_grid(msg["product_cards"], key_prefix=f"chat_{msg_idx}")

    # Quick Suggestion Pills for the latest turn
    latest_msg = st.session_state.chat_history[-1] if st.session_state.chat_history else {}
    suggestions = latest_msg.get("suggested_prompts", [])
    selected_pill = None
    if suggestions:
        selected_pill = st.pills(
            "💡 Suggested Next Prompts (Click to run):",
            options=suggestions,
            key=f"pills_{len(st.session_state.chat_history)}",
        )

    user_prompt = st.chat_input(
        "Ask about boxes, cups, pouches, shipping policies, custom printing, or say 'Add 2 packs of ... to my cart'..."
    )

    active_query = user_prompt or selected_pill
    if active_query:
        st.session_state.chat_history.append({"role": "user", "content": active_query})
        with st.spinner("LangGraph A0 Supervisor dispatching across specialist agents & ChromaDB..."):
            result = call_agent_turn(
                query=active_query,
                category_filter=sidebar_category,
                max_price_pkr=float(sidebar_max_price) if sidebar_max_price > 0 else None,
                in_stock_only=sidebar_in_stock,
            )
        if result.get("cart_notification"):
            st.toast(result["cart_notification"], icon="🛒")
        if result.get("captured_lead"):
            st.toast(f"Saved Lead {result['captured_lead']['lead_id']} to leads.csv!", icon="📋")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": _clean_message_content(result.get("response", "")),
            "product_cards": result.get("product_cards", []),
            "executed_agents": ["A0_supervisor"] + result.get("executed_agents", []),
            "suggested_prompts": result.get("suggested_prompts", []),
        })
        st.rerun()

# ---------------------------------------------------------------------
# TAB 2: Interactive Visual Catalog & Collection Explorer
# ---------------------------------------------------------------------
with tab_catalog:
    st.markdown("### 🛍️ Explore All 187 PackItUp Products & 56 Collections")
    f_col1, f_col2, f_col3 = st.columns([2, 1.5, 1])
    with f_col1:
        cat_search_q = st.text_input(
            "Search by keyword, dimension (e.g., 10x10x4, 12oz, 200ml), or tag",
            placeholder="e.g., brownie box, stand up pouch, paper straw, burger box...",
        )
    with f_col2:
        col_options = ["All Collections"] + [
            f"{v['title']} ({k})" for k, v in sorted(catalog.collection_map.items())
        ]
        chosen_collection = st.selectbox("Browse by Collection Hub (56 Hubs)", options=col_options)
    with f_col3:
        max_cards_to_show = st.selectbox("Products per page", options=[8, 16, 24, 48], index=0)

    if cat_search_q.strip():
        catalog_cards = rag_engine.search_products(
            query=cat_search_q,
            category=None if sidebar_category == "All Categories" else sidebar_category,
            max_price_pkr=float(sidebar_max_price) if sidebar_max_price > 0 else None,
            in_stock_only=sidebar_in_stock,
            top_k=max_cards_to_show,
        )
    elif chosen_collection != "All Collections":
        col_handle = chosen_collection.rsplit("(", 1)[-1].rstrip(")")
        handles = catalog.collection_map.get(col_handle, {}).get("product_handles", [])
        catalog_cards = []
        for h in handles:
            p = catalog.get_product(h)
            if p:
                if sidebar_in_stock and not p.in_stock and not p.requires_custom_quote:
                    continue
                catalog_cards.append(p.to_card_dict())
        catalog_cards = catalog_cards[:max_cards_to_show]
    else:
        all_prods = list(catalog.products_by_handle.values())
        if sidebar_category != "All Categories":
            all_prods = [p for p in all_prods if p.enriched_category == sidebar_category]
        if sidebar_in_stock:
            all_prods = [p for p in all_prods if p.in_stock]
        if sidebar_max_price > 0:
            all_prods = [p for p in all_prods if p.price_min_pkr <= sidebar_max_price]
        catalog_cards = [p.to_card_dict() for p in all_prods[:max_cards_to_show]]

    st.caption(f"Showing **{len(catalog_cards)}** products (Prices in PKR per MOQ pack)")
    render_product_cards_grid(catalog_cards, key_prefix="catalog_tab")

# ---------------------------------------------------------------------
# TAB 3: B2B Custom Leads CRM (`data/leads.csv`)
# ---------------------------------------------------------------------
with tab_leads:
    st.markdown("### 📋 B2B Custom Packaging & Wholesale Lead Capture (`data/leads.csv`)")
    l_col1, l_col2 = st.columns([1.1, 1.5])
    with l_col1:
        with st.container(border=True):
            st.markdown("#### ➕ Log a New B2B / Custom Printing Lead")
            with st.form("tab_lead_form", clear_on_submit=True):
                f_name = st.text_input("Customer Name *", placeholder="e.g., Sara Siddiqui")
                f_phone = st.text_input("Phone / WhatsApp *", placeholder="e.g., +92 321 9876543")
                f_email = st.text_input("Email Address", placeholder="e.g., sara@artisancafe.pk")
                f_comp = st.text_input("Company / Café / Bakery", placeholder="e.g., Artisan Roasters")
                f_city = st.selectbox("City", ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Multan"])
                f_prod = st.text_input("Product of Interest *", placeholder="e.g., Custom Printed 12oz Coffee Cups & Sleeves")
                f_qty = st.text_input("Estimated Quantity (MOQ)", value="1,000 pcs")
                f_spec = st.text_area("Customization / Printing Specs", placeholder="2-color logo, matte finish")
                f_budget = st.text_input("Estimated Budget (PKR)", placeholder="e.g., 55000")
                if st.form_submit_button("💾 Save Lead to leads.csv", type="primary", use_container_width=True):
                    if not f_name.strip() or not f_phone.strip():
                        st.error("Customer Name and Phone/WhatsApp are required.")
                    else:
                        rec = LeadStore.add_lead(
                            customer_name=f_name,
                            phone=f_phone,
                            email=f_email,
                            company_name=f_comp,
                            city=f_city,
                            inquiry_type="custom_printing",
                            product_interest=f_prod,
                            estimated_quantity=f_qty,
                            customization_details=f_spec,
                            budget_pkr=f_budget,
                            source_channel="streamlit_crm_tab",
                        )
                        st.success(f"Saved Lead `{rec['lead_id']}` to `data/leads.csv`!")
                        st.rerun()

    with l_col2:
        st.markdown("#### 📊 Live `data/leads.csv` Database")
        leads_list = LeadStore.list_leads()
        if leads_list:
            df_leads = pd.DataFrame(leads_list)
            st.dataframe(df_leads, use_container_width=True, hide_index=True)
            csv_bytes = settings.leads_csv_path.read_bytes() if settings.leads_csv_path.exists() else b""
            st.download_button(
                "⬇️ Download Complete `leads.csv`",
                data=csv_bytes,
                file_name="packitup_leads.csv",
                mime="text/csv",
                type="primary",
            )
        else:
            st.info("No leads captured yet.")

# ---------------------------------------------------------------------
# TAB 4: Architecture Diagrams, Hosting & WhatsApp Business Blueprint
# ---------------------------------------------------------------------
with tab_arch:
    st.markdown("### 🏗️ System Architecture & End-to-End WhatsApp + Shopify Blueprint")
    diag_dir = PROJECT_ROOT / "docs" / "diagrams"
    diag_files = [
        ("1. Multi-Agent LangGraph Orchestration & Tool Topology", "01_multi_agent_langgraph_architecture.png"),
        ("2. Local ChromaDB Ingestion, Data Hygiene & Hybrid RAG Pipeline", "02_chroma_rag_hybrid_pipeline.png"),
        ("3. WhatsApp Business Cloud API & Shopify End-to-End Automation", "03_whatsapp_shopify_end_to_end_automation.png"),
        ("4. Cloud Hosting Topology: Streamlit Community Cloud & FastAPI", "04_cloud_deployment_topology.png"),
    ]
    for title, fname in diag_files:
        p = diag_dir / fname
        if p.exists():
            with st.expander(title, expanded=True):
                st.image(str(p), use_container_width=True)
