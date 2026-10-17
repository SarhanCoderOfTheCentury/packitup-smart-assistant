"""Production FastAPI Server for PackItUp Multi-Agent LangGraph & ChromaDB RAG Assistant.

Run locally:
    uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.agents.graph import run_shopping_assistant
from src.backend.schemas import (
    CartAddRequest,
    CartUpdateRequest,
    ChatRequest,
    ChatResponse,
    LeadCreateRequest,
)
from src.backend.whatsapp_webhook import router as whatsapp_router
from src.config import get_settings
from src.data_loader import get_catalog
from src.rag_chroma import get_rag_engine
from src.state_store import CartManager, LeadStore
from src.tools.shopify_adapter import ShopifyStorefrontClient

app = FastAPI(
    title="PackItUp Smart Shopping Assistant API",
    description=(
        "Multi-Agent LangGraph + ChromaDB Hybrid RAG + Shopify Storefront & "
        "WhatsApp Business Cloud Automation API for PackItUp.com.pk"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(whatsapp_router)


@app.get("/health")
def health_check() -> Dict[str, Any]:
    settings = get_settings()
    catalog = get_catalog()
    rag = get_rag_engine()
    return {
        "status": "ok",
        "store": catalog.business_facts.get("name", "PackItUp"),
        "domain": catalog.business_facts.get("domain", "packitup.com.pk"),
        "total_products": len(catalog.products_by_handle),
        "total_variants": len(catalog.variants_by_id),
        "total_collections": len(catalog.collection_map),
        "total_chunks": len(catalog.chunks),
        "embedding_backend": rag.embedding_backend,
        "gemini_configured": settings.has_gemini_key,
    }


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> Dict[str, Any]:
    """Executes a complete Multi-Agent LangGraph turn and returns reply, product cards, cart, and leads."""
    return run_shopping_assistant(
        user_query=req.query,
        session_id=req.session_id,
        channel=req.channel,
        category_filter=req.category_filter,
        max_price_pkr=req.max_price_pkr,
        in_stock_only=req.in_stock_only,
    )


@app.get("/api/v1/catalog/products")
def list_catalog_products(
    query: str = Query(default="", description="Optional search query"),
    category: Optional[str] = Query(default=None),
    max_price_pkr: Optional[float] = Query(default=None),
    in_stock_only: bool = Query(default=False),
    limit: int = Query(default=24, ge=1, le=200),
) -> Dict[str, Any]:
    catalog = get_catalog()
    rag = get_rag_engine()
    if query.strip():
        cards = rag.search_products(
            query=query,
            category=category,
            max_price_pkr=max_price_pkr,
            in_stock_only=in_stock_only,
            top_k=limit,
        )
    else:
        prods = list(catalog.products_by_handle.values())
        if category and category != "All Categories":
            prods = [
                p for p in prods
                if category.lower() in p.enriched_category.lower()
                or any(category.lower() in ct.lower() for ct in p.collection_titles)
            ]
        if in_stock_only:
            prods = [p for p in prods if p.in_stock]
        if max_price_pkr is not None and max_price_pkr > 0:
            prods = [p for p in prods if p.price_min_pkr <= max_price_pkr]
        cards = [p.to_card_dict() for p in prods[:limit]]

    return {
        "count": len(cards),
        "categories": catalog.list_categories(),
        "products": cards,
    }


@app.get("/api/v1/cart/{session_id}")
def get_cart_endpoint(session_id: str) -> Dict[str, Any]:
    return CartManager.get_cart(session_id)


@app.post("/api/v1/cart/add")
def add_to_cart_endpoint(req: CartAddRequest) -> Dict[str, Any]:
    return CartManager.add_item(
        session_id=req.session_id,
        product_identifier=req.product_identifier,
        variant_id=req.variant_id,
        quantity_packs=req.quantity_packs,
        pack_size_hint=req.pack_size_hint,
        color_hint=req.color_hint,
    )


@app.post("/api/v1/cart/update")
def update_cart_endpoint(req: CartUpdateRequest) -> Dict[str, Any]:
    return CartManager.update_quantity(
        session_id=req.session_id,
        variant_id=req.variant_id,
        quantity_packs=req.quantity_packs,
    )


@app.delete("/api/v1/cart/{session_id}")
def clear_cart_endpoint(session_id: str) -> Dict[str, Any]:
    return CartManager.clear_cart(session_id)


@app.post("/api/v1/cart/{session_id}/checkout")
def checkout_cart_endpoint(session_id: str) -> Dict[str, Any]:
    client = ShopifyStorefrontClient()
    return client.create_checkout_from_session(session_id=session_id)


@app.get("/api/v1/leads")
def list_leads_endpoint() -> Dict[str, Any]:
    leads = LeadStore.list_leads()
    return {"count": len(leads), "leads": leads}


@app.post("/api/v1/leads")
def create_lead_endpoint(req: LeadCreateRequest) -> Dict[str, Any]:
    record = LeadStore.add_lead(**(req.model_dump() if hasattr(req, 'model_dump') else req.dict()))
    return {"success": True, "lead": record}


@app.get("/api/v1/leads/export")
def export_leads_csv() -> FileResponse:
    settings = get_settings()
    LeadStore.list_leads()  # Ensures file exists
    return FileResponse(
        path=str(settings.leads_csv_path),
        media_type="text/csv",
        filename="packitup_leads.csv",
    )
