"""Industry-grade LangChain tools for Product Discovery (Agent A1) & Collection Navigation (Agent A3)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from src import tool
from pydantic import BaseModel, Field

from src.data_loader import get_catalog
from src.rag_chroma import get_rag_engine


class ProductSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="Natural language search query for products (e.g., '10x10x4 cake box with window', '12oz boba cups with lids', 'kraft stand up pouch', 'bagasse bowl').",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional category or collection filter (e.g., 'Cake boxes', 'Burger and Sandwich box', 'All Straws', 'Bagasse Bowls').",
    )
    max_price_pkr: Optional[float] = Field(
        default=None,
        description="Optional maximum starting pack price in PKR.",
    )
    in_stock_only: bool = Field(
        default=False,
        description="Set to True to restrict results to in-stock products with positive pricing.",
    )
    top_k: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Number of matching product cards to return.",
    )


@tool("search_products_rag", args_schema=ProductSearchInput)
def search_products_rag(
    query: str,
    category: Optional[str] = None,
    max_price_pkr: Optional[float] = None,
    in_stock_only: bool = False,
    top_k: int = 6,
) -> Dict[str, Any]:
    """Searches PackItUp's 187 products (244 vector chunks) using Hybrid ChromaDB + Lexical RRF retrieval.

    Always use this tool when a customer asks to find, compare, recommend, or browse packaging products.
    Returns structured product cards including image URL, title, handle, PKR price range, variants, and source_url.
    """
    rag = get_rag_engine()
    cards = rag.search_products(
        query=query,
        category=category,
        max_price_pkr=max_price_pkr,
        in_stock_only=in_stock_only,
        top_k=top_k,
    )
    return {
        "query": query,
        "count": len(cards),
        "products": cards,
        "citations": [c["url"] for c in cards],
    }


class ProductDetailInput(BaseModel):
    product_identifier: str = Field(
        ...,
        description="Product handle (e.g. '10x10x4-bakery-boxwindow'), title, or numeric product ID.",
    )


@tool("get_product_details", args_schema=ProductDetailInput)
def get_product_details(product_identifier: str) -> Dict[str, Any]:
    """Fetches the complete record for a specific PackItUp product including all priced variants, MOQ pack options, colors, stock status, images, and full description."""
    catalog = get_catalog()
    prod = catalog.get_product(product_identifier)
    if not prod:
        return {
            "found": False,
            "error": f"No product found matching '{product_identifier}'. Try `search_products_rag` first.",
        }
    return {
        "found": True,
        "product": prod.to_card_dict(),
        "source_url": prod.url,
    }


class CollectionBrowseInput(BaseModel):
    query: Optional[str] = Field(
        default=None,
        description="Optional industry or packaging category keyword (e.g., 'bakery', 'burger', 'coffee', 'pouches', 'eco', 'custom').",
    )


@tool("browse_collections", args_schema=CollectionBrowseInput)
def browse_collections(query: Optional[str] = None) -> Dict[str, Any]:
    """Lists PackItUp's 56 curated packaging collections (Agent A3) or searches collections by keyword."""
    catalog = get_catalog()
    rag = get_rag_engine()
    if query and query.strip():
        chunks = rag.retrieve(query=query, doc_type="collection", k=8)
        matched = []
        seen = set()
        for ch in chunks:
            url = ch["metadata"].get("source_url", "")
            col_handle = url.split("/collections/")[-1].strip("/") if "/collections/" in url else ""
            if col_handle in catalog.collection_map and col_handle not in seen:
                seen.add(col_handle)
                cinfo = catalog.collection_map[col_handle]
                matched.append({
                    "handle": col_handle,
                    "title": cinfo["title"],
                    "url": cinfo["url"],
                    "product_count": len(cinfo.get("product_handles", [])),
                    "sample_handles": cinfo.get("product_handles", [])[:5],
                })
        return {"query": query, "collections": matched}

    all_cols = [
        {
            "handle": k,
            "title": v["title"],
            "url": v["url"],
            "product_count": len(v.get("product_handles", [])),
        }
        for k, v in sorted(catalog.collection_map.items())
    ]
    return {"total_collections": len(all_cols), "collections": all_cols[:30]}


class CollectionItemsInput(BaseModel):
    collection_handle_or_title: str = Field(
        ...,
        description="Collection handle or title (e.g., 'cake-boxes', 'burger-box', 'bagasse-bowls', 'brown-kraft-pouches', 'all-straws').",
    )
    limit: int = Field(default=6, ge=1, le=16)


@tool("get_collection_items", args_schema=CollectionItemsInput)
def get_collection_items(collection_handle_or_title: str, limit: int = 6) -> Dict[str, Any]:
    """Returns full product cards for products inside a specific PackItUp collection hub."""
    catalog = get_catalog()
    key = collection_handle_or_title.strip().lower()
    target_handle = None
    if key in catalog.collection_map:
        target_handle = key
    else:
        for ch, cinfo in catalog.collection_map.items():
            if key in ch.lower() or key in cinfo.get("title", "").lower():
                target_handle = ch
                break

    if not target_handle:
        return {"found": False, "error": f"Collection '{collection_handle_or_title}' not found."}

    cinfo = catalog.collection_map[target_handle]
    cards = []
    for ph in cinfo.get("product_handles", [])[:limit]:
        prod = catalog.get_product(ph)
        if prod:
            cards.append(prod.to_card_dict())

    return {
        "found": True,
        "collection_handle": target_handle,
        "collection_title": cinfo["title"],
        "collection_url": cinfo["url"],
        "products": cards,
    }
