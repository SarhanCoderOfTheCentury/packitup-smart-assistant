"""Exports all LangChain tools for the PackItUp Multi-Agent LangGraph system."""
from src.tools.catalog_tools import (
    browse_collections,
    get_collection_items,
    get_product_details,
    search_products_rag,
)
from src.tools.cart_tools import (
    add_to_cart_tool,
    calculate_pack_quote,
    clear_cart_tool,
    remove_from_cart_tool,
    view_cart_tool,
)
from src.tools.lead_tools import (
    capture_customer_lead,
    search_b2b_custom_packaging,
)
from src.tools.policy_faq_tools import (
    escalate_to_human,
    get_business_facts,
    lookup_store_policy,
)
from src.tools.shopify_adapter import (
    ShopifyStorefrontClient,
    build_shopify_cart_permalink,
    create_shopify_checkout,
)

ALL_TOOLS = [
    search_products_rag,
    get_product_details,
    browse_collections,
    get_collection_items,
    calculate_pack_quote,
    add_to_cart_tool,
    remove_from_cart_tool,
    view_cart_tool,
    clear_cart_tool,
    capture_customer_lead,
    search_b2b_custom_packaging,
    lookup_store_policy,
    get_business_facts,
    escalate_to_human,
    create_shopify_checkout,
]

__all__ = [
    "ALL_TOOLS",
    "search_products_rag",
    "get_product_details",
    "browse_collections",
    "get_collection_items",
    "calculate_pack_quote",
    "add_to_cart_tool",
    "remove_from_cart_tool",
    "view_cart_tool",
    "clear_cart_tool",
    "capture_customer_lead",
    "search_b2b_custom_packaging",
    "lookup_store_policy",
    "get_business_facts",
    "escalate_to_human",
    "create_shopify_checkout",
    "build_shopify_cart_permalink",
    "ShopifyStorefrontClient",
]
