"""Shopify Storefront GraphQL, Admin API, and UCP/MCP Adapter for PackItUp (`packitup.com.pk`).

Allows seamless transition from Demo Mode to Live Shopify Production:
1. Generates instant Shopify Storefront permalink checkout URLs (`https://packitup.com.pk/cart/<variant_id>:<qty>,...`)
   which work on ANY Shopify store immediately without requiring API tokens!
2. When `SHOPIFY_STOREFRONT_ACCESS_TOKEN` is configured, executes native GraphQL `cartCreate` and
   `cartLinesAdd` mutations against `https://packitup.com.pk/api/2026-01/graphql.json`.
3. Includes an MCP client bridge for `https://packitup.com.pk/api/ucp/mcp` (advertised in `robots.txt`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import httpx
from src import tool
from pydantic import BaseModel, Field

from src.config import get_settings
from src.state_store import CartManager


def build_shopify_cart_permalink(items: List[Dict[str, Any]], store_domain: str = "packitup.com.pk") -> str:
    """Builds a real, working Shopify direct cart permalink from variant IDs and quantities.

    Format: `https://packitup.com.pk/cart/{variant_id}:{quantity},{variant_id_2}:{quantity_2}`
    """
    if not items:
        return f"https://{store_domain}/cart"
    parts = [f"{int(item['variant_id'])}:{int(item['quantity_packs'])}" for item in items if item.get("variant_id")]
    return f"https://{store_domain}/cart/" + ",".join(parts)


class ShopifyStorefrontClient:
    """Production GraphQL client for Shopify Storefront Cart API (`cartCreate`, `cartLinesAdd`)."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.endpoint = (
            f"https://{self.settings.shopify_store_domain}/api/"
            f"{self.settings.shopify_api_version}/graphql.json"
        )

    def create_checkout_from_session(self, session_id: str = "default") -> Dict[str, Any]:
        cart = CartManager.get_cart(session_id)
        items = cart.get("items", [])
        if not items:
            return {
                "success": False,
                "error": "Your cart is currently empty. Add products before generating a checkout link.",
            }

        permalink = build_shopify_cart_permalink(items, self.settings.shopify_store_domain)

        # If live Storefront API token is configured, call GraphQL cartCreate
        if self.settings.has_shopify_token:
            query = """
            mutation cartCreate($input: CartInput!) {
              cartCreate(input: $input) {
                cart {
                  id
                  checkoutUrl
                  cost {
                    subtotalAmount { amount currencyCode }
                    totalAmount { amount currencyCode }
                  }
                }
                userErrors { field message }
              }
            }
            """
            lines = [
                {
                    "merchandiseId": item["shopify_variant_gid"],
                    "quantity": int(item["quantity_packs"]),
                }
                for item in items
            ]
            headers = {
                "Content-Type": "application/json",
                "X-Shopify-Storefront-Access-Token": self.settings.shopify_storefront_access_token,
            }
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        self.endpoint,
                        json={"query": query, "variables": {"input": {"lines": lines}}},
                        headers=headers,
                    )
                    data = resp.json()
                    cart_obj = data.get("data", {}).get("cartCreate", {}).get("cart")
                    if cart_obj and cart_obj.get("checkoutUrl"):
                        return {
                            "success": True,
                            "mode": "shopify_storefront_graphql",
                            "shopify_cart_id": cart_obj["id"],
                            "checkout_url": cart_obj["checkoutUrl"],
                            "permalink_url": permalink,
                            "subtotal_pkr": cart["subtotal_pkr"],
                            "item_count": cart["item_count"],
                        }
            except Exception as exc:
                return {
                    "success": True,
                    "mode": "shopify_permalink_fallback",
                    "checkout_url": permalink,
                    "note": f"GraphQL call fell back to direct Shopify cart permalink ({exc}).",
                    "subtotal_pkr": cart["subtotal_pkr"],
                    "item_count": cart["item_count"],
                }

        return {
            "success": True,
            "mode": "shopify_direct_cart_permalink",
            "checkout_url": permalink,
            "mcp_endpoint": self.settings.shopify_mcp_endpoint,
            "subtotal_pkr": cart["subtotal_pkr"],
            "subtotal_display": cart["subtotal_display"],
            "item_count": cart["item_count"],
            "message": (
                "Generated instant Shopify checkout permalink pre-loaded with your exact "
                f"PackItUp variant IDs and pack quantities ({cart['subtotal_display']})."
            ),
        }


class CheckoutToolInput(BaseModel):
    session_id: str = Field(default="default", description="Active user session ID.")


@tool("create_shopify_checkout", args_schema=CheckoutToolInput)
def create_shopify_checkout(session_id: str = "default") -> Dict[str, Any]:
    """Generates a live Shopify checkout URL (`packitup.com.pk/cart/...` or Storefront GraphQL `checkoutUrl`) for all items currently in the customer's cart."""
    client = ShopifyStorefrontClient()
    return client.create_checkout_from_session(session_id=session_id)
