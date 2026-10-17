"""Industry-grade Pricing, Pack Quote & Automatic Cart Management Tools (Agent A2)."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional
from src import tool
from pydantic import BaseModel, Field

from src.data_loader import get_catalog
from src.state_store import CartManager


class PackQuoteInput(BaseModel):
    product_identifier: str = Field(
        ...,
        description="Product handle, title, or ID (e.g., '10x10x4-bakery-boxwindow', 'u-shaped-pp-plastic-cups-with-pp-sip-lids').",
    )
    desired_total_pieces_or_packs: int = Field(
        default=1,
        ge=1,
        description="Either the number of packs (e.g. 2 packs) or total unit pieces needed (e.g. 500 cups).",
    )
    is_piece_count: bool = Field(
        default=False,
        description="Set True if `desired_total_pieces_or_packs` is total individual pieces (e.g. '500 cups') rather than number of packs.",
    )
    pack_size_hint: Optional[str] = Field(
        default=None,
        description="Optional preferred pack MOQ size (e.g. '20', '50', '100', '200').",
    )
    color_or_size_hint: Optional[str] = Field(
        default=None,
        description="Optional variant option such as color ('White', 'Brown', 'Black') or size ('12oz', '16oz', '8mm').",
    )


@tool("calculate_pack_quote", args_schema=PackQuoteInput)
def calculate_pack_quote(
    product_identifier: str,
    desired_total_pieces_or_packs: int = 1,
    is_piece_count: bool = False,
    pack_size_hint: Optional[str] = None,
    color_or_size_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculates an exact, deterministic PKR price quote from structured variant records (Agent A2).

    Guardrail: PackItUp sells in packs (MOQ 20, 25, 50, 100, etc.). This tool resolves the exact
    variant, checks for PKR 0.00 custom-quote placeholders, and computes exact pack multiples.
    """
    catalog = get_catalog()
    prod = catalog.get_product(product_identifier)
    if not prod:
        return {"found": False, "error": f"Product '{product_identifier}' not found."}

    variant = catalog.resolve_variant(
        product=prod,
        pack_size=pack_size_hint,
        color_or_option=color_or_size_hint,
        natural_query=f"{pack_size_hint or ''} {color_or_size_hint or ''}",
    )

    if variant.requires_custom_quote or variant.price_pkr <= 0:
        return {
            "found": True,
            "requires_custom_quote": True,
            "product_title": prod.title,
            "variant_title": variant.title,
            "source_url": prod.url,
            "message": (
                f"'{prod.title}' ({variant.title}) is listed as PKR 0.00 because it requires a custom "
                "B2B quote based on artwork/quantity, or that specific size is currently out of stock. "
                "Please offer to capture a custom order lead or contact +92 322 2620169."
            ),
        }

    pack_units = variant.pack_size_int or 1
    if is_piece_count and pack_units > 1:
        packs_needed = max(1, math.ceil(desired_total_pieces_or_packs / pack_units))
    else:
        packs_needed = max(1, desired_total_pieces_or_packs)

    total_pieces = packs_needed * pack_units
    total_pkr = round(packs_needed * variant.price_pkr, 2)
    per_piece_pkr = round(total_pkr / max(total_pieces, 1), 2)

    return {
        "found": True,
        "requires_custom_quote": False,
        "product_id": prod.id,
        "product_handle": prod.handle,
        "product_title": prod.title,
        "variant_id": variant.id,
        "variant_title": variant.title,
        "pack_size_units": pack_units,
        "unit_pack_price_pkr": variant.price_pkr,
        "packs_quoted": packs_needed,
        "total_pieces": total_pieces,
        "total_price_pkr": total_pkr,
        "total_price_display": f"PKR {total_pkr:,.0f}",
        "effective_per_piece_pkr": per_piece_pkr,
        "available": variant.available,
        "source_url": prod.url,
        "all_variants": [v.to_dict() for v in prod.variants],
    }


class AddToCartInput(BaseModel):
    session_id: str = Field(default="default", description="Active user session ID.")
    product_identifier: str = Field(
        ...,
        description="Product handle, title, or numeric ID to add to the shopping cart.",
    )
    quantity_packs: int = Field(
        default=1,
        ge=1,
        description="Number of packs to add to the cart.",
    )
    variant_id: Optional[int] = Field(
        default=None,
        description="Optional exact Shopify variant ID if known.",
    )
    pack_size_hint: Optional[str] = Field(
        default=None,
        description="Optional pack MOQ size requested by user (e.g. '20', '50', '100', '200').",
    )
    color_or_option_hint: Optional[str] = Field(
        default=None,
        description="Optional color or size option requested by user (e.g. 'White', 'Brown', '12oz').",
    )


@tool("add_to_cart_tool", args_schema=AddToCartInput)
def add_to_cart_tool(
    product_identifier: str,
    quantity_packs: int = 1,
    session_id: str = "default",
    variant_id: Optional[int] = None,
    pack_size_hint: Optional[str] = None,
    color_or_option_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Automatically adds a product (and its resolved variant/pack size) to the customer's live shopping cart.

    Invoke this tool immediately whenever the user prompts to add one or more items to their cart
    (e.g., 'Add 2 packs of the 10x10x4 white bakery box (50 pack) to my cart').
    """
    return CartManager.add_item(
        session_id=session_id,
        product_identifier=product_identifier,
        variant_id=variant_id,
        quantity_packs=quantity_packs,
        pack_size_hint=pack_size_hint,
        color_hint=color_or_option_hint,
        natural_query=f"{pack_size_hint or ''} {color_or_option_hint or ''}",
    )


class RemoveFromCartInput(BaseModel):
    session_id: str = Field(default="default", description="Active user session ID.")
    product_or_variant: str = Field(
        ...,
        description="Product title, handle, or variant ID to remove from the cart.",
    )


@tool("remove_from_cart_tool", args_schema=RemoveFromCartInput)
def remove_from_cart_tool(product_or_variant: str, session_id: str = "default") -> Dict[str, Any]:
    """Removes a specified product or variant from the customer's shopping cart."""
    return CartManager.remove_item(session_id=session_id, product_or_variant=product_or_variant)


class ViewCartInput(BaseModel):
    session_id: str = Field(default="default", description="Active user session ID.")


@tool("view_cart_tool", args_schema=ViewCartInput)
def view_cart_tool(session_id: str = "default") -> Dict[str, Any]:
    """Returns the current contents, pack counts, and PKR subtotal of the user's shopping cart."""
    return CartManager.get_cart(session_id=session_id)


@tool("clear_cart_tool", args_schema=ViewCartInput)
def clear_cart_tool(session_id: str = "default") -> Dict[str, Any]:
    """Empties all items from the user's shopping cart."""
    return CartManager.clear_cart(session_id=session_id)
