"""Thread-safe Session Cart Manager & CSV Lead Persistence (`data/leads.csv`).

Provides unified state management shared across:
- LangGraph tool invocations (natural language prompt-to-cart & lead capture)
- Streamlit interactive UI buttons ("Add to Cart" on product cards, quantity steppers)
- FastAPI REST endpoints & WhatsApp Business webhook sessions
"""
from __future__ import annotations

import csv
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get_settings
from src.data_loader import get_catalog

_CART_LOCK = threading.Lock()
_LEAD_LOCK = threading.Lock()

# session_id -> dict of variant_id -> cart line item dict
_SESSION_CARTS: Dict[str, Dict[int, Dict[str, Any]]] = {}


class CartManager:
    """Manages per-session shopping carts with automatic pack pricing & PKR totals."""

    @staticmethod
    def get_cart(session_id: str = "default") -> Dict[str, Any]:
        with _CART_LOCK:
            items_map = _SESSION_CARTS.setdefault(session_id, {})
            items = list(items_map.values())
            subtotal_pkr = sum(float(i["line_total_pkr"]) for i in items)
            total_packs = sum(int(i["quantity_packs"]) for i in items)
            total_units = sum(int(i["quantity_packs"]) * int(i.get("pack_size_int") or 1) for i in items)
            return {
                "session_id": session_id,
                "items": items,
                "item_count": len(items),
                "total_packs": total_packs,
                "estimated_total_units": total_units,
                "subtotal_pkr": round(subtotal_pkr, 2),
                "subtotal_display": f"PKR {subtotal_pkr:,.0f}",
                "currency": "PKR",
            }

    @staticmethod
    def add_item(
        session_id: str,
        product_identifier: str,
        variant_id: Optional[int] = None,
        quantity_packs: int = 1,
        pack_size_hint: Optional[str] = None,
        color_hint: Optional[str] = None,
        natural_query: str = "",
    ) -> Dict[str, Any]:
        """Resolves product + variant and adds `quantity_packs` to the session cart."""
        catalog = get_catalog()
        product = catalog.get_product(product_identifier)
        if not product:
            return {
                "success": False,
                "error": f"Could not find a PackItUp product matching '{product_identifier}'.",
                "cart": CartManager.get_cart(session_id),
            }

        variant = catalog.resolve_variant(
            product=product,
            variant_id=variant_id,
            pack_size=pack_size_hint,
            color_or_option=color_hint,
            natural_query=natural_query,
        )

        if variant.requires_custom_quote or variant.price_pkr <= 0:
            return {
                "success": False,
                "requires_custom_quote": True,
                "product_title": product.title,
                "product_url": product.url,
                "error": (
                    f"'{product.title}' ({variant.title}) is a custom-order / B2B quote item "
                    "(listed at PKR 0.00 placeholder on the storefront) and cannot be added directly "
                    "to the retail cart. Please submit your custom specs to capture a B2B quote lead!"
                ),
                "cart": CartManager.get_cart(session_id),
            }

        qty = max(1, int(quantity_packs))
        with _CART_LOCK:
            cart_map = _SESSION_CARTS.setdefault(session_id, {})
            if variant.id in cart_map:
                cart_map[variant.id]["quantity_packs"] += qty
                cart_map[variant.id]["line_total_pkr"] = round(
                    cart_map[variant.id]["quantity_packs"] * variant.price_pkr, 2
                )
            else:
                cart_map[variant.id] = {
                    "product_id": product.id,
                    "product_handle": product.handle,
                    "product_title": product.title,
                    "product_url": product.url,
                    "image_url": product.primary_image,
                    "variant_id": variant.id,
                    "shopify_variant_gid": variant.shopify_gid,
                    "variant_title": variant.title,
                    "sku": variant.sku,
                    "pack_size_int": variant.pack_size_int or 1,
                    "unit_pack_price_pkr": variant.price_pkr,
                    "quantity_packs": qty,
                    "line_total_pkr": round(qty * variant.price_pkr, 2),
                }
            added_item = dict(cart_map[variant.id])

        return {
            "success": True,
            "added_item": added_item,
            "message": (
                f"Added {qty} pack(s) of {product.title} [{variant.title}] "
                f"@ PKR {variant.price_pkr:,.0f}/pack (Line Total: PKR {added_item['line_total_pkr']:,.0f})."
            ),
            "cart": CartManager.get_cart(session_id),
        }

    @staticmethod
    def update_quantity(session_id: str, variant_id: int, quantity_packs: int) -> Dict[str, Any]:
        with _CART_LOCK:
            cart_map = _SESSION_CARTS.setdefault(session_id, {})
            vid = int(variant_id)
            if vid not in cart_map:
                return {"success": False, "error": f"Variant {vid} not in cart.", "cart": CartManager.get_cart(session_id)}
            if quantity_packs <= 0:
                removed = cart_map.pop(vid)
                return {
                    "success": True,
                    "message": f"Removed {removed['product_title']} ({removed['variant_title']}) from cart.",
                    "cart": CartManager.get_cart(session_id),
                }
            cart_map[vid]["quantity_packs"] = int(quantity_packs)
            cart_map[vid]["line_total_pkr"] = round(
                int(quantity_packs) * float(cart_map[vid]["unit_pack_price_pkr"]), 2
            )
        return {
            "success": True,
            "message": "Updated cart quantity.",
            "cart": CartManager.get_cart(session_id),
        }

    @staticmethod
    def remove_item(session_id: str, product_or_variant: str) -> Dict[str, Any]:
        with _CART_LOCK:
            cart_map = _SESSION_CARTS.setdefault(session_id, {})
            target_vid = None
            if str(product_or_variant).isdigit() and int(product_or_variant) in cart_map:
                target_vid = int(product_or_variant)
            else:
                q_low = str(product_or_variant).lower()
                for vid, item in cart_map.items():
                    if q_low in item["product_title"].lower() or q_low in item["product_handle"].lower():
                        target_vid = vid
                        break
            if target_vid is not None:
                removed = cart_map.pop(target_vid)
                msg = f"Removed {removed['product_title']} [{removed['variant_title']}] from your cart."
                return {"success": True, "message": msg, "cart": CartManager.get_cart(session_id)}
        return {
            "success": False,
            "error": f"Item '{product_or_variant}' was not found in your cart.",
            "cart": CartManager.get_cart(session_id),
        }

    @staticmethod
    def clear_cart(session_id: str) -> Dict[str, Any]:
        with _CART_LOCK:
            _SESSION_CARTS[session_id] = {}
        return {"success": True, "message": "Cleared all items from your shopping cart.", "cart": CartManager.get_cart(session_id)}


LEAD_CSV_COLUMNS = [
    "lead_id",
    "timestamp_utc",
    "customer_name",
    "phone",
    "email",
    "company_name",
    "city",
    "inquiry_type",
    "product_interest",
    "estimated_quantity",
    "customization_details",
    "budget_pkr",
    "source_channel",
    "status",
    "notes",
]


class LeadStore:
    """Thread-safe CSV Lead Manager storing B2B, custom printing, and wholesale leads in `leads.csv`."""

    @staticmethod
    def _ensure_csv_exists(csv_path: Path) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_path.exists():
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LEAD_CSV_COLUMNS)
                writer.writeheader()

    @staticmethod
    def add_lead(
        customer_name: str,
        phone: str,
        email: str = "",
        company_name: str = "",
        city: str = "Karachi",
        inquiry_type: str = "custom_printing",
        product_interest: str = "",
        estimated_quantity: str = "",
        customization_details: str = "",
        budget_pkr: str = "",
        source_channel: str = "chat_assistant",
        notes: str = "",
    ) -> Dict[str, Any]:
        settings = get_settings()
        csv_path = settings.leads_csv_path
        with _LEAD_LOCK:
            LeadStore._ensure_csv_exists(csv_path)
            lead_id = f"LEAD-{uuid.uuid4().hex[:6].upper()}"
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            row = {
                "lead_id": lead_id,
                "timestamp_utc": now_iso,
                "customer_name": (customer_name or "Valued Customer").strip(),
                "phone": (phone or "Not provided").strip(),
                "email": (email or "").strip(),
                "company_name": (company_name or "").strip(),
                "city": (city or "Karachi").strip(),
                "inquiry_type": (inquiry_type or "b2b_inquiry").strip(),
                "product_interest": (product_interest or "Custom Packaging").strip(),
                "estimated_quantity": str(estimated_quantity or "").strip(),
                "customization_details": (customization_details or "").strip(),
                "budget_pkr": str(budget_pkr or "").strip(),
                "source_channel": (source_channel or "chat_assistant").strip(),
                "status": "New",
                "notes": (notes or "").strip(),
            }
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LEAD_CSV_COLUMNS)
                writer.writerow(row)
        return row

    @staticmethod
    def list_leads() -> List[Dict[str, Any]]:
        settings = get_settings()
        csv_path = settings.leads_csv_path
        with _LEAD_LOCK:
            LeadStore._ensure_csv_exists(csv_path)
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return list(reader)
