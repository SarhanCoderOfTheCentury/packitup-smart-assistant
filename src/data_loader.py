"""Deep dataset loader & data-hygiene enricher for the PackItUp.com.pk corpus.

Addresses all 5 data-quality anomalies identified during the store crawl:
1. 73 of 187 products have an empty `category` (`product_type`). We cross-reference
   `collection_product_map.json` (56 collections) to assign accurate categories &
   collection memberships to every product.
2. 6 products have `PKR 0.00` variants (5 completely zero-priced custom/placeholder
   products: `serving-tray`, `hang-tags`, `kraft-paper-2-cup-holder`,
   `thankyou-compliment-cards`, `visiting-cards`, plus 3 16oz variants of
   `u-shaped-pp-plastic-cups-with-pp-sip-lids`). We mark zero-price variants as
   `requires_custom_quote=True` so the agent never quotes PKR 0.00 as free and
   instead routes those items to B2B Lead Capture (`leads.csv`).
3. Variant option names contain typos (`Quanity`, `Qunatity`, `quantity`, `MOQ`,
   `Sizes`, `sizes`, `Colour`). We normalize them for reliable variant matching.
4. No SKUs are populated (`sku: null`). We synthesize deterministic display codes
   (`PIU-<product_id_suffix>-<variant_id_suffix>`) alongside native Shopify variant GIDs.
5. Three policy pages (`terms-of-service`, `payment-policy`, `contact-information`)
   returned HTTP 404 on the live site. We preserve explicit 404 flags so Agent A4
   never hallucinates unpublished legal/payment policies.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import get_settings

OPTION_NAME_CANONICAL_MAP = {
    "moq": "Pack Size (MOQ)",
    "quantity": "Pack Size (MOQ)",
    "quanity": "Pack Size (MOQ)",
    "qunatity": "Pack Size (MOQ)",
    "size": "Size",
    "sizes": "Size",
    "straw size": "Size",
    "color": "Color",
    "colour": "Color",
    "printing": "Printing",
    "variant": "Style",
    "title": "Option",
}


def normalize_option_name(raw_name: Optional[str]) -> str:
    if not raw_name:
        return "Option"
    cleaned = raw_name.strip().lower()
    return OPTION_NAME_CANONICAL_MAP.get(cleaned, raw_name.strip())


@dataclass
class EnrichedVariant:
    id: int
    title: str
    price_pkr: float
    available: bool
    option1: Optional[str] = None
    option2: Optional[str] = None
    option3: Optional[str] = None
    sku: str = ""
    requires_custom_quote: bool = False
    pack_size_int: Optional[int] = None

    @property
    def shopify_gid(self) -> str:
        return f"gid://shopify/ProductVariant/{self.id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "shopify_gid": self.shopify_gid,
            "title": self.title,
            "price_pkr": self.price_pkr,
            "available": self.available,
            "option1": self.option1,
            "option2": self.option2,
            "option3": self.option3,
            "sku": self.sku,
            "requires_custom_quote": self.requires_custom_quote,
            "pack_size_int": self.pack_size_int,
        }


@dataclass
class EnrichedProduct:
    id: int
    title: str
    handle: str
    url: str
    vendor: str
    raw_category: Optional[str]
    enriched_category: str
    collections: List[str]
    collection_titles: List[str]
    tags: List[str]
    price_min_pkr: float
    price_max_pkr: float
    currency: str
    in_stock: bool
    variant_count: int
    available_variant_count: int
    options: List[Dict[str, Any]]
    images: List[str]
    description_text: str
    variants: List[EnrichedVariant]
    requires_custom_quote: bool = False

    @property
    def primary_image(self) -> str:
        if self.images:
            return self.images[0]
        return "https://cdn.shopify.com/s/files/1/0655/1915/7408/files/packitup_placeholder.png"

    @property
    def price_display(self) -> str:
        if self.requires_custom_quote and self.price_max_pkr <= 0:
            return "Custom Quote (B2B / Made-to-Order)"
        if self.price_min_pkr == self.price_max_pkr:
            return f"PKR {self.price_min_pkr:,.0f}"
        return f"PKR {self.price_min_pkr:,.0f} – PKR {self.price_max_pkr:,.0f}"

    def to_card_dict(self, selected_variant: Optional[EnrichedVariant] = None) -> Dict[str, Any]:
        """Serializes product into a UI-ready ProductCard dictionary."""
        priced_variants = [v for v in self.variants if v.price_pkr > 0 and v.available]
        if not priced_variants:
            priced_variants = [v for v in self.variants if v.price_pkr > 0] or self.variants

        default_var = selected_variant or (priced_variants[0] if priced_variants else None)
        short_desc = re.sub(r"\s+", " ", self.description_text).strip()
        if len(short_desc) > 220:
            short_desc = short_desc[:217].rsplit(" ", 1)[0] + "..."

        return {
            "id": self.id,
            "handle": self.handle,
            "title": self.title,
            "url": self.url,
            "category": self.enriched_category,
            "collections": self.collection_titles,
            "price_min_pkr": self.price_min_pkr,
            "price_max_pkr": self.price_max_pkr,
            "price_display": self.price_display,
            "in_stock": self.in_stock,
            "requires_custom_quote": self.requires_custom_quote,
            "image_url": self.primary_image,
            "images": self.images[:4],
            "short_description": short_desc,
            "full_description": self.description_text,
            "tags": self.tags[:8],
            "options": self.options,
            "selected_variant": default_var.to_dict() if default_var else None,
            "variants": [v.to_dict() for v in self.variants],
        }


def _extract_pack_size(variant_title: str, opt1: Optional[str], opt2: Optional[str], opt3: Optional[str]) -> Optional[int]:
    """Extracts integer pack quantity (MOQ) from variant options when present."""
    for candidate in (opt1, opt2, opt3):
        if candidate and candidate.strip().isdigit():
            val = int(candidate.strip())
            if val in (5, 10, 12, 20, 24, 25, 30, 48, 50, 95, 100, 200, 250, 300, 500, 1000, 2000, 5000):
                return val
    # Check patterns like "50 pcs" or "12oz / 50"
    m = re.search(r"\b(10|20|25|50|100|200|250|500|1000)\b", variant_title or "")
    if m:
        return int(m.group(1))
    return None


class PackItUpCatalog:
    """In-memory indexed store of all PackItUp products, collections, policies, pages, and chunks."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.data_dir = Path(data_dir or settings.data_dir)
        self.business_facts: Dict[str, Any] = {}
        self.collection_map: Dict[str, Dict[str, Any]] = {}
        self.collections_pages: List[Dict[str, Any]] = []
        self.policies: List[Dict[str, Any]] = []
        self.pages: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []
        self.products_by_handle: Dict[str, EnrichedProduct] = {}
        self.products_by_id: Dict[int, EnrichedProduct] = {}
        self.variants_by_id: Dict[int, Tuple[EnrichedProduct, EnrichedVariant]] = {}

        self._load_all()

    def _load_all(self) -> None:
        bf_path = self.data_dir / "business_facts.json"
        if bf_path.exists():
            self.business_facts = json.loads(bf_path.read_text(encoding="utf-8"))

        cmap_path = self.data_dir / "collection_product_map.json"
        if cmap_path.exists():
            self.collection_map = json.loads(cmap_path.read_text(encoding="utf-8"))

        cpages_path = self.data_dir / "collections_pages.json"
        if cpages_path.exists():
            self.collections_pages = json.loads(cpages_path.read_text(encoding="utf-8"))

        pol_path = self.data_dir / "policies.json"
        if pol_path.exists():
            self.policies = json.loads(pol_path.read_text(encoding="utf-8"))

        pages_path = self.data_dir / "pages.json"
        if pages_path.exists():
            self.pages = json.loads(pages_path.read_text(encoding="utf-8"))

        # Build reverse index: product_handle -> list of (collection_handle, collection_title)
        handle_to_collections: Dict[str, List[Tuple[str, str]]] = {}
        for col_handle, col_info in self.collection_map.items():
            if col_handle == "frontpage":
                continue
            col_title = col_info.get("title", col_handle)
            for prod_handle in col_info.get("product_handles", []):
                handle_to_collections.setdefault(prod_handle, []).append((col_handle, col_title))

        prods_path = self.data_dir / "products.json"
        raw_products = json.loads(prods_path.read_text(encoding="utf-8")) if prods_path.exists() else []

        for rp in raw_products:
            handle = rp["handle"]
            pid = int(rp["id"])
            col_pairs = handle_to_collections.get(handle, [])
            col_handles = [c[0] for c in col_pairs]
            col_titles = [c[1] for c in col_pairs]

            raw_cat = rp.get("category")
            if raw_cat and str(raw_cat).strip() and str(raw_cat).lower() != "nan":
                enriched_cat = str(raw_cat).strip()
                if col_titles and enriched_cat == "Packaging":
                    # Make category more specific using primary collection title
                    enriched_cat = f"Packaging — {col_titles[0]}"
            elif col_titles:
                enriched_cat = col_titles[0]
            else:
                enriched_cat = "General Packaging"

            # Normalize options
            norm_options = []
            for opt in rp.get("options", []):
                norm_options.append({
                    "name": normalize_option_name(opt.get("name")),
                    "raw_name": opt.get("name"),
                    "position": opt.get("position"),
                    "values": opt.get("values", []),
                })

            # Process variants
            enriched_variants: List[EnrichedVariant] = []
            positive_prices: List[float] = []
            for rv in rp.get("variants", []):
                vid = int(rv["id"])
                vprice = float(rv.get("price_pkr") or 0.0)
                vavail = bool(rv.get("available", False))
                req_quote = vprice <= 0.0
                if vprice > 0.0:
                    positive_prices.append(vprice)

                sku_val = rv.get("sku")
                if not sku_val:
                    sku_val = f"PIU-{str(pid)[-4:]}-{str(vid)[-4:]}"

                pack_sz = _extract_pack_size(
                    rv.get("title", ""),
                    rv.get("option1"),
                    rv.get("option2"),
                    rv.get("option3"),
                )
                ev = EnrichedVariant(
                    id=vid,
                    title=rv.get("title", "Default"),
                    price_pkr=vprice,
                    available=vavail and not req_quote,
                    option1=rv.get("option1"),
                    option2=rv.get("option2"),
                    option3=rv.get("option3"),
                    sku=sku_val,
                    requires_custom_quote=req_quote,
                    pack_size_int=pack_sz,
                )
                enriched_variants.append(ev)

            # Compute accurate positive price bounds
            price_min = min(positive_prices) if positive_prices else 0.0
            price_max = max(positive_prices) if positive_prices else 0.0
            prod_requires_quote = len(positive_prices) == 0 or "custom" in handle.lower()

            ep = EnrichedProduct(
                id=pid,
                title=rp.get("title", ""),
                handle=handle,
                url=rp.get("url", f"https://packitup.com.pk/products/{handle}"),
                vendor=rp.get("vendor", "Pack it up"),
                raw_category=raw_cat,
                enriched_category=enriched_cat,
                collections=col_handles,
                collection_titles=col_titles,
                tags=rp.get("tags", []),
                price_min_pkr=price_min,
                price_max_pkr=price_max,
                currency=rp.get("currency", "PKR"),
                in_stock=bool(rp.get("in_stock", False)) and len(positive_prices) > 0,
                variant_count=len(enriched_variants),
                available_variant_count=sum(1 for v in enriched_variants if v.available),
                options=norm_options,
                images=rp.get("images", []),
                description_text=rp.get("description_text", ""),
                variants=enriched_variants,
                requires_custom_quote=prod_requires_quote,
            )
            self.products_by_handle[handle] = ep
            self.products_by_id[pid] = ep
            for ev in enriched_variants:
                self.variants_by_id[ev.id] = (ep, ev)

        # Load & enrich chunks.jsonl
        chunks_path = self.data_dir / "chunks.jsonl"
        if chunks_path.exists():
            for line in chunks_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                c = json.loads(line)
                meta = c.get("metadata", {})
                if meta.get("type") == "product":
                    pid = meta.get("product_id")
                    prod = self.products_by_id.get(int(pid)) if pid else None
                    if prod:
                        meta["category"] = prod.enriched_category
                        meta["collections"] = ", ".join(prod.collection_titles)
                        meta["price_min_pkr"] = prod.price_min_pkr
                        meta["price_max_pkr"] = prod.price_max_pkr
                        meta["in_stock"] = prod.in_stock
                        meta["requires_custom_quote"] = prod.requires_custom_quote
                        meta["handle"] = prod.handle
                self.chunks.append(c)

    def list_categories(self) -> List[str]:
        cats = sorted({p.enriched_category for p in self.products_by_handle.values()})
        return cats

    def get_product(self, identifier: str) -> Optional[EnrichedProduct]:
        """Finds a product by handle, numeric ID, URL, or fuzzy title match."""
        if not identifier:
            return None
        ident = str(identifier).strip()
        # 1. Exact handle
        if ident in self.products_by_handle:
            return self.products_by_handle[ident]
        # 2. URL suffix
        if "/products/" in ident:
            slug = ident.split("/products/")[-1].strip("/").split("?")[0]
            if slug in self.products_by_handle:
                return self.products_by_handle[slug]
        # 3. Numeric ID
        if ident.isdigit() and int(ident) in self.products_by_id:
            return self.products_by_id[int(ident)]
        # 4. Case-insensitive title or handle substring match
        ident_low = ident.lower()
        for p in self.products_by_handle.values():
            if p.title.lower() == ident_low or p.handle.lower() == ident_low:
                return p
        # 5. Token overlap match
        query_tokens = [t for t in re.split(r"[^a-z0-9]+", ident_low) if len(t) > 1]
        best_prod = None
        best_score = 0.0
        for p in self.products_by_handle.values():
            title_clean = p.title.lower().replace('"', '').replace("'", "")
            hay = f"{title_clean} {p.handle} {' '.join(p.tags)}".lower()
            matches = sum((2.0 if any(c.isdigit() for c in t) else 1.0) for t in query_tokens if t in hay)
            title_matches = sum((2.5 if any(c.isdigit() for c in t) else 1.0) for t in query_tokens if t in title_clean or t in p.handle.lower())
            denom = sum((2.0 if any(c.isdigit() for c in t) else 1.0) for t in query_tokens) or 1.0
            score = (matches + title_matches) / (denom * 2.0)
            if ident_low.replace('"', '') in title_clean:
                score += 0.6
            if score > best_score and score >= 0.35:
                best_score = score
                best_prod = p
        return best_prod

    def resolve_variant(
        self,
        product: EnrichedProduct,
        variant_id: Optional[int] = None,
        pack_size: Optional[str] = None,
        color_or_option: Optional[str] = None,
        natural_query: str = "",
    ) -> EnrichedVariant:
        """Deterministically resolves the best matching variant for a product."""
        if variant_id and int(variant_id) in self.variants_by_id:
            prod, var = self.variants_by_id[int(variant_id)]
            if prod.id == product.id:
                return var

        candidates = [v for v in product.variants if not v.requires_custom_quote and v.available]
        if not candidates:
            candidates = [v for v in product.variants if not v.requires_custom_quote] or product.variants

        combined_hint = f"{pack_size or ''} {color_or_option or ''} {natural_query or ''}".lower().strip()
        if not combined_hint:
            return candidates[0]

        best_var = candidates[0]
        best_score = -1
        for v in candidates:
            score = 0
            v_tokens = [
                str(v.option1 or "").lower(),
                str(v.option2 or "").lower(),
                str(v.option3 or "").lower(),
                v.title.lower(),
            ]
            # Exact pack size match
            if pack_size and str(pack_size).lower() in v_tokens:
                score += 10
            elif v.pack_size_int and re.search(rf"\b{v.pack_size_int}\b", combined_hint):
                score += 8

            # Color or secondary option match
            for opt_val in (v.option1, v.option2, v.option3):
                if opt_val and opt_val.lower() in combined_hint:
                    score += 6

            if score > best_score:
                best_score = score
                best_var = v

        return best_var


@lru_cache(maxsize=1)
def get_catalog() -> PackItUpCatalog:
    """Singleton accessor for the enriched PackItUp catalog."""
    return PackItUpCatalog()
