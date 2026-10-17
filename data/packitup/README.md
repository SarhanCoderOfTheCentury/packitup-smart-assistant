# PackItUp.com.pk — Vector-DB-Ready Dataset

Scraped from https://packitup.com.pk/ (Shopify storefront) on 2026-10-17.

## Files

| File | Rows | What it is |
|---|---|---|
| `products.json` | 187 | Full product records: id, title, handle, url, category, tags, price range (PKR), stock, SKUs, options, images, description, **all 1,131 priced variants** |
| `products.csv` | 187 | Flat catalog for spreadsheets / bulk import |
| `policies.json` | 6 | Shipping, Refund, Privacy (scraped) + Terms of Service, Payment, Contact Information (marked NOT PUBLISHED / 404) |
| `pages.json` | 25 | Site pages with extracted text |
| `collections_pages.json` | 56 | Collection hubs with extracted text |
| `collection_product_map.json` | 56 | Collection → member product handles |
| `business_facts.json` | 1 | Store name, address, phone, email, socials, MCP endpoint |
| `chunks.jsonl` | 451 | **Embedding-ready chunks**: `{id, text, metadata}` |
| `corpus.md` | 1 | Entire site as one flat Markdown document |
| `architecture.md` | 1 | Multi-agent architecture + Mermaid diagram + vector DB schema |

## chunks.jsonl schema

```jsonc
{
  "id": "product-<handle>-<n>",
  "text": "PRODUCT: <title>\nCategory: ...\nPrice: PKR ...\nStock: ...\nSKU: ...\nURL: ...\nTags: ...\nVariants: ...\n\nDescription:\n...",
  "metadata": {
    "source_url": "https://packitup.com.pk/products/<handle>",
    "type": "product | collection | page | policy",
    "title": "...",
    "category": "...",
    "price_min_pkr": 1500.0,
    "price_max_pkr": 52500.0,
    "in_stock": true,
    "skus": [],
    "tags": [],
    "images": [],
    "product_id": 15874142797984,
    "chunk_index": 0
  }
}
```

**Chunking:** ~1,800 characters (≈450–500 tokens), 250-character overlap, split on newline boundaries.
Each chunk is prefixed with a **structured header** (title, category, price, stock, SKU, URL) so price and stock
survive retrieval instead of having to be inferred from prose.

## Loading

```python
from langchain_community.document_loaders import JSONLoader
docs = JSONLoader("chunks.jsonl", jq_schema=".", text_content=False, json_lines=True).load()
# then embed doc.page_content and upsert with doc.metadata
```

Recommended production layout: four namespaces — `products` (244 chunks), `collections` (195),
`pages` (7), `policies` (5) — matching the agent boundaries in `architecture.md`.

## Data-quality flags found during the crawl

1. **73 of 187 products have an empty `product_type`** — they will not appear in category-filtered search.
2. **One product is priced at PKR 0.00** — likely a placeholder that must not be quoted.
3. **No SKUs are populated** on any variant (all `sku: null`).
4. **`/policies/terms-of-service` and `/policies/contact-information` return HTTP 404.**
5. Many `/pages/*` landing pages render empty shells (theme-driven) — their content is not in the HTML.
