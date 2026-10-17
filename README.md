# 📦 PackItUp.com.pk — Multi-Agent AI Smart Shopping Assistant & Storefront Automation

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent%20StateGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Hybrid%20RAG-green.svg)](https://www.trychroma.com/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20%2F%203.1%20Flash--Lite-blueviolet.svg)](https://ai.google.dev/)
[![Shopify](https://img.shields.io/badge/E--Commerce-Shopify%20Storefront-96BF48.svg)](https://packitup.com.pk/)
[![WhatsApp](https://img.shields.io/badge/Messaging-WhatsApp%20Cloud%20API%20v21.0-25D366.svg)](https://developers.facebook.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, full-stack **Multi-Agent Conversational AI Assistant** engineered specifically for **[PackItUp.com.pk](https://packitup.com.pk/)**, Karachi's leading B2B/B2C Shopify store specializing in food-service packaging, bakery & cake boxes, café cups, pouches, and custom-branded retail packaging.

---

## 📑 Table of Contents

- [Architectural Highlights](#-architectural-highlights)
- [Multi-Agent Topology (LangGraph)](#-multi-agent-topology-langgraph)
- [Dataset Hygiene & Ingestion Pipeline](#-dataset-hygiene--ingestion-pipeline)
- [Hybrid RAG Engine (ChromaDB)](#-hybrid-rag-engine-chromadb)
- [Key Features & Capabilities](#-key-features--capabilities)
- [Architecture Diagrams](#-architecture-diagrams)
- [Repository Structure](#-repository-structure)
- [Local Quickstart](#-local-quickstart)
- [Deploying to Streamlit Community Cloud](#-deploying-to-streamlit-community-cloud)
- [FastAPI Microservice & WhatsApp Cloud API](#-fastapi-microservice--whatsapp-cloud-api)
- [Automated Verification Suite](#-automated-verification-suite)
- [Author & Acknowledgements](#-author--acknowledgements)

---

## 🌟 Architectural Highlights

- **Sub-Second Latency**: Optimized routing via LangGraph supervisor with deterministic fast-path heuristics and local hybrid RAG (BM25 + Chroma dense embeddings) yielding response times under **1 second**.
- **Snug & Polished Chatbot UI**: Native messenger feel featuring an auto-scrolling message container, compact 4-column product cards, synchronized image aspect ratios, and zero wasted vertical space.
- **Natural-Language Prompt-to-Cart**: Directly adds items to the session cart from natural conversational requests (e.g. *"Add 2 packs of 10x10x4 white bakery box (50 pack) to my cart"*).
- **Shopify Checkout Bridge**: Generates instant Shopify checkout permalinks (`/cart/{variant_id}:{qty}`) and supports GraphQL Storefront `cartCreate` mutations.
- **WhatsApp Cloud API Ready**: Full bi-directional webhook handler generating interactive WhatsApp v21.0 messages with CTA buttons and catalog lists.
- **B2B Lead Engine**: Automatic qualification and CSV logging of wholesale and custom logo printing inquiries into `data/leads.csv`.

---

## 🤖 Multi-Agent Topology (LangGraph)

The system is coordinated by a stateful supervisor orchestrating six specialist agents:

```
                      ┌───────────────────────────────────────┐
                      │        node_a0_supervisor             │
                      │  (Intent Classifier & Slot Extractor) │
                      └──────────────────┬────────────────────┘
                                         │
        ┌───────────────┬────────────────┼────────────────┬───────────────┐
        ▼               ▼                ▼                ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│A1: Discovery │ │ A2: Pricing  │ │A3: Collection│ │ A4: Policy   │ │A5: B2B Leads │
│  & Search    │ │   & Cart     │ │  Navigator   │ │ & Compliance │ │& Custom Print│
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        │               │                │                │               │
        └───────────────┴────────────────┼────────────────┴───────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │      A6: Shopify & WhatsApp Bridge    │
                      │  (Checkout Permalinks & Outbound API) │
                      └──────────────────┬────────────────────┘
                                         ▼
                      ┌───────────────────────────────────────┐
                      │          node_synthesizer             │
                      │  (Clean GitHub Markdown Formatting)   │
                      └───────────────────────────────────────┘
```

| Agent Node | Primary Responsibility | Backing Tools & Services |
| :--- | :--- | :--- |
| **`A0_supervisor`** | Analyzes user queries, extracts product dimensions, MOQs, colors, and dispatches parallel specialist sub-graphs. | Heuristic keyword matcher & Gemini structured output |
| **`A1_discovery`** | Product catalog search with CDN images, variant matrices, and live pricing. | `search_product_catalog` + ChromaDB |
| **`A2_pricing_cart`**| MOQ pack quote calculation, volume tier pricing, and instant session cart mutation. | `get_pack_quote`, `add_to_cart_tool`, `view_cart_tool` |
| **`A3_collection`** | Browses the 56 PackItUp collection hubs (Bakery, Café, Stand-up Pouches, etc.). | `get_collection_details`, `list_collections` |
| **`A4_policy`** | Queries shipping timelines (Karachi express, nationwide), return/refund rules, and flags 404 policies. | ChromaDB partitioned on `type="policy"` |
| **`A5_leads`** | Captures B2B custom printing specifications and writes qualified leads. | `log_b2b_lead` (`data/leads.csv`) |
| **`A6_checkout`** | Converts active shopping carts into Shopify web checkout permalinks. | `ShopifyStorefrontClient` |

---

## 📊 Dataset Hygiene & Ingestion Pipeline

The assistant ingests and validates the complete PackItUp e-commerce catalog:

- **187 Products & 1,131 Priced Variants**: Preserves all SKU options, dimensions (`10"x10"x4"`, `12oz`, `200ml`, `8mm`), and pack sizes (`20`, `25`, `50`, `100`, `200`, `500`).
- **73 Uncategorized Products Restored**: Cross-referenced against 56 Shopify collection mappings (`collection_product_map.json`) to eliminate orphaned items.
- **Guardrails for PKR 0.00 Anomalies**: Identifies products lacking public prices (`Serving Tray`, `Hang Tags`, `Thankyou Cards`) and marks them as `requires_custom_quote=True`, preventing 0 PKR cart errors.
- **MOQ & Option Normalization**: Normalizes disparate vendor field keys (`Quanity`, `Qunatity`, `MOQ`) into clean canonical fields.

---

## 🔍 Hybrid RAG Engine (ChromaDB)

- **Vector Partitioning**: 451 text chunks partitioned across 4 distinct domains (`product`, `collection`, `page`, `policy`).
- **Reciprocal Rank Fusion (RRF)**: Combines dense vector similarity with BM25 lexical keyword matching and exact dimension extraction.
- **Cloud-Ready Fallback**: Includes deterministic local embedding indexing ensuring 100% catalog coverage even when external embedding quotas are limited.

---

## 🖼️ Architecture Diagrams

All architectural design blueprints are stored in `docs/diagrams/`:

1. **`01_multi_agent_langgraph_architecture.png`**: Multi-agent StateGraph flow, state schemas, and parallel execution.
2. **`02_chroma_rag_hybrid_pipeline.png`**: Ingestion, enrichment, chunking, and dual-retrieval pipeline.
3. **`03_whatsapp_shopify_end_to_end_automation.png`**: WhatsApp Cloud API v21.0 webhook lifecycle and Shopify permalink generation.
4. **`04_cloud_deployment_topology.png`**: Dual-deployment model on Streamlit Community Cloud and containerized FastAPI on Render/Cloud Run.

---

## 📁 Repository Structure

```
packitup_smart_assistant/
├── .streamlit/
│   ├── config.toml               # Streamlit theme & layout configuration
│   └── secrets.toml.example      # Example secrets for Cloud deployment
├── data/
│   ├── leads.csv                 # Live CRM lead log
│   └── packitup/                 # Ingested datasets (187 products, chunks, collections)
│       ├── products.csv
│       ├── products.json
│       ├── chunks.jsonl
│       └── collection_product_map.json
├── docs/
│   └── diagrams/                 # High-resolution architectural blueprints
├── frontend/
│   └── app.py                    # Streamlit Storefront & Multi-Agent UI
├── src/
│   ├── agents/                   # LangGraph StateGraph, agents & prompt definitions
│   │   ├── graph.py
│   │   ├── prompts.py
│   │   └── state.py
│   ├── backend/                  # FastAPI REST API & WhatsApp Webhook
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── whatsapp_webhook.py
│   ├── tools/                    # Specialist agent toolkits
│   │   ├── cart_tools.py
│   │   ├── catalog_tools.py
│   │   ├── lead_tools.py
│   │   ├── policy_faq_tools.py
│   │   └── shopify_adapter.py
│   ├── config.py                 # Pydantic Settings & environment validation
│   ├── data_loader.py            # Dataset hygiene, parsing & catalog models
│   ├── rag_chroma.py             # ChromaDB vector store & Hybrid RAG search
│   └── state_store.py            # In-memory Cart & Lead CRM store
├── tests/
│   └── test_smoke.py             # Automated end-to-end verification suite
├── .gitignore                    # Safe gitignore protecting .env and cache
├── Dockerfile                    # Container definition for FastAPI microservice
├── docker-compose.yml            # Multi-container orchestration
├── render.yaml                   # Infrastructure-as-code for Render deployment
└── requirements.txt              # Production Python dependencies
```

---

## ⚡ Local Quickstart

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/SarhanCoderOfTheCentury/packitup-smart-assistant.git
cd packitup-smart-assistant

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```ini
GOOGLE_API_KEY="AIzaSyYourGeminiApiKeyHere"
GEMINI_MODEL="gemini-3.1-flash-lite"
AGENT_EXECUTION_MODE="direct"
```

### 3. Run the Automated Test Suite

Verify catalog loading, hybrid RAG retrieval, prompt-to-cart automation, and WhatsApp payload generation:

```bash
python tests/test_smoke.py
```

### 4. Run the Streamlit Storefront

```bash
streamlit run frontend/app.py
```

Open your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud

The codebase is pre-configured with `pysqlite3-binary` and a SQLite3 shim in `frontend/app.py` for out-of-the-box Debian compatibility on Streamlit Cloud:

1. Fork or push this repository to your GitHub account.
2. Sign in to **[share.streamlit.io](https://share.streamlit.io/)**.
3. Click **New app** and specify:
   - **Repository**: `SarhanCoderOfTheCentury/packitup-smart-assistant`
   - **Branch**: `main`
   - **Main file path**: `frontend/app.py`
4. Expand **Advanced settings → Secrets** and provide:
   ```toml
   GOOGLE_API_KEY = "AIzaSyYourGeminiApiKeyHere"
   GEMINI_API_KEY = "AIzaSyYourGeminiApiKeyHere"
   GEMINI_MODEL = "gemini-3.1-flash-lite"
   AGENT_EXECUTION_MODE = "direct"
   ```
5. Click **Deploy!** Your app will be live within 2 minutes.

---

## 🚀 FastAPI Microservice & WhatsApp Cloud API

### Run the FastAPI Backend Locally

```bash
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI docs will be available at `http://localhost:8000/docs`.

### Meta WhatsApp Business Cloud API Integration

1. Configure Webhook URL in Meta App Dashboard:
   `https://<your-domain>/api/v1/whatsapp/webhook`
2. Set your verification token in `.env`:
   `WHATSAPP_VERIFY_TOKEN="your_secure_verify_token"`
3. The webhook automatically maps incoming customer WhatsApp chats into LangGraph agent turns and returns interactive button and list payloads compliant with WhatsApp Cloud API v21.0.

---

## 🧪 Automated Verification Suite

Run the full smoke test suite covering all operational layers:

```bash
python tests/test_smoke.py
```

Verification covers:
1. **Catalog Sanitization**: 187 products, 1,131 variants, 56 collections, 451 chunks with 0 uncategorized items.
2. **Hybrid RAG Precision**: Retrieval on complex dimensional queries (`10x10x4 bakery cake box with window`).
3. **Multi-Agent Prompt-to-Cart**: Parsing natural-language prompts and calculating line totals.
4. **Multi-Domain Fan-Out**: Multi-agent dispatch across discovery, pricing, and shipping policies.
5. **B2B Lead Capture**: Writing valid lead records to `data/leads.csv`.
6. **WhatsApp Cloud API Payload Validation**: Interactive message payload structure conformity.

---

## 👤 Author & Acknowledgements

- **Developer**: Sarhan Ahmed ([@SarhanCoderOfTheCentury](https://github.com/SarhanCoderOfTheCentury) / [@SarhanDev-SE](https://github.com/SarhanDev-SE))
- **Platform**: [PackItUp.com.pk](https://packitup.com.pk/) (Karachi, Pakistan)
- **Frameworks**: LangGraph, LangChain, Streamlit, ChromaDB, Google Gemini.
