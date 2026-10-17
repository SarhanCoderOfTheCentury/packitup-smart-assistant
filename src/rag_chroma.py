"""Local ChromaDB Vector Store & Hybrid Retrieval Engine for PackItUp.

Uses `langchain-chroma` and `GoogleGenerativeAIEmbeddings` (`models/gemini-embedding-001`)
from `langchain-google-genai`, with Reciprocal Rank Fusion (RRF) combining:
1. Dense semantic vector similarity in ChromaDB (partitioned by `type` metadata:
   `product`, `collection`, `page`, `policy`)
2. Lexical BM25-style keyword & dimension matching (e.g., `10x10x4`, `12oz`, `200ml`)
3. Pre-filtering on structured metadata (`in_stock`, `price_min_pkr`, `category`).

Also supports deterministic local hash/lexical fallback when `GOOGLE_API_KEY` is not
yet configured, ensuring the demo UI and automated tests run seamlessly offline or online.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src import Document, Embeddings

from src.config import _get_secret_or_env, get_settings
from src.data_loader import PackItUpCatalog, get_catalog


class DeterministicHashEmbeddings(Embeddings):
    """Fallback local n-gram hash embedding model when GOOGLE_API_KEY is absent."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


def get_embeddings_model() -> Tuple[Embeddings, str]:
    """Returns `(Embeddings, backend_name)` using fast local embeddings or Gemini when configured."""
    settings = get_settings()
    use_remote = _get_secret_or_env("USE_GEMINI_EMBEDDINGS", "false").lower() in ("true", "1")
    if use_remote and settings.has_gemini_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            emb = GoogleGenerativeAIEmbeddings(
                model=settings.gemini_embedding_model,
                google_api_key=settings.google_api_key,
            )
            return emb, f"gemini ({settings.gemini_embedding_model})"
        except Exception:
            pass
    return DeterministicHashEmbeddings(dim=384), "local_deterministic_384d"


def _sanitize_chroma_metadata(raw_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens list/dict metadata values into Chroma-compatible scalar primitives."""
    clean: Dict[str, Any] = {}
    for k, v in raw_meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            if k == "images" and v:
                clean["primary_image"] = str(v[0])
                clean["images_joined"] = "|".join(str(x) for x in v[:5])
            else:
                clean[k] = ", ".join(str(x) for x in v)
        else:
            clean[k] = str(v)
    return clean


class PackItUpHybridRAG:
    """Domain-partitioned Local ChromaDB + Lexical RRF Retriever."""

    def __init__(self, catalog: Optional[PackItUpCatalog] = None) -> None:
        self.settings = get_settings()
        self.catalog = catalog or get_catalog()
        self.embeddings, self.embedding_backend = get_embeddings_model()
        self.vectorstore = None
        self._init_chroma()

    def _init_chroma(self) -> None:
        """Initializes or loads local persistent ChromaDB collection from `chunks.jsonl`."""
        try:
            # Handle Linux SQLite3 version requirement for ChromaDB on Streamlit Cloud
            try:
                __import__("pysqlite3")
                import sys
                sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
            except ImportError:
                pass

            from langchain_chroma import Chroma

            persist_dir = str(self.settings.chroma_persist_dir)
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            collection_name = (
                f"{self.settings.chroma_collection_name}_gemini"
                if "gemini" in self.embedding_backend
                else f"{self.settings.chroma_collection_name}_local"
            )

            vs = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_dir,
            )

            existing_count = vs._collection.count()
            if existing_count == 0 and self.catalog.chunks:
                docs: List[Document] = []
                ids: List[str] = []
                for chunk in self.catalog.chunks:
                    cid = str(chunk["id"])
                    meta = _sanitize_chroma_metadata(chunk.get("metadata", {}))
                    meta["chunk_id"] = cid
                    docs.append(Document(page_content=chunk["text"], metadata=meta))
                    ids.append(cid)

                batch_size = 100
                for i in range(0, len(docs), batch_size):
                    vs.add_documents(documents=docs[i : i + batch_size], ids=ids[i : i + batch_size])

            self.vectorstore = vs
        except Exception:
            # Graceful fallback to in-memory lexical + cosine search if chromadb isn't installed in test env
            self.vectorstore = None

    def _lexical_search(
        self,
        query: str,
        doc_type: Optional[str] = None,
        category_filter: Optional[str] = None,
        max_price_pkr: Optional[float] = None,
        in_stock_only: bool = False,
        k: int = 10,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """BM25-inspired lexical & dimension scorer across enriched chunks."""
        q_low = (query or "").lower()
        q_tokens = [t for t in re.findall(r"[a-z0-9]+", q_low) if len(t) > 1]
        scored: List[Tuple[Dict[str, Any], float]] = []

        for chunk in self.catalog.chunks:
            meta = chunk.get("metadata", {})
            if doc_type and meta.get("type") != doc_type:
                continue
            if in_stock_only and meta.get("type") == "product" and not meta.get("in_stock", False):
                continue
            if (
                max_price_pkr is not None
                and max_price_pkr > 0
                and meta.get("type") == "product"
                and float(meta.get("price_min_pkr") or 0.0) > max_price_pkr
            ):
                continue
            if category_filter and meta.get("type") == "product":
                cat_hay = f"{meta.get('category', '')} {meta.get('collections', '')}".lower()
                if category_filter.lower() not in cat_hay:
                    continue

            text_low = chunk["text"].lower().replace('"', '').replace("'", "")
            title_low = f"{meta.get('title', '')} {meta.get('handle', '')}".lower().replace('"', '').replace("'", "")
            score = 0.0
            for tok in q_tokens:
                if tok in title_low:
                    score += 4.0 + (2.0 if any(c.isdigit() for c in tok) else 0.0)
                if tok in text_low:
                    score += 1.2
            # Exact phrase / dimension boost (e.g. "10x10x4", "12oz", "cupcake", "boba")
            if q_low and len(q_low) > 3 and q_low in title_low:
                score += 6.0
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def retrieve(
        self,
        query: str,
        doc_type: Optional[str] = None,
        category_filter: Optional[str] = None,
        max_price_pkr: Optional[float] = None,
        in_stock_only: bool = False,
        k: int = 6,
    ) -> List[Dict[str, Any]]:
        """Executes Hybrid (Chroma Dense + Lexical RRF) retrieval with metadata filtering."""
        rrf_scores: Dict[str, float] = Counter()
        chunk_by_id: Dict[str, Dict[str, Any]] = {c["id"]: c for c in self.catalog.chunks}

        # 1. Chroma Vector Search
        if self.vectorstore is not None:
            try:
                where_clauses: List[Dict[str, Any]] = []
                if doc_type:
                    where_clauses.append({"type": doc_type})
                if in_stock_only and doc_type == "product":
                    where_clauses.append({"in_stock": True})
                if max_price_pkr is not None and max_price_pkr > 0 and doc_type == "product":
                    where_clauses.append({"price_min_pkr": {"$lte": float(max_price_pkr)}})

                chroma_filter: Optional[Dict[str, Any]] = None
                if len(where_clauses) == 1:
                    chroma_filter = where_clauses[0]
                elif len(where_clauses) > 1:
                    chroma_filter = {"$and": where_clauses}

                dense_docs = self.vectorstore.similarity_search(
                    query=query,
                    k=max(k * 2, 10),
                    filter=chroma_filter,
                )
                for rank, d in enumerate(dense_docs):
                    cid = d.metadata.get("chunk_id")
                    if cid and cid in chunk_by_id:
                        if category_filter and doc_type == "product":
                            cat_hay = f"{d.metadata.get('category', '')} {d.metadata.get('collections', '')}".lower()
                            if category_filter.lower() not in cat_hay:
                                continue
                        rrf_scores[cid] += 1.0 / (60.0 + rank + 1.0)
            except Exception:
                pass

        # 2. Lexical BM25-style Search
        lex_hits = self._lexical_search(
            query=query,
            doc_type=doc_type,
            category_filter=category_filter,
            max_price_pkr=max_price_pkr,
            in_stock_only=in_stock_only,
            k=max(k * 2, 10),
        )
        for rank, (chunk, _lscore) in enumerate(lex_hits):
            cid = chunk["id"]
            rrf_scores[cid] += 1.15 / (60.0 + rank + 1.0)

        ranked_ids = [cid for cid, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)]
        results: List[Dict[str, Any]] = []
        for cid in ranked_ids[:k]:
            c = chunk_by_id[cid]
            results.append({
                "id": cid,
                "text": c["text"],
                "metadata": c["metadata"],
                "rrf_score": round(rrf_scores[cid], 5),
            })
        return results

    def search_products(
        self,
        query: str,
        category: Optional[str] = None,
        max_price_pkr: Optional[float] = None,
        in_stock_only: bool = False,
        top_k: int = 6,
    ) -> List[Dict[str, Any]]:
        """Retrieves deduplicated ProductCard dictionaries ranked by hybrid RAG relevance."""
        chunks = self.retrieve(
            query=query,
            doc_type="product",
            category_filter=category,
            max_price_pkr=max_price_pkr,
            in_stock_only=in_stock_only,
            k=top_k * 3,
        )
        seen_handles = set()
        cards: List[Dict[str, Any]] = []
        for ch in chunks:
            meta = ch["metadata"]
            handle = meta.get("handle")
            if not handle and meta.get("source_url"):
                handle = str(meta["source_url"]).split("/products/")[-1].strip("/")
            if not handle or handle in seen_handles:
                continue
            prod = self.catalog.get_product(handle)
            if not prod:
                continue
            seen_handles.add(handle)
            best_var = self.catalog.resolve_variant(prod, natural_query=query)
            cards.append(prod.to_card_dict(selected_variant=best_var))
            if len(cards) >= top_k:
                break
        return cards


_RAG_INSTANCE: Optional[PackItUpHybridRAG] = None


def get_rag_engine() -> PackItUpHybridRAG:
    """Singleton accessor for the Hybrid ChromaDB RAG engine."""
    global _RAG_INSTANCE
    if _RAG_INSTANCE is None:
        _RAG_INSTANCE = PackItUpHybridRAG()
    return _RAG_INSTANCE
