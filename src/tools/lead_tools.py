"""B2B Custom Packaging & Lead Capture Tools storing records in `data/leads.csv` (Agent A5)."""
from __future__ import annotations

from typing import Any, Dict
from src import tool
from pydantic import BaseModel, Field

from src.rag_chroma import get_rag_engine
from src.state_store import LeadStore


class LeadCaptureInput(BaseModel):
    customer_name: str = Field(..., description="Customer's full name or contact person.")
    phone: str = Field(..., description="Customer's phone or WhatsApp number (e.g., '+92 300 1234567').")
    email: str = Field(default="", description="Customer's email address if provided.")
    company_name: str = Field(default="", description="Brand, bakery, café, restaurant, or company name.")
    city: str = Field(default="Karachi", description="Delivery city in Pakistan (e.g. Karachi, Lahore, Islamabad).")
    inquiry_type: str = Field(
        default="custom_printing",
        description="One of: 'custom_printing', 'wholesale_bulk', 'custom_size_quote', 'callback_request'.",
    )
    product_interest: str = Field(
        ...,
        description="Products or packaging categories of interest (e.g. 'Custom Logo Burger Boxes & Paper Bags').",
    )
    estimated_quantity: str = Field(
        default="500-1000 pcs",
        description="Estimated order volume / MOQ requested.",
    )
    customization_details: str = Field(
        default="",
        description="Printing colors (1-color, 2-color, full CMYK), logo finish, dimensions, or material specs.",
    )
    budget_pkr: str = Field(default="", description="Optional target budget in PKR.")
    source_channel: str = Field(default="chat_assistant", description="Channel where lead originated.")
    notes: str = Field(default="", description="Any additional context from the conversation.")


@tool("capture_customer_lead", args_schema=LeadCaptureInput)
def capture_customer_lead(
    customer_name: str,
    phone: str,
    product_interest: str,
    email: str = "",
    company_name: str = "",
    city: str = "Karachi",
    inquiry_type: str = "custom_printing",
    estimated_quantity: str = "500-1000 pcs",
    customization_details: str = "",
    budget_pkr: str = "",
    source_channel: str = "chat_assistant",
    notes: str = "",
) -> Dict[str, Any]:
    """Saves a qualified customer lead or B2B custom packaging request to `data/leads.csv`.

    Use this tool whenever a user:
    - Requests custom logo printing, branded packaging, wholesale pricing, or bulk restaurant supply
    - Enquires about a PKR 0.00 custom-quote product (e.g. Custom Serving Tray, Hang Tags, Visiting Cards)
    - Provides their name and phone/email to request a quote or callback from PackItUp sales.
    """
    record = LeadStore.add_lead(
        customer_name=customer_name,
        phone=phone,
        email=email,
        company_name=company_name,
        city=city,
        inquiry_type=inquiry_type,
        product_interest=product_interest,
        estimated_quantity=estimated_quantity,
        customization_details=customization_details,
        budget_pkr=budget_pkr,
        source_channel=source_channel,
        notes=notes,
    )
    return {
        "success": True,
        "lead": record,
        "message": (
            f"Lead {record['lead_id']} saved to leads.csv for {record['customer_name']} ({record['phone']}). "
            "Our Karachi B2B Packaging Specialist (+92 322 2620169) will follow up promptly."
        ),
    }


class B2BPageSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="Query about custom orders, restaurant supply, eco packaging, or B2B capabilities.",
    )


@tool("search_b2b_custom_packaging", args_schema=B2BPageSearchInput)
def search_b2b_custom_packaging(query: str) -> Dict[str, Any]:
    """Searches PackItUp's B2B landing pages (`custom-orders`, `food-restaurant`, `eco-packaging`) and custom collections."""
    rag = get_rag_engine()
    page_chunks = rag.retrieve(query=query, doc_type="page", k=4)
    custom_prods = rag.search_products(query=f"custom {query}", top_k=4)
    return {
        "b2b_pages": [
            {
                "title": c["metadata"].get("title"),
                "url": c["metadata"].get("source_url"),
                "excerpt": c["text"],
            }
            for c in page_chunks
        ],
        "custom_products": custom_prods,
    }
