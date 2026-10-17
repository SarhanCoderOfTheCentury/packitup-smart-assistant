"""Meta WhatsApp Business Cloud API (Graph API v21.0) Webhook Router for FastAPI.

Implements end-to-end WhatsApp Business automation:
1. `GET /api/v1/whatsapp/webhook` — Meta App Dashboard webhook verification (`hub.verify_token` & `hub.challenge`).
2. `POST /api/v1/whatsapp/webhook` — Receives inbound customer messages (text, interactive button replies,
   list replies), verifies `X-Hub-Signature-256` HMAC-SHA256 signature, and dispatches an async background
   task to the Multi-Agent LangGraph (`run_shopping_assistant`) mapped by customer phone (`wa_id`).
3. Formats LangGraph product cards, cart updates, and Shopify checkout links into native WhatsApp Cloud API
   Interactive Messages (Reply Buttons / List Messages) and pushes them to `https://graph.facebook.com/v21.0/{PHONE_ID}/messages`.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse

from src.agents.graph import run_shopping_assistant
from src.config import get_settings

logger = logging.getLogger("packitup.whatsapp")
router = APIRouter(prefix="/api/v1/whatsapp", tags=["WhatsApp Business Automation"])

_PROCESSED_MESSAGE_IDS = set()


def verify_meta_signature(raw_body: bytes, signature_header: Optional[str], app_secret: str) -> bool:
    """Verifies Meta's `X-Hub-Signature-256: sha256=<hex>` header using HMAC-SHA256."""
    if not app_secret:
        # In local development without app secret configured, allow testing
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    provided = signature_header.split("sha256=", 1)[-1].strip()
    return hmac.compare_digest(expected, provided)


def build_whatsapp_outbound_payloads(
    recipient_phone: str,
    agent_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Transforms a LangGraph turn result into WhatsApp Cloud API v21.0 message payloads."""
    payloads: List[Dict[str, Any]] = []
    reply_text = agent_result.get("response", "").strip()
    # Strip markdown headers for cleaner WhatsApp rendering
    clean_wa_text = (
        reply_text.replace("### ", "*")
        .replace("**", "*")
    )
    if len(clean_wa_text) > 3800:
        clean_wa_text = clean_wa_text[:3790] + "..."

    # 1. Primary Text + Interactive Action Buttons
    cards = agent_result.get("product_cards", [])
    cart = agent_result.get("cart", {})

    buttons = []
    if cards:
        top_card = cards[0]
        buttons.append({
            "type": "reply",
            "reply": {
                "id": f"ADD_CART::{top_card['handle']}",
                "title": "Add #1 to Cart"[:20],
            },
        })
    if cart.get("item_count", 0) > 0:
        buttons.append({
            "type": "reply",
            "reply": {
                "id": "ACTION::CHECKOUT",
                "title": "Checkout Link"[:20],
            },
        })
    buttons.append({
        "type": "reply",
        "reply": {
            "id": "ACTION::CUSTOM_QUOTE",
            "title": "Custom Logo Quote"[:20],
        },
    })

    payloads.append({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_phone,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "header": {"type": "text", "text": "PackItUp Smart Assistant"},
            "body": {"text": clean_wa_text[:1020]},
            "footer": {"text": f"Cart: {cart.get('item_count', 0)} items ({cart.get('subtotal_display', 'PKR 0')})"},
            "action": {"buttons": buttons[:3]},
        },
    })

    # 2. Optional top product image card if discovery returned products
    if cards and cards[0].get("image_url"):
        top = cards[0]
        payloads.append({
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient_phone,
            "type": "image",
            "image": {
                "link": top["image_url"],
                "caption": (
                    f"*{top['title']}*\n"
                    f"Price: {top['price_display']}\n"
                    f"View: {top['url']}\n"
                    f"Reply *\"Add 1 pack of {top['title']} to cart\"* to order!"
                )[:1000],
            },
        })

    return payloads


async def _send_whatsapp_messages(payloads: List[Dict[str, Any]]) -> None:
    """Sends formatted message payloads to Meta Graph API v21.0."""
    settings = get_settings()
    if not settings.whatsapp_access_token or not settings.whatsapp_phone_number_id:
        logger.info("WhatsApp credentials not set; dry-run logged %d outbound payloads.", len(payloads))
        return

    url = f"https://graph.facebook.com/v21.0/{settings.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        for payload in payloads:
            try:
                await client.post(url, json=payload, headers=headers)
            except Exception as exc:
                logger.error("Failed sending WhatsApp message: %s", exc)


async def process_whatsapp_turn(sender_phone: str, user_text: str) -> Dict[str, Any]:
    """Runs LangGraph turn for `wa_<sender_phone>` session and dispatches WhatsApp replies."""
    session_id = f"wa_{sender_phone}"
    result = run_shopping_assistant(
        user_query=user_text,
        session_id=session_id,
        channel="whatsapp",
    )
    outbound = build_whatsapp_outbound_payloads(sender_phone, result)
    await _send_whatsapp_messages(outbound)
    return {"session_id": session_id, "agent_result": result, "whatsapp_payloads": outbound}


@router.get("/webhook")
async def whatsapp_webhook_verify(
    hub_mode: Optional[str] = Query(default=None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(default=None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(default=None, alias="hub.challenge"),
) -> Response:
    """Meta WhatsApp Cloud API GET verification endpoint."""
    settings = get_settings()
    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        return PlainTextResponse(content=str(hub_challenge or ""), status_code=200)
    raise HTTPException(status_code=403, detail="Invalid WhatsApp webhook verification token.")


@router.post("/webhook")
async def whatsapp_webhook_receive(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(default=None, alias="X-Hub-Signature-256"),
) -> Dict[str, Any]:
    """Receives inbound Meta WhatsApp webhook events and processes them asynchronously."""
    settings = get_settings()
    raw_body = await request.body()
    if not verify_meta_signature(raw_body, x_hub_signature_256, settings.whatsapp_app_secret):
        raise HTTPException(status_code=401, detail="Invalid X-Hub-Signature-256 header.")

    body = await request.json()
    entries = body.get("entry", [])
    dispatched = 0

    for entry in entries:
        for change in entry.get("changes", []):
            val = change.get("value", {})
            for msg in val.get("messages", []):
                msg_id = msg.get("id")
                if msg_id and msg_id in _PROCESSED_MESSAGE_IDS:
                    continue
                if msg_id:
                    _PROCESSED_MESSAGE_IDS.add(msg_id)

                sender = msg.get("from", "unknown")
                msg_type = msg.get("type")
                user_text = ""

                if msg_type == "text":
                    user_text = msg.get("text", {}).get("body", "")
                elif msg_type == "interactive":
                    inter = msg.get("interactive", {})
                    btn_reply = inter.get("button_reply") or inter.get("list_reply") or {}
                    btn_id = btn_reply.get("id", "")
                    if btn_id.startswith("ADD_CART::"):
                        handle = btn_id.split("ADD_CART::", 1)[-1]
                        user_text = f"Add 1 pack of {handle} to my cart"
                    elif btn_id == "ACTION::CHECKOUT":
                        user_text = "Generate my Shopify checkout link"
                    elif btn_id == "ACTION::CUSTOM_QUOTE":
                        user_text = "I want a custom logo packaging quote"
                    else:
                        user_text = btn_reply.get("title", "")

                if user_text:
                    background_tasks.add_task(process_whatsapp_turn, sender, user_text)
                    dispatched += 1

    return {"status": "accepted", "dispatched_turns": dispatched}


@router.post("/simulate")
async def simulate_whatsapp_message(
    sender_phone: str = "+923001234567",
    message_text: str = "Show me 10x10x4 cake boxes and add 2 packs of white 50 pack to my cart",
) -> Dict[str, Any]:
    """Developer testing endpoint to preview exact WhatsApp Cloud API payloads generated by LangGraph."""
    return await process_whatsapp_turn(sender_phone=sender_phone, user_text=message_text)
