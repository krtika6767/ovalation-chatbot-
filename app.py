import pandas as pd
import os
from catalog_manager import search_catalog, load_catalogs, all_products, get_code, get_name, get_category, detect_intent

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()

app = FastAPI(title="Ovation AI Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =====================================
# DATA MODELS
# =====================================

class ChatMessage(BaseModel):
    role: str
    content: str

class OrderRequest(BaseModel):
    name: str
    email: str
    phone: str
    product: str
    quantity: int

# =====================================
# COMPANY INFO
# =====================================

COMPANY_INFO = """
Company: Ovation International
Phone: +91 141 4020600 | WhatsApp: +91 8949765166
Email: info@ovationint.com
Address: B-24 (B-1), Prabhu Marg, Tilak Nagar, Jaipur, Rajasthan, India
"""

# =====================================
# SYSTEM PROMPT
# =====================================

SYSTEM_PROMPT = """
You are Ovation International's AI Sales Assistant for ophthalmic microsurgery instruments.

CRITICAL RULES:
1. Use ONLY catalog information provided. Never invent products, prices, or stock.
2. Use the full conversation history to understand context and references.

3. NUMBER REFERENCES — VERY IMPORTANT:
   - When a user sends just a number like "4" or "18", it means they want item #N
     from the NUMBERED LIST you showed in your PREVIOUS message.
   - You must look at your previous response, count the items in order, and identify
     which product is at that position.
   - Do NOT search the catalog for that number. Do NOT treat it as a product code.
   - Example: if your last list had 20 products and user says "3", find the 3rd product
     in that list and give its full details.

4. When a user asks for description of a product just listed, give the full description
   from catalog context — do NOT say "not found".
5. When listing products, list ALL products shown in catalog context.
6. Match products by BOTH product code AND product name.
7. For category queries, list every product in that category.
8. Be concise. Do not repeat product details unnecessarily.
9. When listing products, number each item (1. 2. 3. ...) so users can reference by number.

AVAILABLE ACTIONS (append when relevant):
[ACTION:REQUEST_QUOTE] — when user wants a quotation
[ACTION:CREATE_ORDER]  — when user wants to place an order
[ACTION:CONTACT_SALES] — when user wants to speak to sales team
[ACTION:ADD_TO_CART]   — when user wants to add a product to cart
"""

# =====================================
# LAST LIST TRACKER
# =====================================
# Tracks the last numbered product list the bot showed, per session.
# Key: a simple in-memory store (works for single-user; use Redis for multi-user prod)
last_shown_list: List[dict] = []  # list of {"index": N, "code": "...", "name": "..."}


def extract_numbered_list_from_response(response_text: str) -> List[dict]:
    """
    Parse a bot response and extract numbered product entries like:
    '1. **OI-3301** — TOOKE CORNEAL KNIFE'
    '1. OI-3301 — TOOKE CORNEAL KNIFE'
    Returns a list of dicts: [{"index": 1, "code": "OI-3301", "name": "TOOKE CORNEAL KNIFE"}, ...]
    """
    items = []
    # Match lines like: "1. **CODE** — NAME" or "1. CODE — NAME" or "- **CODE**: NAME"
    patterns = [
        r'(\d+)\.\s+\*?\*?([A-Z0-9\s\-]+?)\*?\*?\s+[—\-–]\s+(.+)',
        r'(\d+)\.\s+([A-Z0-9\s\-]+?)\s+[—\-–]\s+(.+)',
        r'-\s+\*?\*?([A-Z0-9\s\-]+?)\*?\*?:\s+(.+)',  # bullet style: "- **CODE**: NAME"
    ]
    for line in response_text.split('\n'):
        line = line.strip()
        for i, pattern in enumerate(patterns):
            m = re.match(pattern, line)
            if m:
                if i < 2:
                    idx  = int(m.group(1))
                    code = m.group(2).strip().strip('*').strip()
                    name = m.group(3).strip().strip('*').strip()
                else:
                    idx  = len(items) + 1  # bullet style — infer index
                    code = m.group(1).strip().strip('*').strip()
                    name = m.group(2).strip().strip('*').strip()
                items.append({"index": idx, "code": code, "name": name})
                break
    return items


def get_item_at_position(position: int) -> Optional[dict]:
    """Return the product at position N from the last shown list."""
    for item in last_shown_list:
        if item["index"] == position:
            return item
    # Fallback: by list order
    if 1 <= position <= len(last_shown_list):
        return last_shown_list[position - 1]
    return None


def is_number_reference(text: str) -> Optional[int]:
    """
    Returns the integer if the user message is just a number or 'item N' / '#N'.
    Returns None otherwise.
    """
    text = text.strip()
    # Pure number
    if re.fullmatch(r'\d+', text):
        return int(text)
    # "item 5", "#5", "no. 5", "number 5"
    m = re.match(r'(?:item|#|no\.?|number)\s*(\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


# =====================================
# TOKEN-SAFE CONVERSATION TRIMMER
# =====================================

def trim_conversation(messages: List[ChatMessage], max_chars: int = 8000) -> List[ChatMessage]:
    if not messages:
        return messages

    must_include = [messages[-1]]
    candidates   = list(messages[:-1])
    used_chars   = sum(len(m.content) for m in must_include)
    kept = []

    for msg in reversed(candidates):
        if used_chars + len(msg.content) < max_chars:
            kept.insert(0, msg)
            used_chars += len(msg.content)
        else:
            break

    if not kept and candidates:
        kept = candidates[-4:]

    return kept + must_include


# =====================================
# CATALOG CONTEXT BUILDER
# =====================================

def build_catalog_context(messages: List[ChatMessage], number_ref: Optional[int] = None) -> str:
    if not messages:
        return ""

    # If it's a number reference, look up from last shown list instead of searching catalog
    if number_ref is not None:
        item = get_item_at_position(number_ref)
        if item:
            # Search the catalog for this specific product
            context = search_catalog(item["code"])
            if "No products found" in context:
                context = search_catalog(item["name"])
            return (
                f"The user is asking about item #{number_ref} from the previous list.\n"
                f"That item is: {item['code']} — {item['name']}\n\n"
                f"Full catalog details:\n{context}"
            )
        else:
            return (
                f"The user typed '{number_ref}' but there is no item #{number_ref} "
                f"in the last shown list (list had {len(last_shown_list)} items). "
                f"Politely tell the user which numbers are valid (1 to {len(last_shown_list)})."
            )

    latest_user_msg = messages[-1].content
    primary_context = search_catalog(latest_user_msg)

    is_ambiguous = (
        len(latest_user_msg.strip()) <= 3 or
        any(w in latest_user_msg.lower() for w in [
            "that", "this", "it", "first", "second", "third", "fourth", "fifth",
            "last", "previous", "above", "description", "desc", "details", "tell me more"
        ])
    )

    secondary_context = ""
    if is_ambiguous:
        recent_window = messages[-5:-1] if len(messages) > 1 else []
        combined_recent = " ".join(m.content for m in recent_window)
        if combined_recent.strip():
            secondary_context = search_catalog(combined_recent)

    if secondary_context and secondary_context != primary_context:
        return (
            f"=== Current message search results ===\n{primary_context}\n\n"
            f"=== Context from recent conversation ===\n{secondary_context}"
        )

    return primary_context


# =====================================
# ROUTES
# =====================================

@app.get("/")
def home():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    return FileResponse(template_path)

@app.get("/api/health")
def health():
    return {"status": "running", "message": "Ovation AI Assistant Running"}

@app.post("/api/create-order")
def create_order(order: OrderRequest):
    import uuid
    order_id = str(uuid.uuid4())[:8]
    return {
        "success": True,
        "order_id": order_id,
        "customer": order.name,
        "product": order.product,
        "quantity": order.quantity,
        "message": "Order request submitted successfully. Sales team will contact you shortly."
    }

@app.get("/api/info")
def info():
    return {"company": "Ovation International", "assistant": "AI Sales Assistant", "status": "ready"}

@app.get("/api/test/{query}")
def test(query: str):
    return search_catalog(query)


# =====================================
# CHAT ENDPOINT
# =====================================

@app.post("/api/chat")
def chat(messages: List[ChatMessage]):
    global last_shown_list

    try:
        user_message = messages[-1].content.strip()

        # ── Detect if user is referencing a numbered item ──────────────────
        number_ref = is_number_reference(user_message)

        # ── Build catalog context ──────────────────────────────────────────
        catalog_context = build_catalog_context(messages, number_ref=number_ref)

        # ── Trim conversation history ──────────────────────────────────────
        trimmed_messages = trim_conversation(messages, max_chars=8000)

        conversation_history = []
        for msg in trimmed_messages:
            role = msg.role if msg.role in ("user", "assistant") else "user"
            conversation_history.append({"role": role, "content": msg.content})

        # ── Build Groq prompt ──────────────────────────────────────────────
        groq_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    f"Company Information:\n{COMPANY_INFO}\n\n"
                    f"Catalog Search Results:\n{catalog_context}"
                )
            },
            *conversation_history
        ]

        # ── Call Groq ──────────────────────────────────────────────────────
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1500,
            messages=groq_messages
        )

        ai_response = response.choices[0].message.content

        # ── Update last shown list from this response ──────────────────────
        parsed = extract_numbered_list_from_response(ai_response)
        if parsed:
            last_shown_list = parsed  # replace with new list

        # ── Extract action tags ────────────────────────────────────────────
        action_pattern = r'\[ACTION:([A-Z_]+)\]'
        actions = re.findall(action_pattern, ai_response)
        response_text = re.sub(action_pattern, '', ai_response).strip()

        return {
            "response": response_text,
            "actions": list(set(actions))
        }

    except Exception as e:
        error_str = str(e)
        if "413" in error_str or "rate_limit_exceeded" in error_str or "tokens" in error_str.lower():
            return {
                "response": (
                    "This conversation has become too long to process. "
                    "Please start a new chat session — it will work perfectly. "
                    "Sorry for the inconvenience!"
                ),
                "actions": ["CONTACT_SALES"]
            }
        return {"response": f"Error: {error_str}", "actions": []}