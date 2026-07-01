import json
import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool
from typing import List

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Load data files
with open(os.path.join(DATA_DIR, "limits.json")) as f:
    LIMITS = json.load(f)

with open(os.path.join(DATA_DIR, "receipts.json")) as f:
    RECEIPTS = json.load(f)

# Setup ChromaDB with policy doc
_chroma_client = None
_collection = None

def get_policy_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    _chroma_client = chromadb.Client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    _collection = _chroma_client.get_or_create_collection(
        name="travel_policy",
        embedding_function=ef
    )

    # Load and chunk policy doc
    with open(os.path.join(DATA_DIR, "policy.md")) as f:
        policy_text = f.read()

    sections = [s.strip() for s in policy_text.split("##") if s.strip()]
    for i, section in enumerate(sections):
        _collection.upsert(
            documents=[section],
            ids=[f"section_{i}"]
        )

    return _collection


@tool
def policy_lookup(query: str) -> str:
    """Look up relevant travel policy sections for a given topic or expense category."""
    collection = get_policy_collection()
    results = collection.query(query_texts=[query], n_results=3)
    docs = results["documents"][0]
    return "\n\n---\n\n".join(docs)


@tool
def check_limits(category: str, amount: float, destination: str, days_or_nights: int = 1) -> str:
    """
    Check if an expense amount exceeds policy limits for a given category and destination.
    Returns whether it's within limits and the applicable limit.
    """
    cats = LIMITS["categories"]
    tiers = LIMITS["city_tiers"]

    # Find city tier
    city_tier = "tier3"
    for tier, info in tiers.items():
        if destination in info["cities"]:
            city_tier = tier
            break

    tier_info = tiers[city_tier]
    cat = category.lower()

    if cat not in cats:
        return f"Category '{category}' is not a recognized reimbursable expense category."

    cat_info = cats[cat]
    issues = []
    limit_used = None

    if cat == "hotel":
        max_night = tier_info.get("hotel_max", cat_info["max_per_night"])
        total_limit = max_night * days_or_nights
        limit_used = f"${max_night}/night x {days_or_nights} nights = ${total_limit}"
        if amount > total_limit:
            issues.append(f"Hotel exceeds limit: claimed ${amount}, limit is {limit_used}")

    elif cat == "meals":
        max_day = tier_info.get("meals_max", cat_info["max_per_day"])
        total_limit = max_day * days_or_nights
        limit_used = f"${max_day}/day x {days_or_nights} days = ${total_limit}"
        if amount > total_limit:
            issues.append(f"Meals exceed limit: claimed ${amount}, limit is {limit_used}")

    elif cat == "transport":
        max_day = cat_info["max_per_day"]
        total_limit = max_day * days_or_nights
        limit_used = f"${max_day}/day x {days_or_nights} days = ${total_limit}"
        if amount > total_limit:
            issues.append(f"Transport exceeds limit: claimed ${amount}, limit is {limit_used}")

    elif cat == "conference":
        max_amt = cat_info["max_per_claim"]
        limit_used = f"${max_amt}/claim"
        if amount > max_amt:
            issues.append(f"Conference fee exceeds limit: claimed ${amount}, limit is {limit_used}")
        if cat_info.get("requires_preapproval"):
            issues.append("Conference requires pre-approval documentation.")

    elif cat == "flight":
        max_amt = cat_info["max_per_claim"]
        limit_used = f"${max_amt}/claim"
        if amount > max_amt:
            issues.append(f"Flight exceeds limit: claimed ${amount}, limit is {limit_used}")

    if issues:
        return "LIMIT EXCEEDED:\n" + "\n".join(issues)
    return f"WITHIN LIMITS: ${amount} for {category} in {destination} ({limit_used})"


@tool
def check_receipts(item_ids: List[str]) -> str:
    """
    Check receipt status for a list of claim item IDs.
    Returns which items have missing receipts and their amounts.
    """
    missing = []
    present = []

    for item_id in item_ids:
        if item_id not in RECEIPTS:
            missing.append(f"{item_id}: receipt data not found")
        elif RECEIPTS[item_id]["status"] == "missing":
            r = RECEIPTS[item_id]
            missing.append(f"{item_id}: MISSING receipt — {r['vendor']} ${r['amount']} on {r['date']}")
        else:
            r = RECEIPTS[item_id]
            present.append(f"{item_id}: receipt OK — {r['vendor']} ${r['amount']}")

    result = []
    if present:
        result.append("RECEIPTS PRESENT:\n" + "\n".join(present))
    if missing:
        result.append("RECEIPTS MISSING:\n" + "\n".join(missing))

    return "\n\n".join(result) if result else "No receipt data found."


@tool
def detect_duplicates(items: List[dict]) -> str:
    """
    Detect duplicate expense items in a claim.
    A duplicate is same vendor + same date + same or very close amount.
    Input: list of dicts with keys: id, vendor, date, amount
    """
    seen = {}
    duplicates = []

    for item in items:
        key = f"{item['vendor'].lower().strip()}_{item['date']}"
        if key in seen:
            prev = seen[key]
            if abs(item['amount'] - prev['amount']) < 10:
                duplicates.append(
                    f"DUPLICATE: {item['id']} (${item['amount']}) matches {prev['id']} "
                    f"(${prev['amount']}) — same vendor '{item['vendor']}' on {item['date']}"
                )
            else:
                duplicates.append(
                    f"POSSIBLE DUPLICATE: {item['id']} and {prev['id']} — "
                    f"same vendor '{item['vendor']}' on {item['date']}, different amounts"
                )
        seen[key] = item

    if duplicates:
        return "DUPLICATES FOUND:\n" + "\n".join(duplicates)
    return "NO DUPLICATES DETECTED"