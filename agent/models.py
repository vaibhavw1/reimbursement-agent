from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class Decision(str, Enum):
    APPROVE = "Approve"
    PARTIALLY_APPROVE = "Partially Approve"
    REJECT = "Reject"
    MANUAL_REVIEW = "Manual Review"

class ClaimItem(BaseModel):
    id: str
    category: str
    amount: float
    vendor: str
    date: str
    receipt: bool
    nights: Optional[int] = None
    days: Optional[int] = None
    notes: Optional[str] = None

class ClaimInput(BaseModel):
    claim_id: str
    employee: str
    travel_dates: str
    destination: str
    purpose: str
    total_claimed: float
    items: List[ClaimItem]

class EvaluationResult(BaseModel):
    claim_id: str
    employee: str
    decision: Decision
    approved_amount: float
    deducted_amount: float
    rejected_items: List[str]
    missing_documents: List[str]
    policy_references: List[str]
    confidence: float  # 0.0 to 1.0
    explanation: str
    approval_required_from: Optional[str] = None
    audit_trail: List[str] = []