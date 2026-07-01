import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from agent.models import ClaimInput, EvaluationResult
from agent.graph import evaluate_claim
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(
    title="Travel Reimbursement Approval Agent",
    description="AI agent that evaluates employee travel claims against company policy",
    version="1.0.0"
)

@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "static", "index.html"))

@app.get("/claims")
def get_claims():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "claims.json")
    with open(data_path) as f:
        return json.load(f)

@app.post("/evaluate-claim", response_model=EvaluationResult)
def evaluate(claim: ClaimInput):
    try:
        result = evaluate_claim(claim)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate-all-samples")
def evaluate_all():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "claims.json")
    with open(data_path) as f:
        claims_data = json.load(f)

    results = []
    for claim_dict in claims_data:
        try:
            claim = ClaimInput(**claim_dict)
            result = evaluate_claim(claim)
            results.append(result.model_dump())
        except Exception as e:
            results.append({"claim_id": claim_dict.get("claim_id"), "error": str(e)})

    return {"total": len(results), "results": results}

@app.get("/health")
def health():
    return {"status": "healthy"}