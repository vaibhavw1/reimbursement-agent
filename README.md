# Travel Reimbursement Approval Agent

An AI agent that evaluates employee travel reimbursement claims against company policy and returns a structured decision — built as part of a GenAI developer assignment.

## What it does

The agent takes a travel claim as input, runs it through 4 tools (policy lookup, limit checker, receipt validator, duplicate detector), and returns a decision — Approve, Partially Approve, Reject, or Manual Review — with a confidence score and full explanation.

## Tech Stack

- **LangGraph** — agentic workflow
- **Groq (Llama 3.3-70b)** — LLM for reasoning and tool calling
- **ChromaDB** — vector store for policy RAG
- **FastAPI** — REST API + serves the UI
- **Pydantic** — structured output validation

## Project Structure

```
├── agent/
│   ├── graph.py       # LangGraph agent loop
│   ├── tools.py       # 4 tools used by the agent
│   ├── models.py      # Pydantic schemas
│   └── prompts.py     # System prompt
├── api/
│   └── main.py        # FastAPI endpoints
├── data/
│   ├── policy.md      # Mock travel policy
│   ├── limits.json    # Per-category and city-tier limits
│   ├── claims.json    # 5 sample claims
│   └── receipts.json  # Mock receipt metadata
└── static/
    └── index.html     # UI
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add your Groq API key in .env
GROQ_API_KEY=your_key_here

# Run
uvicorn api.main:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser.

## Notes

- Free Groq API key works fine — get one at console.groq.com
- Evaluating all 5 claims back-to-back may hit Groq's free-tier rate limit; the agent retries automatically
- All data is mocked — no real employee or company information used
