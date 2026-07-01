import json
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent.tools import policy_lookup, check_limits, check_receipts, detect_duplicates
from agent.models import ClaimInput, EvaluationResult, Decision
from agent.prompts import SYSTEM_PROMPT

load_dotenv()

TOOLS = [policy_lookup, check_limits, check_receipts, detect_duplicates]

def build_agent():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    ).bind_tools(TOOLS)

    class AgentState(TypedDict):
        messages: Annotated[list, lambda x, y: x + y]
        claim: dict
        audit_trail: list

    def agent_node(state: AgentState):
        messages = state["messages"]
        response = llm.invoke(messages)
        audit = state.get("audit_trail", [])
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                audit.append(f"Tool called: {tc['name']} with args: {json.dumps(tc['args'])}")
        return {"messages": [response], "audit_trail": audit}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            called_tools = set()
            for m in state["messages"]:
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        called_tools.add(tc["name"])
            if len(called_tools) >= 4:  # all 4 tools already used at least once
                return "parse"
            return "tools"
        return "parse"

    def parse_output(state: AgentState):
        claim = state["claim"]
        messages = state["messages"]
        audit = state.get("audit_trail", [])

        # Ask LLM to produce structured JSON output
        parse_prompt = f"""
Based on your analysis of claim {claim['claim_id']}, produce a JSON object with exactly these fields:

{{
  "decision": "Approve" | "Partially Approve" | "Reject" | "Manual Review",
  "approved_amount": <number>,
  "deducted_amount": <number>,
  "rejected_items": ["item_id: reason", ...],
  "missing_documents": ["description", ...],
  "policy_references": ["policy section or rule", ...],
  "confidence": <0.0 to 1.0>,
  "explanation": "clear 2-3 sentence explanation",
  "approval_required_from": null | "Manager" | "Director" | "VP"
}}

Total claimed: ${claim['total_claimed']}

IMPORTANT:
- policy_references must NOT be empty.
- Include every policy section or rule that you relied on to reach the decision.
- Even if the claim is fully approved, list the policy sections you verified.
- Cite the actual policy headings or section names returned by the policy_lookup tool whenever possible.
- Every rejected item should have at least one corresponding policy reference.
- Return ONLY the JSON object. Do not include markdown, code fences, or any additional text.
"""
        parse_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        parse_response = parse_llm.invoke(messages + [HumanMessage(content=parse_prompt)])
        raw = parse_response.content.strip()

        # Clean up markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw.strip())
        except Exception:
            data = {
                "decision": "Manual Review",
                "approved_amount": 0,
                "deducted_amount": claim["total_claimed"],
                "rejected_items": [],
                "missing_documents": ["Could not parse agent output — manual review required"],
                "policy_references": [],
                "confidence": 0.3,
                "explanation": "Agent encountered an error during evaluation. Manual review required.",
                "approval_required_from": "Manager"
            }

        result = EvaluationResult(
            claim_id=claim["claim_id"],
            employee=claim["employee"],
            audit_trail=audit,
            **data
        )
        return {"messages": [AIMessage(content=result.model_dump_json())]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("parse", parse_output)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "parse": "parse"})
    graph.add_edge("tools", "agent")
    graph.add_edge("parse", END)

    return graph.compile()


def evaluate_claim(claim_input: ClaimInput) -> EvaluationResult:
    agent = build_agent()
    claim_dict = claim_input.model_dump()

    user_message = f"""
Please evaluate this travel reimbursement claim:

{json.dumps(claim_dict, indent=2)}

Steps to follow:
1. Call policy_lookup for each expense category to check eligibility
2. Call check_limits for each line item with its category, amount, destination, and number of days/nights
3. Call check_receipts with all item IDs
4. Call detect_duplicates with all items (vendor, date, amount, id)
5. Based on all tool results, produce your final structured decision
"""

    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ],
        "claim": claim_dict,
        "audit_trail": []
    }

    final_state = agent.invoke(initial_state)
    last_message = final_state["messages"][-1]
    result = EvaluationResult.model_validate_json(last_message.content)
    result.audit_trail = final_state.get("audit_trail", [])
    return result