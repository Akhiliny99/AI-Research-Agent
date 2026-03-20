import re

# ── Input guardrails ──────────────────────────────────────────
BLOCKED_PATTERNS = [
    r"ignore (previous|all) instructions",
    r"you are now",
    r"forget your (rules|instructions|system prompt)",
    r"jailbreak",
    r"act as (a )?dan",
]

def check_input(query: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason).
    is_safe=False means block the query.
    """
    if not query or not query.strip():
        return False, "Query cannot be empty."

    if len(query.strip()) < 3:
        return False, "Query is too short. Please ask a proper question."

    if len(query) > 2000:
        return False, "Query is too long. Please keep it under 2000 characters."

    query_lower = query.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query_lower):
            return False, "This query contains disallowed content (prompt injection detected)."

    return True, "OK"


# ── Output guardrails ─────────────────────────────────────────
HALLUCINATION_PHRASES = [
    "as of my knowledge cutoff",
    "i don't have access to real-time",
    "i cannot browse the internet",
    "as an ai language model",
    "i'm not able to provide real-time",
]

def check_output(answer: str, tool_used: str) -> tuple[str, list[str]]:
    """
    Validates and cleans the final answer.
    Returns (cleaned_answer, warnings).
    """
    warnings = []

    if not answer or not answer.strip():
        return "I was unable to generate a response. Please try again.", ["Empty response"]

    answer_lower = answer.lower()
    for phrase in HALLUCINATION_PHRASES:
        if phrase in answer_lower:
            warnings.append(f"Possible hallucination signal detected: '{phrase}'")

    if tool_used == "web_search" and "http" not in answer.lower() and len(answer) > 200:
        warnings.append("Web search was used but no URLs appear in answer.")

    cleaned = answer.strip()
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned, warnings


# ── Format final response for UI ──────────────────────────────
def format_response(answer: str, sources: list[str], tool_used: str, warnings: list[str]) -> dict:
    """Package everything into a clean response object for the UI."""
    return {
        "answer": answer,
        "sources": sources,
        "tool_used": tool_used,
        "warnings": warnings,
        "has_warnings": len(warnings) > 0
    }