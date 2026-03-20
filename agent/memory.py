from langgraph.graph import MessagesState
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(MessagesState):
    """State that flows through the graph."""
    query: str
    sources: list[str]
    final_answer: str
    tool_used: str