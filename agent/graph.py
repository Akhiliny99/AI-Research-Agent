from langgraph.graph import StateGraph, END
from agent.memory import AgentState
from agent.nodes import agent_node, tool_node, synthesizer_node, should_use_tool

def build_graph():
    """Build and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Entry point
    graph.set_entry_point("agent")

    # Conditional routing after agent decides
    graph.add_conditional_edges(
        "agent",
        should_use_tool,
        {
            "use_tool": "tools",
            "synthesize": "synthesizer"
        }
    )

    # After tools → back to agent for multi-step reasoning
    graph.add_edge("tools", "agent")

    # Synthesizer → end
    graph.add_edge("synthesizer", END)

    return graph.compile()


# Export compiled graph
research_agent = build_graph()