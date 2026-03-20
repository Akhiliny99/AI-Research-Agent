from langgraph.graph import StateGraph, END
from agent.memory import AgentState
from agent.nodes import agent_node, tool_node, synthesizer_node, should_use_tool

def build_graph():
    """Build and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

   
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("synthesizer", synthesizer_node)

    
    graph.set_entry_point("agent")

   
    graph.add_conditional_edges(
        "agent",
        should_use_tool,
        {
            "use_tool": "tools",
            "synthesize": "synthesizer"
        }
    )

   
    graph.add_edge("tools", "agent")

   
    graph.add_edge("synthesizer", END)

    return graph.compile()




research_agent = build_graph()

