import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agent.tools import tools, web_search_tool, rag_search
from agent.memory import AgentState


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a helpful research assistant with access to two tools:

1. web_search — use this for current events, recent information, or anything that needs up-to-date data
2. rag_search — use this to search uploaded documents the user has provided

Decision rules:
- If the question is about recent news or current facts → use web_search
- If the question seems to relate to an uploaded document → use rag_search  
- If you already know the answer confidently → answer directly without tools
- Never make up sources or citations
- Always be concise and factual
"""


def agent_node(state: AgentState) -> AgentState:
    """Main agent that decides whether to use a tool or answer directly."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}



def tool_node(state: AgentState) -> AgentState:
    """Executes whichever tool the agent selected."""
    from langchain_core.messages import ToolMessage

    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    sources = []
    results = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "web_search":
            raw = web_search_tool.invoke(tool_args)
            
            if isinstance(raw, dict) and "results" in raw:
                content = "\n\n".join([
                    f"[{r['title']}]\n{r['content']}\nURL: {r['url']}"
                    for r in raw["results"]
                ])
                sources = [r["url"] for r in raw["results"]]
            else:
                content = str(raw)
        elif tool_name == "rag_search":
            content = rag_search.invoke(tool_args)
            sources = ["uploaded documents"]
        else:
            content = "Tool not found."

        results.append(ToolMessage(
            content=content,
            tool_call_id=tool_call["id"]
        ))

    return {
        "messages": results,
        "sources": sources,
        "tool_used": tool_calls[0]["name"] if tool_calls else "none"
    }



def synthesizer_node(state: AgentState) -> AgentState:
    """Takes tool results and produces a clean final answer."""
    synth_prompt = """Based on the research above, provide a clear and concise answer.
- Summarize the key findings
- Be factual and direct
- Do not repeat the raw search results
- If sources were used, mention them naturally in your answer
"""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + \
               state["messages"] + \
               [HumanMessage(content=synth_prompt)]

    response = llm.invoke(messages)

    return {
        "messages": [response],
        "final_answer": response.content
    }



def should_use_tool(state: AgentState) -> str:
    """Decides next step after agent node."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"

    return "synthesize"
