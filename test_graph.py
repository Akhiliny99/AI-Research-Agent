from agent.graph import research_agent
from langchain_core.messages import HumanMessage

print("Testing agent graph...\n")

result = research_agent.invoke({
    "messages": [HumanMessage(content="What is LangGraph and why is it useful?")],
    "sources": [],
    "final_answer": "",
    "tool_used": ""
})

print("Answer:", result["final_answer"])
print("Tool used:", result.get("tool_used", "none"))
print("Sources:", result.get("sources", []))