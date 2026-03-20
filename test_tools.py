from agent.tools import web_search_tool, rag_search

# Test web search
print("Testing web search...")
results = web_search_tool.invoke({"query": "What is LangGraph used for?"})
print("Web search OK:", str(results)[:150], "...\n")

# Test RAG (no docs yet — should return helpful message)
print("Testing RAG tool...")
result = rag_search.invoke("test query")
print("RAG OK:", result)