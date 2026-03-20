AI-Research-Agent

Live demo link: https://ai-research-agent-zt7nhk29psyy5rcdm6w3ow.streamlit.app/
An intelligent research assistant built with LangGraph, LLaMA 3.1, and Groq. The agent autonomously decides which tool to use based on your question.

Features

- Multi-step reasoning with LangGraph state graph
  
- Web search tool (Tavily) for current information
  
- RAG tool (FAISS and Sentence Transformers) for uploaded documents
  
- Conversation memory across turns
  
- Input/output guardrails (prompt injection detection, hallucination flagging)
  
- Clean Streamlit chat UI with source citations



Tech Stack

- LangGraph — agent orchestration and state management
  
- LangChain — tool/LLM integration layer
  
- LLaMA 3.1 via Groq — fast LLM inference
  
- Tavily — real-time web search API
  
- FAISS and Sentence Transformers — vector search for documents
  
- Streamlit — chat UI


Usage
- Ask any question : agent searches the web automatically
  
- Upload `.txt` or `.pdf` files in the sidebar to agent searches your documents
  
- Sources and tool usage are shown for every response
