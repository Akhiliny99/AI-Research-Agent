import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from langchain_core.messages import HumanMessage
from agent.graph import research_agent
from agent.guardrails import check_input, check_output, format_response


st.set_page_config(
    page_title="Research Agent",
    page_icon="🔍",
    layout="wide"
)


st.title("🔍 AI Research Agent")
st.caption("Powered by LangGraph · LLaMA 3.1 · Groq · Tavily · FAISS")

st.markdown("---")


with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("📄 Upload Documents (RAG)")
    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf files",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
        os.makedirs(docs_dir, exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(docs_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"✅ {len(uploaded_files)} file(s) ready for RAG search")

    st.markdown("---")
    st.subheader("🧠 How it works")
    st.markdown("""
    1. You ask a question
    2. Agent decides which tool to use
    3. **Web search** → current info from internet
    4. **RAG search** → your uploaded documents
    5. **Direct** → answers from LLM knowledge
    6. Sources are always cited
    """)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]

            col1, col2 = st.columns(2)
            with col1:
                tool_colors = {
                    "web_search": "🌐",
                    "rag_search": "📄",
                    "none": "🧠"
                }
                tool = meta.get("tool_used", "none")
                emoji = tool_colors.get(tool, "🔧")
                st.caption(f"{emoji} Tool used: `{tool}`")

            with col2:
                sources = meta.get("sources", [])
                if sources and sources != ["uploaded documents"]:
                    st.caption(f"📎 {len(sources)} source(s) found")

            if meta.get("has_warnings"):
                with st.expander("⚠️ Warnings"):
                    for w in meta["warnings"]:
                        st.warning(w)

            sources = meta.get("sources", [])
            if sources and sources != ["uploaded documents"]:
                with st.expander("🔗 Sources"):
                    for url in sources:
                        st.markdown(f"- {url}")


if query := st.chat_input("Ask me anything..."):

   
    is_safe, reason = check_input(query)
    if not is_safe:
        with st.chat_message("assistant"):
            st.error(f"❌ {reason}")
        st.stop()

   
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                st.session_state.chat_history.append(
                    HumanMessage(content=query)
                )

                result = research_agent.invoke({
                    "messages": st.session_state.chat_history,
                    "sources": [],
                    "final_answer": "",
                    "tool_used": ""
                })

                raw_answer = result.get("final_answer", "")
                tool_used = result.get("tool_used", "none")
                sources = result.get("sources", [])

                
                cleaned_answer, warnings = check_output(raw_answer, tool_used)
                response = format_response(cleaned_answer, sources, tool_used, warnings)

               
                st.markdown(cleaned_answer)

                
                col1, col2 = st.columns(2)
                with col1:
                    tool_colors = {"web_search": "🌐", "rag_search": "📄", "none": "🧠"}
                    emoji = tool_colors.get(tool_used, "🔧")
                    st.caption(f"{emoji} Tool used: `{tool_used}`")
                with col2:
                    if sources and sources != ["uploaded documents"]:
                        st.caption(f"📎 {len(sources)} source(s) found")

                if response["has_warnings"]:
                    with st.expander("⚠️ Warnings"):
                        for w in warnings:
                            st.warning(w)

                if sources and sources != ["uploaded documents"]:
                    with st.expander("🔗 Sources"):
                        for url in sources:
                            st.markdown(f"- {url}")

                
                st.session_state.chat_history.append(
                    result["messages"][-1]
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": cleaned_answer,
                    "metadata": response
                })

            except Exception as e:

                st.error(f"Agent error: {str(e)}")
