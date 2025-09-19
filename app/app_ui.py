# app.py 
import streamlit as st
import json
import sys
import os

# Add parent directory to path to find agents module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.query import query_interpreter_agent
from agents.retrieval import retrieve_agreements
from agents.reasoning import reasoning_agent

st.set_page_config(page_title="Rental Agreement Assistant", layout="wide")
st.title("🏢 Rental Agreement Assistant")

query = st.text_input("Ask a question about rental agreements:")

if query:
    st.write("### Response:")

    # --- Step 1: Interpret query ---
    parsed = query_interpreter_agent.run(query)
    intent_str = parsed.content
    try:
        intent = json.loads(intent_str)
    except json.JSONDecodeError:
        intent = {"intent": "general_search", "filters": {}}
    st.json(intent)

    # --- Step 2: Retrieve docs ---
    docs = retrieve_agreements(query, top_k=5)

    # --- Step 3: Reasoning (STREAMING) ---
    st.write("### Streaming Answer:")
    reasoning_placeholder = st.empty()

    streamed_text = ""
    response_gen = reasoning_agent.run(
        f"""
        User Query: {query}
        Parsed Intent: {intent}
        Retrieved Agreements: {docs}

        Provide the answer in this format:
        ### Answer
        - Clean bullet list or table
        - No duplicates
        - Format numbers with commas

        ### Reasoning
        - Short explanation of how result was derived
        """,
        stream=True
    )
    
    for chunk in response_gen:
        # Extract content from RunContentEvent
        content = getattr(chunk, 'content', '') if hasattr(chunk, 'content') else str(chunk)
        streamed_text += content
        reasoning_placeholder.markdown(streamed_text)

    st.success("✅ Done Streaming!")





