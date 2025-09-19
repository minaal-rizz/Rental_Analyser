

# app.py
import streamlit as st
from agents.query import query_interpreter_agent
from agents.retrieval import retrieve_agreements
from agents.reasoning import reasoning_agent

# app/chat_pipeline.py

import json

def process_query(query: str):
    """Main pipeline: interpret → retrieve → reason
    
    Steps:
        1. Interpret the query using the Query Interpreter Agent.
        2. Parse the output (JSON format) into intent and filters.
        3. Retrieve relevant rental agreements based on query and filters.
        4. Use the Reasoning Agent (non-streaming) to provide a structured final answer.
"""
    # Step 1: Interpret query
    parsed = query_interpreter_agent.run(query)
    intent_str = parsed.content
    
    # Parse the JSON string to get the actual intent object
    try:
        intent = json.loads(intent_str)
    except json.JSONDecodeError:
        intent = {"intent": "general_search", "filters": {}}

    # Step 2: Retrieve documents
    filters = intent.get("filters", {})
    docs = retrieve_agreements(query, top_k=5)

    # Step 3: Reasoning agent (non-streaming for return value)
    answer = reasoning_agent.run(
        f"User Query: {query}\n"
        f"Parsed Intent: {intent}\n"
        f"Retrieved Agreements: {docs}\n"
        f"Provide a clear, structured answer.",
        stream=False
    )

    return {
        "query": query,
        "parsed": intent,
        "docs": docs,
        "answer": answer.content,
    }


st.set_page_config(page_title="Rental Agreement Assistant", layout="wide")
st.title("🏢 Rental Agreement Assistant")

query = st.text_input("Ask a question about rental agreements:")

if query:
    st.write("### Response:")
    # Step 1: Interpret query
    parsed = query_interpreter_agent.run(query)
    intent_str = parsed.content
    
    # Parse the JSON string to get the actual intent object
    try:
        intent = json.loads(intent_str)
    except json.JSONDecodeError:
        intent = {"intent": "general_search", "filters": {}}
    
    st.json(intent)

    # Step 2: Retrieve docs
    filters = intent.get("filters", {})
    city = filters.get("city")
    docs = retrieve_agreements(query, top_k=5)

    # Step 3: Reasoning (STREAMING)
    st.write("### Streaming Answer:")
    reasoning_placeholder = st.empty()

    streamed_text = ""
    for event in reasoning_agent.run_stream(f"User Query: {query}\nParsed Intent: {intent}\nRetrieved Agreements: {docs}\n Provide a clear, structured answer based on the retrieved data."):
        if event.delta:  # only when new tokens come
            streamed_text += event.delta
            reasoning_placeholder.markdown(streamed_text)

    st.success("✅ Done Streaming!")
