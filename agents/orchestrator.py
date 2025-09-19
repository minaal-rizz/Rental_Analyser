# orchestrator.py
import os
from dotenv import load_dotenv
from .query import interpret_query
from .retrieval import retrieve_agreements
from .reasoning import reasoning_agent
from .streaming import streaming_agent

load_dotenv()

# -------------------------
# Orchestrator
# -------------------------
def process_user_query(user_query: str):
    """
    Orchestrates the end-to-end query processing pipeline.

    Flow:
        1. Interpret the query → classify intent & entities.
        2. Retrieve relevant rental agreements from Pinecone.
        3. Construct a reasoning prompt combining query + intent + retrieved docs.
        4. Stream the reasoning agent’s answer back to the user.

    Args:
        user_query (str): The raw query asked by the user.

    Returns:
        str: Final structured answer streamed from the reasoning agent.
    """
    print(f"\n❓ User Query: {user_query}\n")

    # Step 1: Interpret Query
    parsed_result = interpret_query(user_query)
    parsed_text = str(parsed_result["parsed"])  # Convert RunOutput to string
    print(f"🧩 Parsed Intent: {parsed_text}\n")

    # Step 2: Retrieve Relevant Docs
    docs = retrieve_agreements(user_query, top_k=10)
    print("📂 Retrieved Agreements:")
    for d in docs:
        print(
            f" - {d['branch_name']} | {d['city']} | Rent: {d['yearly_rent']} | Expiry: {d['expiry_date']}"
        )

    # Step 3: Reasoning Prompt
    reasoning_prompt = f"""
    User Query: {user_query}
    Parsed Intent: {parsed_text}
    Retrieved Agreements: {docs}

    Based on this:
    - If intent is 'rent_retrieval': list branch → rent and sum total.
    - If intent is 'expiry_check': list branches with expiry within timeframe.
    - If intent is 'comparison': compare rents across cities/branches.
    - Otherwise: return the most relevant agreements with explanation.

    Answer clearly and structured.
    """

    # Step 4: Streaming Response
    print("\n💬 Streaming Answer:\n")
    response_gen = streaming_agent.run(reasoning_prompt, stream=True)

    final_response = ""
    for chunk in response_gen:
        # Extract content safely from streaming chunks
        content = getattr(chunk, 'content', '') if hasattr(chunk, 'content') else str(chunk)
        print(content, end="", flush=True)
        final_response += content

    print("\n\n✅ Done.\n")
    return final_response


# -------------------------
# Test Script
# -------------------------
if __name__ == "__main__":
    """
    Manual test runner.
    Runs a few sample queries through the full pipeline and prints results.
    """
    queries = [
        "find me the yearly rentals of all the branches in Lahore",
        "which branch has a rental expiry coming within the next two months",
        "which branch has the most rent in the city of Islamabad",
    ]

    for q in queries:
        answer = process_user_query(q)
        print(f"\n📦 Final Answer: {answer}\n{'-'*60}")
