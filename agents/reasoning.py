# reasoning.py
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from .retrieval import retrieve_agreements
from .query import interpret_query

load_dotenv()

# -------------------------
# Reasoning Agent
# -------------------------
reasoning_agent = Agent(
    name="ReasoningAgent",
    model=Groq(id="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY")),
    description="Aggregates and reasons over rental agreement data to answer user queries.",
    instructions=[
        "Take parsed intent and filters from the Query Interpreter.",
        "Use the Retrieval Agent to fetch relevant rental agreements.",
        "Perform reasoning: sum rents, find max rent, list expiries within timeframe, etc.",
        "Return final structured answer with reasoning steps.",
        "Always return a clean, human-readable answer.",
        "Use markdown lists and tables if needed.",
        "Do not repeat entries. Ensure numbers are formatted with commas.",
        "Answer should be structured as:\n"
        "### Answer\n<final result>\n\n"
        "### Reasoning\n<steps>\n"
    ],
)


# -------------------------
# Reasoning Function
# -------------------------
def process_query(user_query: str):
    """
    Full end-to-end query processing pipeline for reasoning.

    Workflow:
        1. Interpret the user query (intent + filters).
        2. Retrieve relevant agreements from Pinecone.
        3. Pass results + parsed intent into the Reasoning Agent.
        4. Get back a structured, human-readable answer.

    Args:
        user_query (str): The natural language query from the user.

    Returns:
        str: The Reasoning Agent’s structured answer with reasoning steps.
    """
    # Step 1: Interpret
    parsed_result = interpret_query(user_query)
    parsed_text = str(parsed_result["parsed"])  # Convert RunOutput to string

    # Step 2: Retrieve agreements
    docs = retrieve_agreements(user_query, top_k=10)

    # Step 3: Hand reasoning to LLM
    reasoning_prompt = f"""
    User Query: {user_query}
    Parsed Intent: {parsed_text}
    Retrieved Agreements: {docs}

    Based on the above:
    - If intent is 'rent_retrieval': list branch → rent and sum total.
    - If intent is 'expiry_check': list branches with expiry within timeframe.
    - If intent is 'comparison': compare rents across cities/branches.
    - Otherwise: return the most relevant agreements with explanation.

    Answer clearly and structured.
    """

    response = reasoning_agent.run(reasoning_prompt)
    return response


# -------------------------
# Test Script
# -------------------------
if __name__ == "__main__":
    """
    Run test queries through the reasoning pipeline.
    Prints both the query and the structured reasoning output.
    """
    test_queries = [
        "find me the yearly rentals of all the branches in Lahore",
        "which branch has a rental expiry coming within the next two months",
        "which branch has the most rent in the city of Islamabad",
    ]

    for q in test_queries:
        print(f"\n❓ Query: {q}")
        result = process_query(q)
        print(f"🧠 Reasoned Answer:\n{result}\n")
