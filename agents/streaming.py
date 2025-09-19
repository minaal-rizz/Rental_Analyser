# streaming_agent.py
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from .reasoning import process_query

load_dotenv()

# -------------------------
# Streaming Agent
# -------------------------
streaming_agent = Agent(
    name="StreamingResponseAgent",
    model=Groq(id="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY")),
    description="Streams reasoning responses back to the user in real-time.",
    instructions=[
        "Take the final answer from the Reasoning Agent.",
        "Stream the output token by token instead of returning all at once.",
        "Ensure responses remain structured and clear.",
    ],
)


# -------------------------
# Streaming Function
# -------------------------
def stream_response(user_query: str):
    """
    Stream the reasoning agent's output for a given query.

    Args:
        user_query (str): The user’s natural language query.

    Returns:
        str: The final collected response after streaming is complete.

    Process:
        1. Sends the user query to the `streaming_agent` for processing.
        2. Uses streaming mode so tokens are returned incrementally instead of all at once.
        3. Prints each token as it arrives (simulating real-time chat experience).
        4. Collects the tokens into a final response string.
        5. Returns the fully assembled response after streaming completes.

    Example:
        >>> response = stream_response("find me the yearly rentals of all the branches in Lahore")
        >>> print(response)
        "The yearly rentals for Lahore branches are ..."
    """
    print(f"\n❓ User Query: {user_query}\n")

    # Get generator from LLM with streaming
    response_gen = streaming_agent.run(
        f"Process this query with reasoning: {user_query}\n"
        "Return the answer in a structured, clear format.",
        stream=True
    )

    # Print tokens as they arrive
    final_response = ""
    for chunk in response_gen:
        # Extract content from RunContentEvent
        content = getattr(chunk, 'content', '') if hasattr(chunk, 'content') else str(chunk)
        print(content, end="", flush=True)
        final_response += content

    print("\n\n✅ Streaming complete.\n")
    return final_response


# -------------------------
# Test Script
# -------------------------
if __name__ == "__main__":
    """
    Test script to validate streaming functionality.

    Runs a series of example queries through `stream_response` and prints the
    collected final answers for verification.

    Example Queries:
        1. Yearly rentals of Lahore branches.
        2. Branch with rental expiry within the next 2 months.
        3. Branch with the highest rent in Islamabad.
    """
    queries = [
        "find me the yearly rentals of all the branches in Lahore",
        "which branch has a rental expiry coming within the next two months",
        "which branch has the most rent in the city of Islamabad",
    ]

    for q in queries:
        answer = stream_response(q)
        print(f"📦 Final Answer (collected): {answer}\n{'-'*60}")
