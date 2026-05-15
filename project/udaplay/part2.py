import json
import os
from datetime import datetime

from chromadb.api.types import QueryResult
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from tavily import TavilyClient

import chromadb
from lib.agents import Agent
from lib.tooling import tool

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE")

chroma_client = chromadb.PersistentClient(path="chromadb")

# Use ChromaDB's built-in OpenAI embedding function, pointed at the Vocareum proxy.
# The collection was stored with the 'openai' type, so this avoids a type conflict.
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    api_base=API_BASE,
    model_name="text-embedding-ada-002",
)
collection = chroma_client.get_collection("udaplay", embedding_function=embedding_fn)


@tool
def retrieve_game(query: str) -> list:
    """
    Semantic search: Finds one result in the vector DB.

    Args:
       - query: a question about game industry

    Returns:
       You'll receive results as list. Each element contains:
       - Platform: like Game Boy, Playstation 5, Xbox 360...
       - Name: Name of the Game
       - YearOfRelease: Year when that game was released for that platform
       - Description: Additional details about the game

    This database is not comprehensive; it only contains a few sample entries. Further investigation using other
    resources might be necessary to answer questions.
    """
    # We don't need a state machine for this.
    results: QueryResult = collection.query(query_texts=[query], n_results=1)
    documents = results["documents"][0] if results["documents"] else []
    return documents


# You might use an LLM as judge in this tool to evaluate the performance
# You need to prompt that LLM with something like:
# "Your task is to evaluate if the documents are enough to respond the query. "
# "Give a detailed explanation, so it's possible to take an action to accept it or not."
# Use EvaluationReport to parse the result
#
@tool
def evaluate_retrieval(question: str, retrieved_docs: list) -> dict:
    """
    Based on the user's question and on the list of retrieved documents,
    it will analyze the usability of the documents to respond to that question.

    Args:
       - question: original question from user
       - retrieved_docs: retrieved documents most similar to the user query
                        in the Vector Database

    Returns:
       The result includes:
       - useful: whether the documents are useful to answer the question
       - description: description about the evaluation result
    """
    evaluation_agent: Agent = Agent(
        instructions="""
        Your task is to evaluate whether retrieved documents are useful in answering the given question.
        If the supplied documents do not contain sufficient information to answer the question, respond with
        a suggestion to do a web search.
        """
    )
    agent_query = {question: question, retrieved_docs: str(retrieved_docs)}
    result: dict = evaluation_agent.invoke(str(agent_query)).get_final_state()
    result_content = result["messages"][-1].content
    return result_content


# Use Tavily client to search the web
#
@tool
def game_web_search(question: str) -> str:
    """
    Performs semantic web search for information about games.

    Args:
       - question: a question about game industry

    Returns:
       Search results related to the game industry question
    """
    client = TavilyClient(api_key=TAVILY_API_KEY)

    # Perform the search
    search_result = client.search(
        query=question,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )

    # Return clean JSON with answer + results (sources), no metadata
    formatted_results = {
        "answer": search_result.get("answer", ""),
        "results": [
            {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("snippet")}
            for r in search_result.get("results", [])[:3]  # Top 3 results
        ],
    }

    return json.dumps(formatted_results)
