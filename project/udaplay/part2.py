import os

from lib.agents import Agent
from lib.llm import LLM
from lib.messages import AIMessage, SystemMessage, ToolMessage, UserMessage
from lib.tooling import tool

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# It should use chroma client and collection you created
chroma_client = chromadb.PersistentClient(path="chromadb")
collection = chroma_client.get_collection("udaplay")


@tool
def retrieve_game(query: str) -> list:
    """
    Semantic search: Finds most results in the vector DB

    Args:
       - query: a question about game industry

    Returns:
       You'll receive results as list. Each element contains:
       - Platform: like Game Boy, Playstation 5, Xbox 360...
       - Name: Name of the Game
       - YearOfRelease: Year when that game was released for that platform
       - Description: Additional details about the game
    """
    raise NotImplementedError()


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
    raise NotImplementedError()


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
    raise NotImplementedError()
