#!/usr/bin/env python3
"""
Udaplay Project - AI Research Agent for Video Game Industry

This script combines two parts:
1. Part 01 - Offline RAG: Build VectorDB using ChromaDB with game data
2. Part 02 - Agent: Build an agent that uses RAG and web search to answer questions
"""

import importlib.util
import json
import logging
import os
import sys

# Import necessary libraries for Part 01
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

# Enable debug logging for OpenAI
logging.basicConfig(level=logging.DEBUG)

# ============================================================================
# Custom Embedding Function with Vocareum Support
# ============================================================================


class VocariumEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function that uses Vocareum endpoint with proper base_url."""

    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base
        # Create OpenAI client with proper base_url
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002", input=input
        )
        return [item.embedding for item in response.data]


print("=" * 80)
print("PART 01 - OFFLINE RAG: Setting up Vector Database")
print("=" * 80)

# Only needed for Udacity workspace
# Handle pysqlite3 compatibility issue
if importlib.util.find_spec("pysqlite3") is not None:
    import pysqlite3

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


# Load environment variables from .env file
print("\nLoading environment variables...")
load_dotenv()

# Initialize ChromaDB as a persistent client
print("Initializing ChromaDB client...")
chroma_client: ClientAPI = chromadb.PersistentClient(path="chromadb")

# Create an embedding function using OpenAI embeddings
print("Creating embedding function with OpenAI...")
api_base = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")
print(f"  API Base URL: {api_base}")
print(f"  API Key set: {bool(api_key)}")
embedding_fn = VocariumEmbeddingFunction(api_key=api_key, api_base=api_base)

# Create a collection named "udaplay"
print("Creating 'udaplay' collection...")
try:
    # Try to get existing collection first with our embedding function
    collection = chroma_client.get_collection(
        "udaplay", embedding_function=embedding_fn
    )
    print("Using existing 'udaplay' collection...")
except Exception:
    # Create new collection if it doesn't exist
    collection = chroma_client.create_collection(
        name="udaplay", embedding_function=embedding_fn, get_or_create=True
    )
    print("Created new 'udaplay' collection...")

# Load all JSON files from the "games" directory and add them to the collection
print("\nLoading games from 'games' directory and adding to collection...")
data_dir = "games"

if os.path.exists(data_dir):
    file_count = 0
    for file_name in sorted(os.listdir(data_dir)):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            game = json.load(f)

        # Create document content from game metadata
        # Format: [Platform] Game Name (Year) - Description
        content = f"[{game['Platform']}] {game['Name']} ({game['YearOfRelease']}) - {game['Description']}"

        # Use file name (like 001) as ID
        doc_id = os.path.splitext(file_name)[0]

        # Add document to collection
        collection.add(ids=[doc_id], documents=[content], metadatas=[game])

        file_count += 1
        print(
            f"  ✓ Added: {game['Name']} ({game['Platform']}, {game['YearOfRelease']})"
        )

    print(f"\nSuccessfully added {file_count} games to the collection!")
else:
    print(
        f"Warning: '{data_dir}' directory not found. Please check the directory structure."
    )

print("\n" + "-" * 80)
print("Part 01 Complete: Vector Database is ready!")
print("-" * 80)

# ============================================================================
# PART 02 - AGENT: Set Up Tools and Agent
# ============================================================================

print("\n" + "=" * 80)
print("PART 02 - AGENT: Setting up Tools and Agent")
print("=" * 80)

# TODO: Import the necessary libs
# For example:
# import os
#
# from lib.agents import Agent
# from lib.llm import LLM
# from lib.messages import UserMessage, SystemMessage, ToolMessage, AIMessage
# from lib.tooling import tool

# TODO: Load environment variables
# load_dotenv()
#
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# =============================================================================
# Tools Section
# =============================================================================

print("\nSetting up tools for the agent...")

# TODO: Create retrieve_game tool
# It should use chroma client and collection you created
# chroma_client = chromadb.PersistentClient(path="chromadb")
# collection = chroma_client.get_collection("udaplay")
#
# @tool
# def retrieve_game(query: str) -> list:
#     """
#     Semantic search: Finds most results in the vector DB
#
#     Args:
#        - query: a question about game industry
#
#     Returns:
#        You'll receive results as list. Each element contains:
#        - Platform: like Game Boy, Playstation 5, Xbox 360...
#        - Name: Name of the Game
#        - YearOfRelease: Year when that game was released for that platform
#        - Description: Additional details about the game
#     """
#     pass


# TODO: Create evaluate_retrieval tool
# You might use an LLM as judge in this tool to evaluate the performance
# You need to prompt that LLM with something like:
# "Your task is to evaluate if the documents are enough to respond the query. "
# "Give a detailed explanation, so it's possible to take an action to accept it or not."
# Use EvaluationReport to parse the result
#
# @tool
# def evaluate_retrieval(question: str, retrieved_docs: list) -> dict:
#     """
#     Based on the user's question and on the list of retrieved documents,
#     it will analyze the usability of the documents to respond to that question.
#
#     Args:
#        - question: original question from user
#        - retrieved_docs: retrieved documents most similar to the user query
#                         in the Vector Database
#
#     Returns:
#        The result includes:
#        - useful: whether the documents are useful to answer the question
#        - description: description about the evaluation result
#     """
#     pass


# TODO: Create game_web_search tool
# Please use Tavily client to search the web
#
# @tool
# def game_web_search(question: str) -> str:
#     """
#     Performs semantic web search for information about games.
#
#     Args:
#        - question: a question about game industry
#
#     Returns:
#        Search results related to the game industry question
#     """
#     pass


# =============================================================================
# Agent Section
# =============================================================================

print("Setting up the agent...")

# TODO: Create your Agent abstraction using StateMachine
# Equip with an appropriate model
# Craft a good set of instructions
# Plug all Tools you developed
#
# Example structure:
# agent = Agent(
#     name="UdaPlay Agent",
#     system_prompt="""You are UdaPlay, an AI Research Agent for the video game industry.
#     Your responsibilities include:
#     1. Answering questions using internal knowledge (RAG)
#     2. Searching the web when needed
#     3. Maintaining conversation state
#     4. Returning structured outputs
#     5. Storing useful information for future use
#
#     Use the available tools appropriately to provide accurate and helpful information.
#     """,
#     tools=[retrieve_game, evaluate_retrieval, game_web_search],
#     llm=LLM(model="gpt-4-turbo")
# )


# TODO: Invoke your agent with test questions
# Example invocations:
#
# test_questions = [
#     "When Pokémon Gold and Silver was released?",
#     "Which one was the first 3D platformer Mario game?",
#     "Was Mortal Kombat X released for Playstation 5?"
# ]
#
# for question in test_questions:
#     print(f"\nQuestion: {question}")
#     result = agent.invoke(question)
#     print(f"Answer: {result}")


# =============================================================================
# Optional Advanced Features
# =============================================================================

# TODO: Update your agent with long-term memory
# TODO: Convert the agent to be a state machine, with the tools being pre-defined nodes

print("\n" + "-" * 80)
print("Part 02: Setup complete!")
print("Next steps: Implement the TODOs in Part 02 to activate the agent.")
print("-" * 80)

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Udaplay Project - Both parts initialized successfully!")
    print("=" * 80)
    print("\nPart 01 Status: ✓ Vector Database is ready")
    print("Part 02 Status: ⏳ Awaiting implementation of TODO items")
    print("\nTo complete the project, implement the TODO comments in Part 02.")
    print("=" * 80 + "\n")
