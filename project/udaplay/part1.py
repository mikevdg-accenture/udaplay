import importlib.util
import json
import logging
import os
import sys

from chromadb.api import ClientAPI
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI


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


printHeading("PART 01 - OFFLINE RAG: Setting up Vector Database")

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
print("Loading games from 'games' directory and adding to collection...")
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

    print(f"Successfully added {file_count} games to the collection!")
else:
    print(
        f"Warning: '{data_dir}' directory not found. Please check the directory structure."
    )

print("Part 01 Complete: Vector Database is ready!")
