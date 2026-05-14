#!/usr/bin/env python3
"""
Udaplay Project - AI Research Agent for Video Game Industry

This script combines two parts:
1. Part 01 - Offline RAG: Build VectorDB using ChromaDB with game data
2. Part 02 - Agent: Build an agent that uses RAG and web search to answer questions
"""

import logging

from lib.agents import Agent
from part2 import evaluate_retrieval, game_web_search, retrieve_game

# Enable debug logging for OpenAI
logging.basicConfig(level=logging.DEBUG)


def printHeading(heading: str):
    print("\n")
    print("=" * 80)
    print(heading)
    print("=" * 80)


# =============================================================================
# Optional Advanced Features
# =============================================================================

# TODO: Update your agent with long-term memory
# TODO: Convert the agent to be a state machine, with the tools being pre-defined nodes

if __name__ == "__main__":
    agent: Agent = Agent(
        instructions="""You are UdaPlay, an AI Research Agent for the video game industry.
        Your responsibilities include:
        1. Answering questions using internal knowledge (RAG)
        2. Searching the web when needed
        3. Maintaining conversation state
        4. Returning structured outputs
        5. Storing useful information for future use

        Use the available tools appropriately to provide accurate and helpful information.
        """,
        tools=[retrieve_game, evaluate_retrieval, game_web_search],
        model_name="gpt-4-turbo",
    )

    test_questions = [
        "When Pokémon Gold and Silver was released?",
        "Which one was the first 3D platformer Mario game?",
        "Was Mortal Kombat X released for Playstation 5?",
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = agent.invoke(question)
        print(f"Answer: {result}")
