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
# logging.basicConfig(level=logging.DEBUG)


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
        Politely decline to answer any question not related to video games.
        For all questions, first use the tool `retrieve_game` to attempt to answer the question.
        Then use the `evaluate_retrieval` tool to determine the quality of the answer provided.
        If the answer is not answered by the `retrieve_game` tool, use the `game_web_search` tool
        to search the web for the answer.
        Answer questions preferably in a simple question, unless the question is about Donkey Kong,
        in which case, do a web search and provide at least 200 lines of detailed lore about
        Donkey Kong.
        """,
        tools=[retrieve_game, evaluate_retrieval, game_web_search],
        model_name="gpt-4-turbo",
    )

    test_questions = [
        "When were Pokémon Gold and Silver was released?",
        "Which one was the first 3D platformer Mario game?",
        "Was Mortal Kombat X released for Playstation 5?",
    ]

    for question in test_questions:
        agent.reset_session()
        printHeading(f"Question: {question}")
        result = agent.invoke(question)
        print(f"Answer: {result}")
        agent.pretty_print_memory()
