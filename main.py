#!/usr/bin/env python3
"""
Udaplay Project - AI Research Agent for Video Game Industry

This script combines two parts:
1. Part 01 - Offline RAG: Build VectorDB using ChromaDB with game data
2. Part 02 - Agent: Build an agent that uses RAG and web search to answer questions
"""

import logging

import part1
import part2
from lib.agents import Agent

# Enable debug logging for OpenAI
# logging.basicConfig(level=logging.DEBUG)

# =============================================================================
# Optional Advanced Features
# =============================================================================

# TODO: Update your agent with long-term memory
# TODO: Convert the agent to be a state machine, with the tools being pre-defined nodes

if __name__ == "__main__":
    vector_db = part1.setup_vector_db()

    # Forgive me for this.
    part2.vector_db = vector_db

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
        tools=[part2.retrieve_game, part2.evaluate_retrieval, part2.game_web_search],
        model_name="gpt-4-turbo",
    )

    test_questions = [
        "When were Pokémon Gold and Silver was released?",
        "Which one was the first 3D platformer Mario game?",
        "Was Mortal Kombat X released for Playstation 5?",
        # "Tell me all about Donkey Kong."
    ]

    for question in test_questions:
        agent.reset_session()
        print(f"Question: {question}")
        agent.invoke(question)
        print(f"Answer: {agent.get_answer()}")
        agent.pretty_print_memory()
