#!/usr/bin/env python3
"""
Udaplay Project - AI Research Agent for Video Game Industry

This script combines two parts:
1. Part 01 - Offline RAG: Build VectorDB using ChromaDB with game data
2. Part 02 - Agent: Build an agent that uses RAG and web search to answer questions
"""

import logging

import part2

# Enable debug logging for OpenAI
# logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    part2.run_project()

