---
title: MultiDoc-RAG-Agent
emoji: 💻
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.28.0
app_file: gradio_demo.py
pinned: false
---
# MultiDoc-RAG-Agent

## Overview
The MultiDoc-RAG-Agent is a Retrieval-Augmented Generation (RAG) system designed to interact with users, retrieve relevant documents, and provide intelligent responses. It leverages advanced language models, vector databases, and tools to process queries effectively. This system is particularly useful for scenarios requiring document retrieval and contextual understanding, such as customer support, research assistance, and knowledge management.

## Components

### 1. Agents
- **File**: `agents.py`
- **Description**: Defines the `rag_agent` function, which creates a tool-calling agent using a language model, tools, and a prompt. The agent is responsible for orchestrating interactions between the user, tools, and the language model to generate accurate and contextually relevant responses.

### 2. Main Application
- **File**: `main.py`
- **Description**: Implements a command-line interface for interacting with the RAG agent. It processes user queries, maintains a chat history, and ensures seamless communication between the user and the agent. This serves as the entry point for users to interact with the system.

### 3. Prompts
- **File**: `prompts.py`
- **Description**: Contains functions to generate prompts for the agent and retriever using templates and a hub-pulled prompt. These prompts guide the language model in understanding the context and generating appropriate responses.

### 4. Query Vector Database
- **File**: `query_vectordb.py`
- **Description**: Handles vector database interactions, initializes chat models, and provides a function to retrieve documents based on similarity. This component ensures efficient and accurate retrieval of relevant documents from the vector database.

### 5. Document Storage
- **File**: `store2db.py`
- **Description**: Loads PDF documents, splits them into smaller chunks, and stores them in a Qdrant vector database. This enables the system to handle large documents and retrieve specific sections relevant to user queries.

### 6. Tools
- **File**: `tools.py`
- **Description**: Defines tools for the agent, including a retriever tool for Samsung mobile-related queries and a calculator tool. These tools extend the agent's capabilities, allowing it to perform specialized tasks.

## How to Use

### 1. Setup
- Ensure all dependencies are installed.
- Configure environment variables in a `.env` file. For example:
  - `GROQ_API_KEY`: API key for the language model.
  - `QDRANT_URL`: URL for the Qdrant vector database.
  - `QDRANT_API_KEY`: API key for the Qdrant vector database.

### 2. Run the Application
- Execute `main.py` to start the command-line interface.
- Enter queries to interact with the agent and retrieve intelligent responses.

### 3. Document Storage
- Use `store2db.py` to load and store documents in the vector database. This step is essential for preparing the system to handle user queries effectively.

## Dependencies
- **Python**: The primary programming language used for the project.
- **LangChain**: A framework for building applications with language models.
- **Qdrant**: A vector database for storing and retrieving document embeddings.
- **HuggingFace**: A library for natural language processing and machine learning models.
- **dotenv**: A library for managing environment variables.

## Example Use Case
1. A user queries the system about a specific topic related to Samsung mobile devices.
2. The agent retrieves relevant documents from the vector database using `query_vectordb.py`.
3. The language model processes the retrieved documents and generates a coherent response.
4. The user receives an intelligent and contextually accurate answer.

## License
This project is licensed under the MIT License.

