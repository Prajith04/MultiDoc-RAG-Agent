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

### 1. Agents (`agents.py`)
- Creates a tool-calling agent using LangChain
- Combines the language model, retrieval tools, and prompt templates
- Serves as the core decision-making component that decides when to use tools

### 2. Main Application (`main.py`)
- Provides a command-line interface for interacting with the RAG agent
- Maintains conversation history for contextual understanding
- Processes user queries and displays responses in a continuous loop

### 3. Prompts (`prompts.py`)
- Defines templates for structuring interactions with the language model
- Uses both custom templates and pre-configured prompts from LangChain Hub
- Ensures consistent formatting of retrieved context and queries

### 4. Query Vector Database (`query_vectordb.py`)
- Initializes language models (Mistral-SABA-24B and Llama-3.3-70B)
- Manages connections to the Qdrant vector database
- Provides document retrieval functionality based on semantic similarity

### 5. Document Storage (`store2db.py`)
- Handles document ingestion from PDF files (Samsung device manuals)
- Splits documents into semantic chunks for better retrieval
- Creates vector embeddings using Sentence Transformers
- Stores embeddings in Qdrant for fast similarity search

### 6. Tools (`tools.py`)
- Defines specialized tools for the agent:
  - Retriever tool for Samsung mobile documentation
  - Calculator tool for handling mathematical operations
- Configures tool descriptions to help the agent decide when to use them

### 7. Web Interface (`gradio_demo.py`)
- Creates an interactive web UI using Gradio
- Processes user inputs and displays formatted agent responses
- Shows tool usage and intermediate steps for transparency
- Provides example queries to demonstrate system capabilities

## How to Use

### 1. Setup
- Ensure all dependencies are installed:
  ```bash
  pip install -r requirements.txt
  ```
- Configure environment variables in a `.env` file:
  ```
  GROQ_API_KEY=your_groq_api_key
  QDRANT_URL=your_qdrant_url
  QDRANT_API_KEY=your_qdrant_api_key
  ```

### 2. Run the Application
- For command-line interface:
  ```bash
  python main.py
  ```
- For web interface:
  ```bash
  python gradio_demo.py
  ```

### 3. Document Storage
- To ingest new documents or update the vector database:
  ```bash
  python store2db.py
  ```
- Current documents include Samsung device manuals (A16, S25, Fold)

### 4. Docker Deployment
- Build and run using Docker:
  ```bash
  docker build -t multidoc-rag-agent .
  docker run -p 7860:7860 multidoc-rag-agent
  ```
- Access the web interface at http://localhost:7860

## Dependencies
- **Python**: The primary programming language used for the project
- **LangChain**: Framework for building applications with language models
- **Groq**: API provider for accessing powerful language models
- **Qdrant**: Vector database for storing and retrieving document embeddings
- **Sentence Transformers**: Library for creating document embeddings
- **Gradio**: Framework for creating web interfaces for machine learning models

## Example Use Case
1. A user queries the system about configuring dark mode on their Samsung S25
2. The agent identifies this as a documentation retrieval task
3. The retrieval tool fetches relevant passages from the S25 manual
4. The language model generates a clear, step-by-step response
5. The user receives accurate instructions with context from the official manual

## License
This project is licensed under the MIT License.

