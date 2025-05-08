
![image](https://github.com/user-attachments/assets/e320b07e-9ba9-4a78-88d7-78b5cfaa1199)

<img width="1352" alt="image" src="https://github.com/user-attachments/assets/3440938f-d665-4e36-8e6a-e5d11e8be372" />

## history
<img width="1352" alt="image" src="https://github.com/user-attachments/assets/54dc6245-e0b2-484f-9e6a-5ed1b82a2963" />




# RAG-Powered Multi-Agent Q&A System

This project implements an intelligent knowledge assistant that retrieves relevant information from a document collection, generates natural-language answers via an LLM, and orchestrates the retrieval and generation steps with an advanced agentic workflow. The system can handle both simple and complex queries, including mixed queries that combine factual questions with calculations.

## Architecture

The system is built with the following components:

1. **Document Loader**: Processes text files and splits them into chunks for vector indexing with configurable chunk size and overlap.
2. **Vector Store**: Creates embeddings for document chunks and enables semantic search using TF-IDF vectorization and cosine similarity.
3. **LLM Service**: Generates answers based on retrieved context using Groq's Llama3-8b-8192 model with context-aware prompting.
4. **Agent Orchestrator**: Routes queries to appropriate tools or the RAG pipeline based on query content, with special handling for mixed queries.
5. **Tools**: Specialized functions including a calculator tool (with mathematical operations and age calculations) and a dictionary tool (for word definitions).
6. **User Interface**: Includes both CLI and Streamlit web interfaces for interacting with the system, showing decision paths, retrieved context, and answers.

### Workflow

#### Agentic Routing
The system uses an intelligent routing mechanism to direct queries to the most appropriate processing pipeline:

1. **Pattern-Based Analysis**: The agent orchestrator analyzes the query using regex patterns to identify query intent and content type.
2. **System-Specific Detection**: Queries about the system itself, its architecture, or technical concepts are automatically routed to the RAG pipeline.
3. **Tool Matching**: Mathematical expressions, age calculations, and definition requests are routed to specialized tools.
4. **Mixed Query Detection**: Queries containing multiple intents (e.g., factual questions with calculations) are identified for hybrid processing.
5. **Fallback Mechanism**: When no specific pattern is matched, the query defaults to the RAG pipeline.

#### Standard Query Processing
1. User submits a question.
2. Agent analyzes the question to determine if it should use a specialized tool, the RAG pipeline, or a hybrid approach.
3. If keywords like "calculate", "define", or mathematical patterns are detected, the query is routed to the appropriate tool.
4. For general knowledge queries, the RAG pipeline retrieves relevant document chunks and passes them to the LLM for answer generation.
5. The system logs each decision step and returns the final answer along with processing details.

#### Mixed Query Processing
1. For queries containing both factual questions and calculations (e.g., "What is RAGent AI and what is the square root of 50?"):
2. The system identifies and extracts the mathematical component.
3. The mathematical part is processed by the calculator tool.
4. The general knowledge part is processed through the RAG pipeline.
5. Both results are combined and sent to the LLM to generate a comprehensive answer addressing all parts of the query.

## Key Design Choices

1. **Document Chunking**: Documents are split into smaller chunks with overlap to ensure context preservation while maintaining retrieval precision.
2. **Vector Search**: TF-IDF vectorization with cosine similarity is used for efficient semantic search, providing good performance without external dependencies.
3. **Intelligent Query Routing**: Advanced pattern matching and query analysis route queries to specialized tools, RAG pipeline, or hybrid processing based on query content and structure.
4. **Mixed Query Handling**: Special processing for queries that contain multiple intents (e.g., both factual questions and calculations), providing comprehensive answers to complex questions.
5. **LLM Integration**: Groq's Llama3-8b-8192 model is used for answer generation with enhanced context-aware prompting that includes source attribution and calculation results.
6. **Robust Error Handling**: Validation and error handling for various query types, with helpful error messages and fallback mechanisms.
7. **Transparency**: The system shows which path was taken (tool, RAG, or mixed) along with retrieved context, calculation steps, and reasoning process.

## Setup and Installation

### Prerequisites

- Python 3.8+
- Groq API key

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/pant-pragyan10/Rag-powered-QnA-Agent
   cd qna-rag-agent
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Running the Application

### Command Line Interface

Run the CLI version:

```
cd qna_rag_agent
python -m src.app
```

### Web Interface

Run the Streamlit web interface:

```
cd qna_rag_agent
streamlit run src/streamlit_app.py
```

## Sample Queries

## Note: RAGent AI is a fictional company used for demonstrating this assistant's capabilities.
### General Knowledge Queries 

- Q: What is RAGent AI?
- "What is the history of RAGent AI?"
- "What products does RAGent AI offer?"
- "Can you describe RAGent Search in more detail?"
- "What makes RAGent Assistant unique?"

### Tool-Based Queries
- "Calculate 25 * 16"
- "What is the square root of 50?"
- "How old am I if I was born in 1990?"
- "Define retrieval augmented generation"
- "What is the meaning of hallucination in AI?"

### Mixed Queries
- "What is RAGent AI and what is the square root of 50?"
- "How many employees does RAGent AI have and calculate 25 * 16"
- "Define RAG and what is 15 + 27?"

## Extending the System

- **Additional Tools**: Add more specialized tools by extending the `tools.py` file (e.g., web search, image analysis, code execution).
- **Document Formats**: Support additional document formats by enhancing the document loader (e.g., PDF, HTML, structured data).
- **Advanced Routing**: Implement ML-based classification for query routing instead of pattern matching.
- **Conversation Memory**: Add memory to the system to support follow-up questions and maintain conversation context.
- **Custom Embeddings**: Replace TF-IDF with domain-specific neural embeddings for improved semantic search.
- **Multi-hop Reasoning**: Enhance the RAG pipeline to answer questions requiring information from multiple documents.
- **Evaluation Framework**: Add automated testing and evaluation of answer quality and retrieval precision.
- **User Feedback Loop**: Implement a mechanism to collect user feedback and improve the system over time.
# RAG-Powered-QnA-Agent.
