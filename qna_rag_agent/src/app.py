"""
Main application for the RAG-powered multi-agent Q&A system.
"""
import os
import argparse
from dotenv import load_dotenv
from src.utils.document_loader import DocumentLoader
from src.utils.vector_store import VectorStore
from src.utils.llm_service import LLMService
from src.agents.agent_orchestrator import AgentOrchestrator

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG-powered multi-agent Q&A system")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing documents")
    parser.add_argument("--chunk_size", type=int, default=500, help="Size of document chunks")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Overlap between document chunks")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment variables.")
        print("Please create a .env file with your Groq API key or set it as an environment variable.")
        exit(1)
    
    # Get absolute path to data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, args.data_dir)
    
    # Initialize document loader
    document_loader = DocumentLoader(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Load and split documents
    print(f"Loading documents from {data_dir}...")
    documents = document_loader.load_and_split_documents(data_dir)
    print(f"Loaded {len(documents)} document chunks")
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Create index
    print("Creating vector index...")
    vector_store.create_index(documents)
    
    # Initialize LLM service
    llm_service = LLMService(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    
    # Initialize agent orchestrator
    agent = AgentOrchestrator(
        vector_store=vector_store,
        llm_service=llm_service
    )
    
    # Interactive CLI
    print("\nRAG-powered multi-agent Q&A system")
    print("Type 'exit' to quit\n")
    
    while True:
        # Get user query
        query = input("Enter your question: ")
        
        if query.lower() == 'exit':
            break
        
        # Process query
        result = agent.process_query(query)
        
        # Display results
        print("\n" + "="*50)
        print(f"Query: {result['query']}")
        print(f"Decision: {result['decision']}")
        
        if result['tool_used']:
            print(f"Tool: {result['tool_used']}")
            print(f"Tool Input: {result['tool_input']}")
            print(f"Tool Output: {result['tool_output']}")
        else:
            print("\nRetrieved Context:")
            for i, chunk in enumerate(result['retrieved_context']):
                print(f"\nChunk {i+1} (from {chunk['metadata']['source']}):")
                print(f"Score: {chunk['score']:.4f}")
                print("-"*40)
                print(chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'])
        
        print("\nAnswer:")
        print(result['answer'])
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
