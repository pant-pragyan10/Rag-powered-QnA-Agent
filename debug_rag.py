"""
Debug script for RAG system to test document loading and retrieval.
"""
import os
import sys
from qna_rag_agent.src.utils.document_loader import DocumentLoader
from qna_rag_agent.src.utils.vector_store import VectorStore

def main():
    # Set up paths
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qna_rag_agent", "data")
    print(f"Looking for data in: {data_dir}")
    
    # List files in data directory
    files = os.listdir(data_dir)
    print(f"Found files: {files}")
    
    # Initialize document loader with larger chunk size
    document_loader = DocumentLoader(
        chunk_size=1500,  # Even larger chunk size
        chunk_overlap=300  # More overlap
    )
    
    # Load and split documents
    documents = document_loader.load_and_split_documents(data_dir)
    print(f"Loaded {len(documents)} document chunks")
    
    # Print first few characters of each chunk from company_faq.txt
    print("\nDocument chunks from company_faq.txt:")
    for i, doc in enumerate(documents):
        if doc["metadata"]["source"] == "company_faq.txt":
            print(f"\nChunk {i}:")
            print(f"Source: {doc['metadata']['source']}")
            print(f"Content (first 100 chars): {doc['content'][:100]}...")
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Create index
    vector_store.create_index(documents)
    
    # Test queries
    test_queries = [
        "What is RAGent AI?",
        "Tell me about RAGent AI",
        "What products does RAGent AI offer?"
    ]
    
    for query in test_queries:
        print(f"\n\nTesting query: {query}")
        results = vector_store.retrieve(query, top_k=3)
        
        print(f"Retrieved {len(results)} chunks:")
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (score: {result['score']:.4f}):")
            print(f"Source: {result['metadata']['source']}")
            print(f"Content: {result['content'][:150]}...")

if __name__ == "__main__":
    main()
