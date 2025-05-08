"""
Document loader utility for processing text files into chunks for vector indexing.
"""
import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document loader with chunking parameters.
        
        Args:
            chunk_size: The size of each text chunk (default: 1000 characters)
            chunk_overlap: The overlap between consecutive chunks (default: 200 characters)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def load_and_split_documents(self, data_dir: str) -> List[Dict[str, Any]]:
        """
        Load documents from a directory and split them into chunks.
        
        Args:
            data_dir: Directory containing text files
            
        Returns:
            List of document chunks with metadata
        """
        documents = []
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split the document into chunks
                chunks = self.text_splitter.create_documents(
                    texts=[content],
                    metadatas=[{"source": filename}]
                )
                
                # Convert LangChain Document objects to dictionaries
                for chunk in chunks:
                    documents.append({
                        "content": chunk.page_content,
                        "metadata": {
                            "source": chunk.metadata["source"],
                            "chunk_id": len(documents)
                        }
                    })
        
        return documents
