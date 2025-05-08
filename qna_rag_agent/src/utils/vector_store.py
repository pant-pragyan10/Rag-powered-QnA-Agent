"""
Vector store utility for creating and querying embeddings.
"""
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self, api_key: str = None):
        """
        Initialize the vector store with a TF-IDF vectorizer.
        
        Args:
            api_key: API key (not used for TF-IDF)
        """
        self.vectorizer = TfidfVectorizer()
        self.document_embeddings = None
        self.documents = []
        
    def create_index(self, documents: List[Dict[str, Any]]):
        """
        Create TF-IDF embeddings for document chunks.
        
        Args:
            documents: List of document chunks with content and metadata
        """
        self.documents = documents
        
        # Get embeddings for all documents
        texts = [doc["content"] for doc in documents]
        self.document_embeddings = self.vectorizer.fit_transform(texts)
        
        print(f"Created TF-IDF embeddings for {len(documents)} document chunks")
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant document chunks for a query.
        
        Args:
            query: The user's question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of the most relevant document chunks
        """
        if self.document_embeddings is None:
            raise ValueError("Embeddings have not been created yet")
        
        # Preprocess the query to enhance retrieval quality
        import re
        
        # Extract product names if present in the query
        product_pattern = r'\b(RAGent\s+(?:Search|Assistant|Analytics|Connect))\b'
        product_matches = re.findall(product_pattern, query, re.IGNORECASE)
        
        # Check for company name mentions
        company_pattern = r'\b(RAGent\s*AI|RAGent)\b'
        company_matches = re.findall(company_pattern, query, re.IGNORECASE)
        
        # Create a boosted query
        boosted_query = query
        
        # If specific products are mentioned, boost their importance
        if product_matches:
            for product in product_matches:
                # Add the product name multiple times to boost its weight
                boosted_query += f" {product} {product}"
        
        # If company name is mentioned, boost its importance
        if company_matches:
            for company in company_matches:
                # Add the company name multiple times to boost its weight
                boosted_query += f" {company} {company}"
                
        query = boosted_query
        
        # Get embedding for the query
        query_embedding = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between query and all documents
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        # Apply additional heuristics for product-specific and company-related queries
        # Boost for product mentions
        for product in product_matches:
            for i, doc in enumerate(self.documents):
                if product.lower() in doc["content"].lower():
                    # Increase similarity score for documents containing the product name
                    similarities[i] += 0.2  # Boost by a fixed amount
        
        # Boost for company mentions
        for company in company_matches:
            for i, doc in enumerate(self.documents):
                if company.lower() in doc["content"].lower():
                    # Increase similarity score for documents containing the company name
                    similarities[i] += 0.3  # Higher boost for company information
        
        # Get indices of top_k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Get the corresponding documents
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": float(similarities[idx])
            })
        
        return results
