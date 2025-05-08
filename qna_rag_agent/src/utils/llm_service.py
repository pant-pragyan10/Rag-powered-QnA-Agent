"""
LLM service for generating answers based on retrieved context.
"""
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

class LLMService:
    def __init__(self, groq_api_key: str, model_name: str = "llama3-8b-8192"):
        """
        Initialize the LLM service.
        
        Args:
            groq_api_key: Groq API key
            model_name: Name of the LLM model to use
        """
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.2
        )
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate an answer to the user's query based on retrieved context.
        
        Args:
            query: The user's question
            context_chunks: Retrieved document chunks
            
        Returns:
            Generated answer
        """
        # Prepare context from retrieved chunks
        context_sections = []
        calculation_result = None
        
        for i, chunk in enumerate(context_chunks):
            # Check if this is a calculation result
            if chunk.get("metadata", {}).get("source") == "calculator_tool":
                calculation_result = chunk["content"]
            else:
                # Format regular context chunks with their source
                source = chunk["metadata"]["source"]
                content = chunk["content"]
                score = chunk.get("score", 0.0)
                context_sections.append(f"[Document {i+1}] From {source} (Relevance: {score:.2f}):\n{content}")
        
        # Join all context sections
        context_text = "\n\n---\n\n".join(context_sections)
        
        # Add special instructions for company information queries
        if "ragent ai" in query.lower():
            context_text = "IMPORTANT: This query is about RAGent AI. If information about RAGent AI is in the context, be sure to include it in your answer.\n\n" + context_text
        
        # Create system and user messages
        system_message = SystemMessage(content=(
            "You are a helpful assistant that answers questions based on the provided context. "
            "If the answer cannot be found in the context, provide a direct and concise response stating that the information is not in the context. "
            "For example, if asked about multilingual support and there's no mention of it, say 'There is no mention of multilingual support in the context. The system appears to be designed primarily for English queries based on the available information.' "
            "Be direct and concise - don't apologize or be overly verbose when information is missing. "
            "Do not make up information that is not in the context. "
            "If the query contains multiple questions or parts, make sure to address all of them in your answer. "
            "When answering questions about specific products or features, be sure to include all relevant details from the context. "
            "For product-specific questions, focus on the product's features, benefits, use cases, and unique selling points. "
            "When answering questions about RAGent AI (the company), be thorough and include key information about its history, products, and other relevant details from the context. "
            "Pay special attention to company-related queries like 'What is RAGent AI?' and ensure you provide complete information from the context."
        ))
        
        # Prepare the user message content
        user_content = f"Context:\n{context_text}\n\n"
        
        # Add calculation result if present
        if calculation_result:
            user_content += f"Calculation Result: {calculation_result}\n\n"
        
        user_content += f"Question: {query}\n\n"
        user_content += f"Answer the question based on the provided context and calculation result (if any). Make sure to address all parts of the question."
        
        user_message = HumanMessage(content=user_content)
        
        # Generate response
        response = self.llm.invoke([system_message, user_message])
        
        return response.content
