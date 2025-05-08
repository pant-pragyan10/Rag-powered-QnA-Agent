"""
Agent orchestrator for routing queries to the appropriate tools or RAG pipeline.
"""
import re
from typing import Dict, Any, List, Optional
from ..utils.vector_store import VectorStore
from ..utils.llm_service import LLMService
from .tools import CalculatorTool, DictionaryTool

class AgentOrchestrator:
    def __init__(self, vector_store: VectorStore, llm_service: LLMService):
        """
        Initialize the agent orchestrator.
        
        Args:
            vector_store: Vector store for retrieving relevant documents
            llm_service: LLM service for generating answers
        """
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.tools = {
            "calculator": CalculatorTool(),
            "dictionary": DictionaryTool()
        }
        
        # Patterns for routing to specific tools
        self.patterns = {
            "calculator": r"\b(calculate|compute|evaluate|solve|find the value of|math|age|born in|how old|years old|square root|sqrt|\d+\s*\+|\d+\s*\-|\d+\s*\*|\d+\s*\/|\d+\s*\^)\b|\b(what is\s+[\d\+\-\*\/\(\)]+)\b",
            "dictionary": r"\b(define|definition of|meaning of|what does .* mean)\b(?!.*\bin AI\b|.*\bin artificial intelligence\b)"
        }
    
    def _should_use_tool(self, query: str) -> Optional[str]:
        """
        Determine if a query should be routed to a specific tool.
        
        Args:
            query: The user's question
            
        Returns:
            Name of the tool to use, or None for RAG
        """
        query_lower = query.lower()
        
        # Check for AI-related queries that should always use RAG
        ai_related_patterns = [
            r'\b(ai|artificial intelligence|machine learning|neural network|llm|large language model)\b',
            r'\bhallucination\b',
            r'\brag\b',
            r'\bretrieval augmented generation\b',
            r'\bwhat\s+is\s+the\s+meaning\s+of\s+\w+\s+in\s+ai\b',
            r'\bwhat\s+does\s+\w+\s+mean\s+in\s+ai\b',
            r'\bdefine\s+["\']?\w+\s+in\s+ai["\']?\b'
        ]
        
        for pattern in ai_related_patterns:
            if re.search(pattern, query_lower):
                return None  # Use RAG for AI-related queries
        
        # Special case for system-specific terminology and concepts
        system_terms = [
            r'\b(ragent|rag agent|company|product|headquarter|employee|technology)\b',
            r'\b(agentic|routing|workflow|pipeline|architecture|vector store|llm service)\b',
            r'\b(this system|the system|our system)\b',
            r'\bhow\s+\w+\s+works\s+in\s+this\s+system\b',
            r'\bdefine\s+["\']?\w+\s+routing["\']?\b',  # Handle 'define X routing' patterns
        ]
        
        # Check for AI terminology definitions that should be handled by RAG
        ai_terms = [
            'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
            'transformer', 'llm', 'large language model', 'retrieval augmented generation', 'rag',
            'vector database', 'embedding', 'fine-tuning', 'prompt engineering', 'attention mechanism',
            'tokenization', 'nlp', 'natural language processing', 'generative ai', 'hallucination',
            'ai alignment', 'bias', 'explainable ai', 'ai safety', 'responsible ai', 'ai governance',
            'multimodal ai', 'diffusion model', 'foundation model', 'agent', 'agentic workflow',
            'agentic routing', 'tool use', 'in-context learning', 'parameter-efficient fine-tuning',
            'quantization', 'perplexity', 'context window', 'token'
        ]
        
        # If the query is asking for a definition of an AI term, use RAG
        for term in ai_terms:
            pattern = r'\bdefine\s+["\']?' + re.escape(term) + r'["\']?\b'
            if re.search(pattern, query_lower):
                return None  # Use RAG for AI term definitions
        
        for pattern in system_terms:
            if re.search(pattern, query_lower):
                return None  # Use RAG for system-related queries
        
        # Check for tool patterns
        for tool_name, pattern in self.patterns.items():
            if re.search(pattern, query_lower):
                return tool_name
        
        return None
    
    def _extract_tool_input(self, query: str, tool_name: str) -> str:
        """
        Extract the relevant input for a tool from the query.
        
        Args:
            query: The user's question
            tool_name: Name of the tool to use
            
        Returns:
            Extracted input for the tool
        """
        query_lower = query.lower()
        
        if tool_name == "calculator":
            # Try to extract a mathematical expression
            # Look for patterns like "calculate 2+2" or "what is 5*3"
            patterns = [
                r"calculate\s+(.*)",
                r"compute\s+(.*)",
                r"evaluate\s+(.*)",
                r"solve\s+(.*)",
                r"what is\s+(.*)",
                r"find the value of\s+(.*)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    return match.group(1).strip()
            
            # If no pattern matches, just use the query as is
            return query
        
        elif tool_name == "dictionary":
            # Try to extract a word to define
            # Look for patterns like "define apple" or "what does apple mean"
            patterns = [
                r"define\s+(.*)",
                r"definition of\s+(.*)",
                r"meaning of\s+(.*)",
                r"what does\s+(.*?)\s+mean"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    return match.group(1).strip()
            
            # If no pattern matches, just use the query as is
            return query
        
        return query
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the agent workflow.
        
        Args:
            query: The user's question
            
        Returns:
            Dictionary containing the processing results
        """
        # Log the query
        print(f"Processing query: {query}")
        
        # Check for mixed queries (containing both tool-related and general knowledge questions)
        import re
        
        # Look for mathematical patterns, especially square root
        math_pattern = r'(square root of \d+(\.\d+)?|sqrt\s*\(?\s*\d+(\.\d+)?\s*\)?|\d+\s*[\+\-\*\/\^]\s*\d+)'
        math_match = re.search(math_pattern, query.lower())
        
        # If we have a mixed query with a math component
        if math_match and len(query.split()) > 6:  # More than 6 words suggests a mixed query
            # Extract the math part
            math_part = math_match.group(0)
            
            # Process the math part with calculator tool
            calculator_tool = self.tools["calculator"]
            calc_result = calculator_tool.run(math_part)
            
            # Process the rest with RAG
            # Retrieve relevant documents
            retrieved_chunks = self.vector_store.retrieve(query)
            
            # Generate answer using LLM, including the calculation result
            context_with_calc = retrieved_chunks + [{
                "content": f"Calculation result: {math_part} = {calc_result['output']}",
                "metadata": {"source": "calculator_tool", "chunk_id": 999}
            }]
            
            answer = self.llm_service.generate_answer(query, context_with_calc)
            
            # Log the decision
            print(f"Using mixed approach: calculator + RAG")
            
            return {
                "query": query,
                "decision": "Used mixed approach: calculator + RAG",
                "tool_used": "mixed",
                "tool_input": math_part,
                "tool_output": calc_result["output"],
                "retrieved_context": retrieved_chunks,
                "answer": answer
            }
        
        # Standard single-intent processing
        tool_name = self._should_use_tool(query)
        
        if tool_name and tool_name in self.tools:
            # Extract tool input
            tool_input = self._extract_tool_input(query, tool_name)
            
            # Use the appropriate tool
            tool = self.tools[tool_name]
            tool_result = tool.run(tool_input)
            
            # Log the decision
            print(f"Using tool: {tool_name}")
            
            return {
                "query": query,
                "decision": f"Used {tool_name} tool",
                "tool_used": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_result["output"],
                "retrieved_context": None,
                "answer": tool_result["output"]
            }
        else:
            # Use RAG pipeline
            # Retrieve relevant documents
            retrieved_chunks = self.vector_store.retrieve(query)
            
            # Generate answer using LLM
            answer = self.llm_service.generate_answer(query, retrieved_chunks)
            
            # Log the decision
            print(f"Using RAG pipeline")
            
            return {
                "query": query,
                "decision": "Used RAG pipeline",
                "tool_used": None,
                "tool_input": None,
                "tool_output": None,
                "retrieved_context": retrieved_chunks,
                "answer": answer
            }
