"""
Streamlit web interface for the RAG-powered multi-agent Q&A system.
"""
import os
import streamlit as st
from dotenv import load_dotenv
import sys
import os.path

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.document_loader import DocumentLoader
from src.utils.vector_store import VectorStore
from src.utils.llm_service import LLMService
from src.agents.agent_orchestrator import AgentOrchestrator

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Error: GROQ_API_KEY not found in environment variables.")
    st.info("Please create a .env file with your Groq API key or set it as an environment variable.")
    st.stop()

# Set page config
st.set_page_config(
    page_title="RAG-Powered Multi-Agent Q&A",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.agent = None
    st.session_state.documents = None
    st.session_state.history = []

def initialize_system():
    """Initialize the RAG system and agent."""
    with st.spinner("Initializing system..."):
        # Get absolute path to data directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(project_dir, "data")
        
        # Initialize document loader
        document_loader = DocumentLoader(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Load and split documents
        documents = document_loader.load_and_split_documents(data_dir)
        st.session_state.documents = documents
        
        # Initialize vector store
        vector_store = VectorStore()
        
        # Create index
        vector_store.create_index(documents)
        
        # Initialize LLM service
        llm_service = LLMService(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
        
        # Initialize agent orchestrator
        agent = AgentOrchestrator(
            vector_store=vector_store,
            llm_service=llm_service
        )
        
        st.session_state.agent = agent
        st.session_state.initialized = True
        
        st.success(f"System initialized with {len(documents)} document chunks")

# Main app
st.title("🤖 RAG-Powered Multi-Agent Q&A")
st.markdown("""
This demo showcases a simple knowledge assistant that:
1. Retrieves relevant information from a document collection
2. Generates natural-language answers via an LLM
3. Orchestrates the retrieval + generation steps with a basic agentic workflow
""")

# Initialize the system if not already done
if not st.session_state.initialized:
    initialize_system()

# User input
query = st.text_input("Ask a question:", placeholder="e.g., What is RAGent AI? or Calculate 25 * 16")

if query:
    with st.spinner("Processing your question..."):
        # Process the query
        result = st.session_state.agent.process_query(query)
        
        # Add to history
        st.session_state.history.append(result)

# Display results
if st.session_state.history:
    result = st.session_state.history[-1]
    
    st.header("Results")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Query Processing")
        st.info(f"**Decision**: {result['decision']}")
        
        if result['tool_used']:
            st.subheader("Tool Details")
            st.write(f"**Tool**: {result['tool_used']}")
            st.write(f"**Tool Input**: {result['tool_input']}")
            st.write(f"**Tool Output**: {result['tool_output']}")
        else:
            st.subheader("Retrieved Context")
            for i, chunk in enumerate(result['retrieved_context']):
                with st.expander(f"Chunk {i+1} (from {chunk['metadata']['source']})"):
                    st.write(f"**Score**: {chunk['score']:.4f}")
                    st.text(chunk['content'])
    
    with col2:
        st.subheader("Answer")
        st.success(result['answer'])

# History
if len(st.session_state.history) > 1:
    with st.expander("Query History"):
        for i, item in enumerate(st.session_state.history[:-1]):
            st.write(f"**Q{i+1}**: {item['query']}")
            st.write(f"**A{i+1}**: {item['answer']}")
            st.write("---")

# Footer
st.markdown("---")
st.markdown("RAG-Powered Multi-Agent Q&A System | Built with LangChain, Groq Llama3, and scikit-learn")
