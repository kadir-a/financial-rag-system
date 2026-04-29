import streamlit as st
import os
import sys
import tempfile
from dotenv import load_dotenv

# --- CRITICAL PATCH 1: PATH RESOLUTION ---
# Append the root directory to sys.path so Streamlit doesn't get trapped in the 'ui' folder.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- CRITICAL PATCH 2: ENVIRONMENT VARIABLES ---
# Unlock the hidden vault: Load the API keys from the .env file.
load_dotenv()

# Import the backend engines (Now able to locate the 'app' module)
from app.services.ingestion import FinancialPDFProcessor
from app.services.vector_store import VectorStoreManager
from app.services.llm_service import LLMService

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Financial RAG", page_icon="📈", layout="wide")

# 2. SESSION STATE (Memory & Vault)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "What insights are you looking for?"}]
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# 3. SIDEBAR (Control Center & Data Ingestion)
with st.sidebar:
    st.title("⚙️ Control Center")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Financial Report (PDF)", type=["pdf"])
    
    if st.button("Analyze Report", type="primary", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Reading report and converting to vectors... Please wait."):
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                processor = FinancialPDFProcessor()
                chunks = processor.process_pdf(tmp_path)
                
                vector_manager = VectorStoreManager()
                # Updated to match the new OpenAI dimensions vault
                vector_db = vector_manager.create_and_store_embeddings(chunks, "report_openai_1536")
                retriever = vector_db.as_retriever(search_kwargs={"k": 3})
                
                llm_service = LLMService()
                st.session_state.rag_chain = llm_service.create_rag_chain(retriever)
                
                # --- SECURITY PATCH: EVICT PROCESSED DATA FROM THE VAULT ---
                os.remove(tmp_path) 
                
                st.success("Engine active! Data processed. You can now start the analysis.")
        else:
            st.error("Empty orders are not accepted. Please upload a PDF first!")

# 4. TRADING BOARD (Main Chat Interface)
st.title("📈 Financial Intelligence RAG Assistant")
st.markdown("Upload reports, build the knowledge base, and start querying.")

for msg in st.session_state.messages:
    msg_avatar = "👤" if msg["role"] == "user" else "🧠"
    st.chat_message(msg["role"], avatar=msg_avatar).write(msg["content"])

# 5. ORDER ENTRY & GPT RESPONSE (User Input)
if prompt := st.chat_input("What is the net loss for the period?"):
    st.chat_message("user", avatar="👤").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if st.session_state.rag_chain is None:
        warning_msg = "Engine is not connected yet. Please upload a report from the left sidebar and click 'Analyze Report'."
        st.chat_message("assistant",avatar="🧠").write(warning_msg)
        st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    else:
        with st.spinner("Scanning board data and executing analysis..."):
            
            history_tuples = [(msg["role"], msg["content"]) for msg in st.session_state.messages[:-1]]
            
            # The engine now returns a direct string, no need to search for an 'answer' key!
            final_answer = st.session_state.rag_chain.invoke({
                "input": prompt,
                "chat_history": history_tuples
            })
            
            st.chat_message("assistant", avatar="🧠").write(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})