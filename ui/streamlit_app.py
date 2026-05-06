import streamlit as st
import requests

# Backend API Endpoint (Resolved via Docker Compose internal network)
API_URL = "http://backend:8000"

# UI Configuration
st.set_page_config(page_title="FinSight AI Terminal", page_icon="⚡", layout="wide")
st.title("⚡ FinSight AI | Enterprise Financial Intelligence Terminal")

# Sidebar: Control Center
with st.sidebar:
    st.header("⚙️ Control Center")
    uploaded_file = st.file_uploader("Upload Financial Report (PDF)", type="pdf")
    
    if st.button("Ingest Document"):
        if uploaded_file is not None:
            with st.spinner("Processing... Converting report into matrix data..."):
                # Transmit PDF directly to the Backend API
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{API_URL}/upload/", files=files)
                    if response.status_code == 200:
                        st.success("✅ Document successfully ingested into the vector vault.")
                        # Purge session state (chat history) upon new document ingestion
                        st.session_state.messages = []
                    else:
                        st.error(f"Ingestion Failed: {response.json().get('detail')}")
                except Exception as e:
                    st.error("Connection Error: Unable to reach the backend service. Check Docker status.")
        else:
            st.warning("Please upload a PDF document before proceeding.")

# Session State Initialization for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query Execution
if prompt := st.chat_input("What insights do you need from the financial report?"):
    # 1. Display User Query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Transmit Query to Backend API
    with st.chat_message("assistant"):
        with st.spinner("Analyst is retrieving data..."):
            try:
                # Format chat history into a tuple list for the API payload
                chat_history = [(msg["role"], msg["content"]) for msg in st.session_state.messages[:-1]]
                
                payload = {
                    "question": prompt,
                    "chat_history": chat_history
                }
                
                response = requests.post(f"{API_URL}/ask/", json=payload)
                
                if response.status_code == 200:
                    answer = response.json().get("answer")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = response.json().get('detail', 'An unknown error occurred.')
                    st.error(f"API Error: {error_msg}")
            except Exception as e:
                st.error("Service Unavailable: Cannot reach the backend API. Please contact system administration.")