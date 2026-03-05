import os
import shutil
import logging
import streamlit as st
import tkinter as tk
from tkinter import filedialog

from main import app as graph_app
from src.ingestion.load_code import ingest_repo
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Code Navigator AI", 
    page_icon="🤖", 
    layout="wide"
)

logger = logging.getLogger(__name__)

# Custom CSS for a cleaner "Chat" look
st.markdown("""
    <style>
    .stStatusWidget { border-radius: 10px; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Logic: Folder Selection ---
def select_folder():
    """Opens a native file dialog. Note: Works on local machines only."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder = filedialog.askdirectory()
    root.destroy()
    return folder

# --- Sidebar: Project Controls ---
with st.sidebar:
    st.title("📁 Workspace")
    st.info("Load a local Python project to begin chatting with the codebase.")
    
    if st.button("📂 Open Project Folder", use_container_width=True):
        selected_path = select_folder()

        if selected_path:
            # Prepare internal data storage
            DATA_DIR = os.path.abspath("./data")
            if os.path.exists(DATA_DIR):
                shutil.rmtree(DATA_DIR)
            os.makedirs(DATA_DIR)
            
            project_name = os.path.basename(selected_path)
            target_path = os.path.join(DATA_DIR, project_name)
            
            try:
                with st.status(f"Indexing {project_name}...", expanded=True) as status:
                    st.write("🚚 Copying files...")
                    shutil.copytree(selected_path, target_path)
                    
                    st.write("🧠 Generating embeddings & updating Qdrant...")
                    ingest_repo(target_path)
                    
                    status.update(label="✅ Project Ready!", state="complete")
                
                st.session_state.messages = [] # Reset chat for the new repo
                st.toast(f"Successfully indexed {project_name}")
            except Exception as e:
                st.error(f"Failed to load project: {e}")

# --- Main UI: Chat Interface ---
st.title("🤖 Code Navigator")
st.caption("Ask me about architecture, logic, or specific functions in your project.")

# Initialize history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Chat Input
if prompt := st.chat_input("Ex: 'How does the authentication flow work?'"):
    # 1. Add and display user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate and display assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Use Streamlit status to show the 'thinking' process of the graph
        with st.status("Thinking...", expanded=False) as status:
            # We pass the full history to give the AI memory
            inputs = {"messages": st.session_state.messages}
            
            for chunk in graph_app.stream(inputs, stream_mode="values"):
                # Get the latest message from the graph state
                last_msg = chunk["messages"][-1]
                
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    full_response = last_msg.content
                    response_placeholder.markdown(full_response)
            
            status.update(label="Check complete", state="complete")
        
        # Store the AI's final answer in history
        if full_response:
            st.session_state.messages.append(AIMessage(content=full_response))