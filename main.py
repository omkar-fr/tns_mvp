import os
import logging
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

# Standard logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# --- Tools & Logic ---

def get_vector_store():
    """Returns a fresh connection to the Qdrant vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    return QdrantVectorStore(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME", "codebase"), 
        embedding=embeddings
    )

@tool
def search_codebase(query: str) -> str:
    """
    Search the codebase for relevant snippets or logic. 
    Use this to understand how the project is structured or how functions work.
    """
    logger.info(f"🔎 Agent is searching for: {query}")
    try:
        vs = get_vector_store()
        docs = vs.similarity_search(query, k=5)
        
        if not docs:
            return "No matching code snippets found."

        return "\n\n".join(
            [f"--- File: {d.metadata.get('source')} ---\n{d.page_content}" for d in docs]
        )
    except Exception as e:
        return f"Error searching vector store: {str(e)}"

# Setup Tooling
tools = [search_codebase]
tool_node = ToolNode(tools)

# --- Graph Definition ---

class State(TypedDict):
    """Simple state tracking for the conversation."""
    messages: Annotated[List[BaseMessage], add_messages]

# Using Gemini 1.5 Flash for speed/cost (or upgrade to 2.0/3.0 if available)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0 
).bind_tools(tools)

def call_model(state: State):
    """The brain: decides what to say or which tool to call."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def router(state: State):
    """Logic to decide if we should stop or run a tool."""
    last_msg = state["messages"][-1]
    return "tools" if last_msg.tool_calls else END

def create_graph():
    """Compiles the LangGraph workflow."""
    workflow = StateGraph(State)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", router)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# --- THE KEY FIX ---
# We instantiate 'app' here so Streamlit can import it successfully.
app = create_graph()

# --- CLI Runner ---
def run_terminal_chat():
    """Allows for testing the agent directly in the terminal."""
    print("\n👋 Code Agent online. Type 'exit' to quit.")
    while True:
        user_input = input("\nUser ❯ ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        inputs = {"messages": [HumanMessage(content=user_input)]}
        # Simplified terminal output
        for chunk in app.stream(inputs, stream_mode="values"):
            msg = chunk["messages"][-1]
            if isinstance(msg, AIMessage) and msg.content:
                print(f"Assistant ❯ {msg.content}")

if __name__ == "__main__":
    run_terminal_chat()