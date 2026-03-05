from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # 'add_messages' tells LangGraph to append new messages to the history 
    # rather than overwriting them
    messages: Annotated[Sequence[BaseMessage], add_messages]