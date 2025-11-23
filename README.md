# Super30_Multi_agent_LangGraph_groq
Multi-Agent Dashboard using LangGraph, Groq (Llama 3), and Streamlit.

System Architecture
Frontend (Dashboard): Streamlit (Handles user input & display).

Brain (Controller): Supervisor Agent (Routes tasks).

Workers: Researcher (Searches web) & Writer (Drafts content).

Step 1: Install Requirements
You need streamlit in addition to the previous libraries.

Bash

pip install streamlit langgraph langchain-groq langchain-community tavily-python
Step 2: The Complete Dashboard Code (app.py)
Save this code as app.py. It combines the agent logic with a chat interface.
