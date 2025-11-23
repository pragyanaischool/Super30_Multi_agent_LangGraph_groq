import streamlit as st
import os
from typing import Annotated, List, TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Agentic AI Dashboard", page_icon="🤖", layout="wide")

# Sidebar for API Keys (Secure entry)
with st.sidebar:
    st.header("🔑 API Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get from console.groq.com")
    tavily_api_key = st.text_input("Tavily API Key", type="password", help="Get from tavily.com")
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key

    st.divider()
    st.markdown("### Agents Active:")
    st.success("✅ Supervisor (Llama 3-70B)")
    st.info("✅ Researcher (Tavily Search)")
    st.info("✅ Writer (Llama 3-70B)")

st.title("🤖 Multi-Agent Research Dashboard")
st.markdown("Ask a question. The **Supervisor** will delegate to the **Researcher** and **Writer** to generate a report.")

# --- 2. DEFINE AGENTS (Only initializes if keys are present) ---

if "app" not in st.session_state and groq_api_key and tavily_api_key:
    
    # Initialize LLM
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)

    # Tools
    tavily_tool = TavilySearchResults(max_results=3)

    # Helper to create agents
    def create_agent(llm, tools, system_prompt):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        if tools:
            llm_with_tools = llm.bind_tools(tools)
            chain = prompt | llm_with_tools
        else:
            chain = prompt | llm
        return chain

    # Agents
    research_agent = create_agent(llm, [tavily_tool], "You are a web researcher. Search the internet for accurate facts.")
    writer_agent = create_agent(llm, [], "You are a technical writer. Write a clear, concise summary based on the research provided.")

    # Nodes
    def research_node(state):
        result = research_agent.invoke(state)
        return {"messages": [result]}

    def writer_node(state):
        result = writer_agent.invoke(state)
        return {"messages": [result]}

    def supervisor_node(state):
        messages = state['messages']
        system_prompt = (
            "You are a supervisor managing: [Researcher, Writer].\n"
            "1. If user asks a question requiring facts, choose 'Researcher'.\n"
            "2. If you have research data and need a summary, choose 'Writer'.\n"
            "3. If the answer is complete and written, choose 'FINISH'."
        )
        options = ["Researcher", "Writer", "FINISH"]
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", f"Who should act next? Select one: {options}")
        ])
        chain = prompt | llm
        response = chain.invoke(messages)
        decision = response.content.strip()
        
        if "Researcher" in decision: return {"next": "Researcher"}
        if "Writer" in decision: return {"next": "Writer"}
        return {"next": "FINISH"}

    # Graph Construction
    class AgentState(TypedDict):
        messages: Annotated[List[BaseMessage], "history"]
        next: str

    workflow = StateGraph(AgentState)
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Researcher", research_node)
    workflow.add_node("Writer", writer_node)

    workflow.set_entry_point("Supervisor")
    workflow.add_conditional_edges("Supervisor", lambda x: x["next"], 
                                   {"Researcher": "Researcher", "Writer": "Writer", "FINISH": END})
    workflow.add_edge("Researcher", "Supervisor")
    workflow.add_edge("Writer", "Supervisor")
    
    st.session_state.app = workflow.compile()

# --- 3. DASHBOARD UI LOGIC ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_query = st.chat_input("E.g., Research the latest features of Llama 3 and write a blog post.")

if user_query:
    if not groq_api_key or not tavily_api_key:
        st.error("⚠️ Please enter your API keys in the sidebar first!")
    else:
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Run the Agent Workflow
        with st.chat_message("assistant"):
            status_container = st.status("🤖 Agents are working...", expanded=True)
            
            inputs = {"messages": [HumanMessage(content=user_query)]}
            final_response = ""
            
            # Stream the graph steps
            app = st.session_state.app
            for output in app.stream(inputs):
                for key, value in output.items():
                    # Update status box with agent activity
                    if key == "Researcher":
                        status_container.write("🔍 **Researcher** is searching the web...")
                    elif key == "Writer":
                        status_container.write("📝 **Writer** is drafting the response...")
                        # Capture the final output from the writer
                        if 'messages' in value:
                            final_response = value['messages'][-1].content
                    elif key == "Supervisor":
                        status_container.write("👮 **Supervisor** is coordinating...")

            status_container.update(label="✅ Task Complete!", state="complete", expanded=False)
            
            # Display Final Result
            if final_response:
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
            else:
                st.warning("The agents finished but returned no content. Try refining your prompt.")
