# Import relevant functionality
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from pathlib import Path
import os

# Import Groq integration
from langchain_groq import ChatGroq  # Replace this with the actual import path for Groq if different

# Load environment variables
dotenv_path = Path('../secrets.env')
load_dotenv(dotenv_path=dotenv_path)

# Fetch API keys from the environment
groq_api_key = os.getenv('GROQ_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')

# Ensure API keys are set
if not groq_api_key or not tavily_api_key:
    raise ValueError("API keys for Groq or Tavily are missing. Please check your secrets.env file.")

# Create the agent components
memory = MemorySaver()
model = ChatGroq(api_key=groq_api_key, model_name="mixtral-8x7b-32768")  # Adjust `model_name` as per Groq API docs
search = TavilySearchResults(max_results=2, api_key=tavily_api_key, include_images = True, include_image_descriptions = True)  # Ensure API key is passed
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent with streaming output
config = {"configurable": {"thread_id": "abc123"}}

# Example 1: Greet the agent
print("Conversation 1: Greeting the agent")
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="give response in hindi language. Tell about nature in hindi")]}, config
):
    print(chunk)
    print("----")

# Example 2: Ask about the weather
print("Conversation 2: Asking about the weather")
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="Kya tum aaj bhopal ka mausam bata sakte ho?")]}, config
):
    print(chunk)
    print("----")
