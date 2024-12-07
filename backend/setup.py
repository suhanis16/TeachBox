import pinecone
from langchain.vectorstores import Pinecone
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
dotenv_path = Path('../secrets.env')
load_dotenv(dotenv_path=dotenv_path)

# Fetch API keys from the environment
groq_api_key = os.getenv('GROQ_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

# Create or connect to an index
index_name = "active-learning"  # You can choose a suitable name
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=1536, metric="cosine")

index = pinecone.Index(index_name)
