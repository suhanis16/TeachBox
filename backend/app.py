import streamlit as st
from streamlit_card import card
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from pathlib import Path
from langchain.embeddings import OllamaEmbeddings
from serpapi import GoogleSearch

# Load environment variables
dotenv_path = Path('../secrets.env')
load_dotenv(dotenv_path=dotenv_path)

# Load the Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')

st.set_page_config(layout="wide")
st.title("Teachbox")

# Initialize session state variables
if "show_cards" not in st.session_state:
    st.session_state.show_cards = True

if "vector" not in st.session_state:
    st.session_state.vectors = FAISS.load_local("faiss_index", embeddings=OllamaEmbeddings(model="llama3.2"), allow_dangerous_deserialization=True)

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
prompt_template = ChatPromptTemplate.from_template(
    """
    You are a specialized educational chatbot designed to assist STEM educators in selecting effective active learning pedagogies tailored to their unique teaching conditions (e.g., subject matter, level of students, class size, time of day).
    Respond to each question using only the information provided in the context.
    Ensure responses are highly accurate and relevant to the educator's conditions.
    Offer specific examples of recommended pedagogies and methods, customized to the given context. Specify the recommended duration and suggested order for implementing each pedagogy or method.
    
    <context>
    {context}
    <context>
    
    Question: {input}
    """
)
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Function to query Google Scholar using SerpAPI
def search_google_scholar(query):
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY") # Replace with your SerpAPI key
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get('organic_results', [])  # List of results

# Google Scholar retrieval agent
def scholar_agent(topic):
    results = search_google_scholar(topic)
    papers = []

    for result in results:
        title = result.get('title', 'No Title')
        link = result.get('link', 'No Link')
        snippet = result.get('snippet', 'No Snippet')
        papers.append({"title": title, "link": link, "snippet": snippet})

    return papers

# Function to run the chatbot and search Google Scholar
def run_chatbot(query):
    if query:
        st.session_state.show_cards = False
        response = retrieval_chain.invoke({"input": query})
        st.write(response['answer'])

        # Scholar search for additional resources
        scholar_results = scholar_agent(query)
        with st.expander("Additional Research Papers from Google Scholar"):
            for paper in scholar_results:
                st.write(f"**Title:** {paper['title']}")
                st.write(f"**Snippet:** {paper['snippet']}")
                st.write(f"[Read more]({paper['link']})")
                st.write("--------------------------------")

        # Display unique sources in an expander
        with st.expander("Sources"):
            seen_sources = set()
            for doc in response["context"]:
                source = doc.metadata['source']
                if source not in seen_sources:
                    if "title" in doc.metadata:
                        st.write("**Title:** " + doc.metadata['title'])
                    if "description" in doc.metadata:
                        st.write("**Description:** " + doc.metadata['description'])
                    st.write("**Link:** " + source)
                    st.write("--------------------------------")
                    seen_sources.add(source)

        # Display document similarity search in an expander
        with st.expander("Document Similarity Search"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")

# Display example cards
if st.session_state.show_cards:
    example_prompts = [
        "What are some active learning strategies to teach cellular biology to high school students?",
        "How can I make an introductory calculus class more interactive for first-year college students?",
        "How can I engage students in remote learning?"
    ]

    cols = st.columns(3)
    for i, example in enumerate(example_prompts):
        with cols[i]:
            card(
                title="",
                text=example,
                styles={
                    "card": {"width": "100%"},
                },
                on_click=lambda ex=example: run_chatbot(ex)
            )

# Input prompt at the bottom of the screen
prompt = st.text_input("Input your prompt here")

# Process input if a prompt is entered
if prompt:
    run_chatbot(prompt)