import streamlit as st
from streamlit_card import card
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('../secrets.env')
load_dotenv(dotenv_path=dotenv_path)

# Load the Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')

st.set_page_config(
    layout="wide",
)
st.title("Teachbox")

# Initialize session state variable to show/hide cards
if "show_cards" not in st.session_state:
    st.session_state.show_cards = True

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="llama3.2")
    st.session_state.loader = WebBaseLoader(
        web_paths=["https://teaching.tools/activities"])
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents, st.session_state.embeddings)

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
)
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Function to run the chatbot with a selected prompt
def run_chatbot(query):
    if query:
        st.session_state.show_cards = False
        response = retrieval_chain.invoke({"input": query})
        st.write(response['answer'])

        # Display unique sources in an expander
        with st.expander("Sources"):
            seen_sources = set()
            for doc in response["context"]:
                source = doc.metadata['source']
                if source not in seen_sources:  # Check for duplicates
                    st.write("Title: " + doc.metadata['title'])
                    st.write("Description: " + doc.metadata['description'])
                    st.write("Link: " + source)
                    st.write("--------------------------------")
                    seen_sources.add(source)  # Add source to the set

        # Display document similarity search in an expander
        with st.expander("Document Similarity Search"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")

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
                title=f"",
                text=example,
                styles={
                    "card": {
                        "width": "100%",
                    },
                },
                on_click=lambda ex=example: run_chatbot(ex)
            )

# Input prompt at the bottom of the screen
prompt = st.text_input("Input your prompt here")

# Process input if a prompt is entered
if prompt:
    run_chatbot(prompt)