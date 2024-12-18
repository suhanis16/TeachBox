import streamlit as st
from streamlit_card import card
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from pathlib import Path
from serpapi import GoogleSearch
from langchain.docstore.document import Document
from langchain.schema import HumanMessage, AIMessage
from streamlit_chat import message

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

if "memory" not in st.session_state or not isinstance(st.session_state.memory, ConversationBufferMemory):
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "vectors" not in st.session_state:
    st.session_state.vectors = FAISS.load_local(
        "faiss_index",
        embeddings=OllamaEmbeddings(model="llama3.2"),
        allow_dangerous_deserialization=True
    )

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768", temperature=0)

# Prompt template for responses
prompt_template = ChatPromptTemplate.from_template(
    """
    You are a specialized educational chatbot designed to assist STEM educators in selecting effective active learning pedagogies tailored to their unique teaching conditions (e.g., subject matter, level of students, class size, time of day).
    Respond to each question using only the information provided in the context.
    Ensure responses are highly accurate and relevant to the educator's conditions.
    Offer specific examples of recommended pedagogies and methods, customized to the given context. 
    For each recommended active learning method, also provide a short definition explaining what it is.
    Specify the recommended duration and suggested order for implementing each pedagogy or method.

    <context>
    {context}
    </context>

    Question: {input}
    """
)
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Function to generate a concise query for Scholar search
def generate_concise_query(input_query):
    concise_query_prompt = f"""
    Given the following detailed query, extract a concise query which I can search on Google Scholar to fetch relevant results:
    ---
    {input_query}
    ---
    Your response should be of the following format: "teaching [subject] [topic] using active learning". Focus on extracting only the subject and/or topic (whichever is applicable). Ignore all other details like class size, level of students, class time etc.
    """
    concise_response = llm.predict(concise_query_prompt)
    return concise_response.strip().strip('"')

# Function to query Google Scholar using SerpAPI
def search_google_scholar(query):
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY")  # Replace with your SerpAPI key
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

# Function to summarize paper snippets
def summarize_paper_snippet(llm, snippet):
    summary_prompt = f"""
    Please provide a concise summary of the following research paper content. Focus on the main idea, methodology, or findings relevant to active learning strategies:

    {snippet}

    Summary:
    """
    return llm.predict(summary_prompt).strip()

# Main function to handle the chatbot logic
def run_chatbot(query):
    if query:
        st.session_state.show_cards = False

        # Retrieve the conversational context
        memory_context = st.session_state.memory.load_memory_variables({})
        previous_context = memory_context.get("history", "")

        # Generate concise query for Google Scholar search
        concise_query = generate_concise_query(query)
        scholar_results = scholar_agent(concise_query)

        # Summarize each paper's snippet using the LLM
        summarized_papers = []
        for paper in scholar_results:
            summary = summarize_paper_snippet(llm, paper['snippet'])
            summarized_papers.append({
                "title": paper['title'],
                "link": paper['link'],
                "summary": summary
            })

        # Convert summarized Scholar results into documents
        scholar_docs = [
            Document(
                page_content=paper['summary'], 
                metadata={"title": paper['title'], "link": paper['link']}
            ) 
            for paper in summarized_papers
        ]

        # Combine original docs and scholar docs
        combined_vectorstore = FAISS.from_documents(scholar_docs, OllamaEmbeddings(model="llama3.2"))
        combined_retriever = combined_vectorstore.as_retriever()
        combined_retrieval_chain = create_retrieval_chain(combined_retriever, document_chain)

        # Run the combined retrieval chain to get the final enriched answer
        final_response = combined_retrieval_chain.invoke({"input": query})

        # Display the response
        # st.write(final_response['answer'])
  # Update chat history
        st.session_state.chat_history = [{"role": "user", "content": query}, {"role": "assistant", "content": final_response['answer']}]

        # Display chat history with unique keys
        for idx, message_data in enumerate(st.session_state.chat_history):
            if message_data["role"] == "user":
                message(message_data["content"], is_user=True, key=f"user_{idx}")
            else:
                message(message_data["content"], key=f"assistant_{idx}")

        # Display sources from scholar papers
        with st.expander("Sources from Research Papers"):
            for doc in scholar_docs:
                st.write("**Title:** " + doc.metadata['title'])
                st.write("**Link:** " + doc.metadata['link'])
                st.write("**Used Summary Content:**")
                st.write(doc.page_content)
                st.write("--------------------------------")
                # Document similarity search (from combined)

        with st.expander("Document Similarity Search"):
            for doc in final_response["context"]:
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
