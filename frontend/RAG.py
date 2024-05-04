import os
from langchain.chat_models import ChatOpenAI

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "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"
os.environ["OPENAI_API_KEY"] = "sk-proj-bJWhMFLr5gB8BP8ZpdbsT3BlbkFJz2v6DKCqiogYsdMEUDQR"


chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    # HumanMessage(content="Hi AI, how are you today?"),
    # AIMessage(content="I'm great thank you. How can I help you?"),
    # HumanMessage(content="Tell me about active learning")
]

res = chat(messages)

import pandas as pd

df = pd.read_excel(r"D:\TeachBox\Active Learning Repo.xlsx")  # Assuming your data is in an Excel file

import pandas as pd
from datasets import Dataset


# Step 2: Define a custom dataset class
class CustomDataset:
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "Major": self.data.iloc[idx]['Major'],
            "Pedagogy Technique": self.data.iloc[idx]['Pedagogy Technique'],
            "Definition": self.data.iloc[idx]['Definition'],
            "Objective": self.data.iloc[idx]['Objective of the teaching technique'],
            "Resources": self.data.iloc[idx]['Resources'],
            "Citation": self.data.iloc[idx]['Citation'],
            "Course Structure": self.data.iloc[idx]['Course Structure'],
            "Method": self.data.iloc[idx]['Method'],
            "Class size": self.data.iloc[idx]['Class size'],
            "Findings": self.data.iloc[idx]['Findings'],
            "S-C": self.data.iloc[idx]['S-C'],
            "S-S": self.data.iloc[idx]['S-S'],
            "S-T": self.data.iloc[idx]['S-T'],
            "I": self.data.iloc[idx]['I'],
            "C": self.data.iloc[idx]['C'],
            "A": self.data.iloc[idx]['A'],
            "P": self.data.iloc[idx]['P'], 
            "ICAP": self.data.iloc[idx]['ICAP'],
            "A,S": self.data.iloc[idx]['A,S'],
            "Steps to try in Sandbox": self.data.iloc[idx]['Steps to try in Sandbox']
        }

# Step 3: Create an instance of your custom dataset
dataset = CustomDataset(df)


from pinecone import Pinecone

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY") or "917f1f5c-3ee8-4a46-8a3c-645b686a334a"

# configure client
pc = Pinecone(api_key=api_key)

from pinecone import ServerlessSpec

spec = ServerlessSpec(
    cloud="aws", region="us-west-2"
)
import time
from pinecone import Pinecone, PodSpec

index_name = 'llama-2-rag'
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of ada 002
        metric='dotproduct',
        spec=spec
    )

    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
from langchain.embeddings.openai import OpenAIEmbeddings

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
# embed_model = OpenAIEmbeddings(model="gpt-3.5-turbo")

from tqdm.auto import tqdm  # for progress bar

# Assuming you have already loaded your custom dataset 'dataset'

batch_size = 100

for i in tqdm(range(0, len(dataset), batch_size)):
    i_end = min(len(dataset), i + batch_size)
    # get batch of data
    batch = dataset[i:i_end]
    # generate unique ids for each chunk (assuming 'Major' and 'Pedagogy Technique' are keys in each row)
    ids = [f"{batch['Major'][i]}-{batch['Pedagogy Technique'][i]}" for i in range(len(batch['Major']))]

    # Concatenate relevant fields into a single string
    texts = [f"{batch['Major'][i]} {batch['Pedagogy Technique'][i]} {batch['Definition'][i]} \
             {batch['Objective'][i]} {batch['Resources'][i]} {batch['Citation'][i]} \
             {batch['Course Structure'][i]} {batch['Method'][i]} {batch['Class size'][i]} \
             {batch['Findings'][i]} {batch['S-C'][i]} {batch['S-S'][i]} \
             {batch['S-T'][i]} {batch['I'][i]} {batch['C'][i]} {batch['A'][i]} {batch['P'][i]} \
             {batch['ICAP'][i]} {batch['A,S'][i]} \
             {batch['Steps to try in Sandbox'][i]}" for i in range(len(batch['Major']))]

    # embed text
    embeds = embed_model.embed_documents(texts)

    # get metadata to store in Pinecone
    metadata = [
        {'Major': str(batch['Major'][i]),  # Convert to string
        'Pedagogy Technique': str(batch['Pedagogy Technique'][i]),  # Convert to string
        'Definition': str(batch['Definition'][i]),  # Convert to string
        'Objective': str(batch['Objective'][i]),  # Convert to string
        'Resources': str(batch['Resources'][i]),  # Convert to string
        'Citation': str(batch['Citation'][i]),  # Convert to string
        'Course Structure': str(batch['Course Structure'][i]),  # Convert to string
        'Method': str(batch['Method'][i]),  # Convert to string
        'Class size': str(batch['Class size'][i]),  # Convert to string
        'Findings': str(batch['Findings'][i]),  # Convert to string
        'S-C': str(batch['S-C'][i]),  # Convert to string
        'S-S': str(batch['S-S'][i]),  # Convert to string
        'S-T': str(batch['S-T'][i]),  # Convert to string
        'I': str(batch['I'][i]),  # Convert to string
        'C': str(batch['C'][i]),  # Convert to string
        'A': str(batch['A'][i]),  # Convert to string
        'P': str(batch['P'][i]),  # Convert to string
        'ICAP': str(batch['ICAP'][i]),  # Convert to string
        'A,S': str(batch['A,S'][i]),  # Convert to string
        'Steps to try in Sandbox': str(batch['Steps to try in Sandbox'][i]),  # Convert to string
        'text': texts[i]  # Include the concatenated text
        } for i in range(len(batch['Major']))
    ]

    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))

from langchain.vectorstores import Pinecone

text_field = "text"  # the metadata field that contains our text

# initialize the vector store object
vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

query = "How can I teach Mechanical engineering using Active Learning"  # Replace "your_query_here" with the actual query
vectorstore.similarity_search(query, k=5)

def augment_prompt(user_input: str):
    '''
    # get top 3 results from knowledge base
    results = vectorstore.similarity_search(query, k=3)
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    '''
    results = vectorstore.similarity_search(user_input, k=5)

    # Filter and sort results based on 'S-S', 'S-C', and 'S-T'
    sorted_results = sorted(results, key=lambda x: (x.metadata['S-S'], x.metadata['S-C'], x.metadata['S-T']))

    # Extract and format top 5 responses
    formatted_responses = [f"{i+1}. {result.metadata['Major']} - {result.metadata['Pedagogy Technique']}\n{result.metadata['Definition']}\n{result.metadata['Objective']}\n\n" for i, result in enumerate(sorted_results[:5])]

    # Combine responses into a single string
    response_text = "\n".join(formatted_responses)

    # Return the formatted response
    return response_text
    #return augmented_prompt

# Replace this with your actual chatbot interaction function
def interact_with_chatbot(user_input):
    # # Example: Sending user input to the chatbot and receiving response
    # # Assume chat() function is defined somewhere to interact with your chatbot
    # prompt = HumanMessage(content=user_input)
    # messages.append(prompt)
    # res = chat(messages + [prompt])
    

    # return res.content
    prompt = HumanMessage(
        content=augment_prompt(user_input)
    )
    # add to messages
    messages.append(prompt)

    # send to OpenAI
    res = chat(messages + [prompt])
    # print(res.content)
    # add latest AI response to messages
    messages.append(AIMessage(res.content))

    return res.content


