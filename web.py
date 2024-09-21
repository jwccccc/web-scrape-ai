import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import numpy as np
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI
# Load environment variables from .env file
load_dotenv()

# Now you can access the API key like this:
api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-ada-002"):
    """Create embeddings for the given text using OpenAI's embedding model."""
    text = text.replace("\n", " ")

    # Get API key from Streamlit secrets (recommended)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key) 

    # Get embeddings using LangChain's simplified method
    embedding_vector = embeddings.embed_query(text) 

    # Convert the embedding to a NumPy array (if needed)
    return np.array(embedding_vector)

# Function to load, split, and retrieve documents
def load_and_retrieve_docs(url):
    # Load the content from the given URL using a web loader
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict() 
    )
    embeddings = OpenAIEmbeddings()
    docs = loader.load()

    # Use a semantic text splitter to break the documents into meaningful chunks
    text_splitter = SemanticChunker(embeddings)
    splits = text_splitter.split_documents(docs)

    # Store the chunks in a FAISS vector store for fast similarity search
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

# Function to format documents
def format_docs(docs):
    # Join the content of each document with double newlines for better readability
    return "\n\n".join(doc.page_content for doc in docs)


def rag_chain(url, question):
    # Load the documents from the URL, split them into chunks, and create a retriever
    retriever = load_and_retrieve_docs(url)

    retrieved_docs = retriever.invoke(question)

    formatted_context = format_docs(retrieved_docs)

    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"

    # Initialize the OpenAI GPT-4o language model with appropriate parameters
    llm = ChatOpenAI(
        model_name="gpt-4o",  # Use GPT-4o model for text generation
        temperature=0,        # Set temperature to 0 for deterministic output
        openai_api_key=api_key # Use the API key for OpenAI
    )

    # Construct the message to pass to the language model
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": formatted_prompt}
    ]
    
    response = llm.invoke(messages)  

    answer = response.content
    return answer.strip()

def main():
    st.set_page_config(page_title="WebScrapeAI")

        # Custom CSS style for the title
    st.markdown(
        """
        <style>
        .title {
            color: #ff69b4;  /* Hot Pink */
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        .stButton button {
            background-color: #ff6b81;  /* Pastel Pink */
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 20px;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            transition: background-color 0.3s;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton button:hover {
            background-color: #ff8fa3;  /* Lighter Pastel Pink */
        }
        .stButton button:active {
            background-color: #ff4d6a;  /* Darker Pastel Pink */
            box-shadow: none;
            transform: translateY(2px);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="title">Web Scrape GPT 🤩</div>', unsafe_allow_html=True)

    st.write("""
        Welcome to **Web Scrape AI**, a tool that combines **web scraping** and **AI-powered question-answering with GPT 4o**. With this app, you can extract content from any webpage and ask targeted questions to quickly retrieve relevant information.
        """)

    url = st.text_input("URL")

    question = st.text_input("Question")

    if st.button("Submit"):
        answer = rag_chain(url, question)
        st.text_area("Answer", value=answer, height=350)
    
if __name__ == "__main__":
    main()