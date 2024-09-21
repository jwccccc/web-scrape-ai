# Web Scrape AI: Extract and Query Web Data with GPT-4o
![LangChain](https://img.shields.io/badge/langchain-v0.2.16-green)
![Streamlit](https://img.shields.io/badge/streamlit-v1.32.0-red)
![BeautifulSoup](https://img.shields.io/badge/beautifulsoup-v4.12.2-pink)
![GPT-4o](https://img.shields.io/badge/GPT--4o-powered-ff69b4)

## Overview

**Web Scrape AI** is a tool that combines **web scraping** with **AI-powered question answering** using OpenAI's **GPT-4o** model. The app allows users to input a URL, scrape the content of a webpage, and ask relevant questions to retrieve specific information from the data.

The entire app is built with **Streamlit** for the user interface, **BeautifulSoup** for web scraping, and **LangChain** for document processing.

---
## Features

- **Web Scraping**: Extract content from any webpage using the URL provided by the user.
- **AI-Powered Question Answering**: Ask specific questions about the scraped content and get answers using GPT-4o.
- **Semantic Search**: Uses FAISS to efficiently search through document chunks for relevant information.
- **Streamlit Interface**: Offers an intuitive web-based interface for easy interaction.

---

## Technologies Used

- **GPT-4o**: Provides AI-powered answers based on the scraped content.
- **LangChain**: Handles embeddings and document chunking for efficient information retrieval.
- **BeautifulSoup**: Used for parsing and scraping webpage content.
- **FAISS**: For fast similarity search across scraped content.
- **Streamlit**: Provides the user-friendly web interface.

---
## How It Works: Process Overview
The application follows a straightforward workflow to extract, process, and answer questions based on web conten. Here's a step-by-step breakdown of the process with code snippets:
- Scrape Web Content:
  The app starts by loading the content from the provided URL using a web scraper.
  ```Python
  loader = WebBaseLoader(web_paths=(url,),)
  ```
- Generate Embeddings:
  The scraped text is converted into vector embeddings using OpenAI's text-embedding-ada-002 model.
  ```Python
  embeddings = OpenAIEmbeddings(openai_api_key=api_key)
  embedding_vector = embeddings.embed_query(text) 
  ```
- Split and Index Content:
  The text is split into meaningful chunks (semantic chunking) for more efficient processing. These chunks are then stored in a FAISS vector store for fast similarity search.
  ```Python
  # Split the document into meaningful chunks
  text_splitter = SemanticChunker(embeddings)
  splits = text_splitter.split_documents(docs)

  # Store the chunks in a FAISS vector store
  vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
  ```
- Retrieve Relevant Chunks:
  When a user asks a question, the app queries the FAISS index to find the most relevant chunks of text based on the question.
  ```Python
  retrieved_docs = vectorstore.as_retriever().invoke(question)
  ```
- Generate Answers with GPT-4o:
  The retrieved chunks are passed as context to the GPT-4o model, which generates a precise and contextually relevant answer to the user’s question.
  ```Python
  response = llm.invoke(messages)
  ```
  

---
## Screenshot
![example](https://github.com/user-attachments/assets/b0b254a2-f6b2-4bb1-bd83-18bc5699ed16)
---
## Installation and Running the App

Follow these quick steps to get the app running:

1. **Clone the repository**:
  ```bash
  git clone https://github.com/jwccccc/web-scrape-ai/
  cd web-scrape-ai
  ```

2. **Install the requirements**:
  ```bash
  pip install -r requirements.txt
  ```

3. **Run the app:**:
  ```bash
  streamlit run web.py
  ```

---
