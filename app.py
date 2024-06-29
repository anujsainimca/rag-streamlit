import streamlit as st
from openai.embeddings_utils import get_embedding
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain, combine_docs_chain, question_generator_chain

# Function to generate responses using RAG
def generate_response(documents, api_key):
    try:
        st.write("Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        st.write("Creating FAISS vector store from documents...")
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        
        st.write("Creating retriever interface...")
        retriever = db.as_retriever()
        
        st.write("Creating Conversational Retrieval Cha
