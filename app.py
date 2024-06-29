import openai
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

def generate_response(documents, api_key):
    st.write("Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(docs, embeddings)
    
    st.write("Creating retriever interface...")
    retriever = db.as_retriever()
    
    st.write("Creating QA chain...")
    llm = OpenAI(api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    query = "What is the main benefit of using RAG?"
    st.write(f"Running query: {query}")
    response = qa_chain.run(query)
    
    return response

st.title("Document QA System using RAG")

openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

uploaded_file = st.file_uploader("Upload a document", type=["txt"])

if uploaded_file is not None and openai_api_key:
    content = uploaded_file.read().decode("utf-8")
    documents = [Document(page_content=content, metadata={"filename": uploaded_file.name})]
    st.write("Document processed successfully.")
    
    response = generate_response(documents, openai_api_key)
    if response:
        st.write("Response:")
        st.write(response)
    else:
        st.write("No response generated.")
