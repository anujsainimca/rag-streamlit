import streamlit as st
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI

# Function to generate responses using RAG
def generate_response(documents, api_key):
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Create FAISS vector store from documents
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    # Create retriever interface
    retriever = db.as_retriever()
    
    # Create QA chain
    llm = OpenAI(api_key=api_key)
    qa_chain = langchain.chains.qa.QA(llm=llm, retriever=retriever)
    
    # Example query (you can replace this with actual user input)
    query = "What is the main benefit of using RAG?"
    response = qa_chain.run(query)
    
    return response

# Streamlit app
st.title("RAG with FAISS on Streamlit Cloud")

openai_api_key = st.text_input("OpenAI API Key", type="password")

uploaded_file = st.file_uploader("Upload a document", type=["txt"])

if uploaded_file and openai_api_key:
    st.write("File uploaded successfully.")
    
    # Read the file
    content = uploaded_file.read().decode("utf-8")
    documents = [langchain.docstore.Document(page_content=content, metadata={"filename": uploaded_file.name})]
    
    if st.button("Generate Response"):
        with st.spinner("Calculating..."):
            response = generate_response(documents, openai_api_key)
            st.write("Response:")
            st.write(response)
