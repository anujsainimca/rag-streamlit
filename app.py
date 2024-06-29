import streamlit as st
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain

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
        
        st.write("Creating Conversational Retrieval Chain...")
        llm = OpenAI(api_key=api_key)
        conv_chain = ConversationalRetrievalChain(llm=llm, retriever=retriever)
        
        # Example query (you can replace this with actual user input)
        query = "What is the main benefit of using RAG?"
        st.write(f"Running query: {query}")
        response = conv_chain.run(query)
        
        return response
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit app
st.title("RAG with FAISS on Streamlit Cloud")

openai_api_key = st.text_input("OpenAI API Key", type="password")

uploaded_file = st.file_uploader("Upload a document", type=["txt"])

if uploaded_file and openai_api_key:
    st.write("File uploaded successfully.")
    
    try:
        # Read the file
        content = uploaded_file.read().decode("utf-8")
        documents = [Document(page_content=content, metadata={"filename": uploaded_file.name})]
        st.write("Document processed successfully.")
        
        if st.button("Generate Response"):
            with st.spinner("Calculating..."):
                response = generate_response(documents, openai_api_key)
                if response:
                    st.write("Response:")
                    st.write(response)
                else:
                    st.error("No response generated.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key.")
    if not uploaded_file:
        st.warning("Please upload a document.")
