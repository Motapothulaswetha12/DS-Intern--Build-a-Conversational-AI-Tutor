import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_text_splitters import NLTKTextSplitter

st.header('RAG (Red,Amber,Green) System in Paper')

# Step 1: Retrieve the "Leave No Context Behind" Paper
loader = PyPDFLoader(r"Task\Gemini_pdf.pdf")
pages = loader.load_and_split()
document_chunks = [p.page_content for p in pages]

# Step 2: Configure LangChain Framework
text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
embedding_model = GoogleGenerativeAIEmbeddings(api_key="gemini_api_key.txt", model='models/embedding-001')

# Step 3: Initialize the RAG System
chroma_db = Chroma.from_documents(document_chunks, embedding_model, persist_directory="./chroma_db")
chroma_db.persist()
retriever = chroma_db.as_retriever(search_kwargs={"k": 5})

# Step 4: Connect with Gemini 1.5 Pro
genai.configure(api_key="gemini_api_key.txt")
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Step 5: User Interaction
user_question = st.text_input("Enter your question...")

if st.button("Generate Answer"):
    retrieved_context = retriever.retrieve(user_question)
    chat_history = document_chunks + retrieved_context  # Providing document context along with retrieved context
    response = model.start_chat(history=chat_history).send_message(user_question)
    
    st.subheader("User Question:")
    st.write(user_question)
    
    st.subheader("System's Response:")
    st.write(response.text)
