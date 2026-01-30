import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()
st.set_page_config(page_title="AI RAG Chatbot", layout="wide")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.title("AI RAG Chatbot")
st.write("Ask questions from the uploaded PDF")

@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader("data.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = load_vectorstore()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    streaming=False  
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    return_source_documents=False
)

question = st.text_input("Ask a question from the PDF")

if question:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke({"query": question})
        st.success(response["result"])
