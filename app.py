import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
load_dotenv()

st.set_page_config(page_title="AI RAG Chatbot", layout="wide")
st.title(" AI RAG Chatbot")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found. Add it in Streamlit → Settings → Secrets")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.3
)

PDF_PATH = "data.pdf"

if not os.path.exists(PDF_PATH):
    st.error("data.pdf not found in repository")
    st.stop()

@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    if not documents:
        st.error("No text found in PDF")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = load_vectorstore()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

question = st.chat_input("Ask a question from the PDF")

if question:
    st.chat_message("user").markdown(question)

    response = qa_chain.invoke({"query": question})
    answer = response["result"]

    st.chat_message("assistant").markdown(answer)

    with st.expander(" Sources"):
        for doc in response["source_documents"]:
            st.write(os.path.basename(doc.metadata.get("source", "data.pdf")))

