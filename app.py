import streamlit as st
from dotenv import load_dotenv
import tempfile

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load env vars
load_dotenv()

st.set_page_config(page_title="AI RAG Chatbot", layout="wide")
st.title("AI RAG Chatbot (PDF Upload)")

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            loader = PyPDFLoader(tmp.name)
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
if not documents:
    st.error("No text found in uploaded PDF")
    st.stop()

chunks = splitter.split_documents(documents)


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = FAISS.from_documents(chunks, embeddings)

qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

    question = st.chat_input("Ask a question from your PDF")

    if question:
        st.chat_message("user").markdown(question)

        response = qa_chain.invoke({"query": question})
        answer = response["result"]

        st.chat_message("assistant").markdown(answer)

        with st.expander("Sources"):
            for doc in response["source_documents"]:
                st.write(doc.metadata.get("source", "Uploaded PDF"))

else:
    st.info(" Upload PDF files to start chatting")



