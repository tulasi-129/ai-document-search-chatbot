import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load env variables
load_dotenv()

st.set_page_config(page_title="AI RAG Chatbot", layout="wide")
st.title(" AI RAG Chatbot")

# Load Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)

@st.cache_resource
def load_vectorstore():
    documents = []
    data_dir = "data"

    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, file))
            documents.extend(loader.load())

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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.chat_input("Ask a question from your documents")

if question:
    st.chat_message("user").markdown(question)

    response = qa_chain.invoke({"query": question})
    answer = response["result"]

    st.chat_message("assistant").markdown(answer)

    with st.expander("Sources"):
        for doc in response["source_documents"]:
            st.write(os.path.basename(doc.metadata.get("source", "")))

    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

