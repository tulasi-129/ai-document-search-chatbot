import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="AI RAG Chatbot", layout="wide")
st.title("AI RAG Chatbot")
st.write("Ask questions from the uploaded PDF")

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

PDF_PATH = "data.pdf"

if not os.path.exists(PDF_PATH):
    st.error("data.pdf not found in repository")
    st.stop()

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
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant.
Answer the question strictly using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

question = st.text_input("Ask a question from the PDF")

if question:
    with st.spinner("Thinking..."):
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(d.page_content for d in docs)

        response = chain.run(
            context=context,
            question=question
        )

    st.subheader("Answer")
    st.write(response)

