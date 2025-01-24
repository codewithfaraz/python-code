#environment variables
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
#other useful imprts
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
#llm
llm = ChatGroq(model_name="Llama3-8b-8192")
#definign prompt
prompt = ChatPromptTemplate.from_template(
    """
You are an ai assistance to answer the given questions based on provided context
<context>
{context}
<context>
question:
{input}
"""
)

def startSession():
    try:
        # Initialize Embedder
        st.session_state.embedder = OllamaEmbeddings(model="gemma2:2b")

        # Load Documents
        st.session_state.loader = PyPDFLoader("mypdf.pdf")
        st.session_state.docs = st.session_state.loader.load()
        st.write(f"Loaded {len(st.session_state.docs)} documents")

        # Split Documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        st.session_state.splitted_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.write(f"Split into {len(st.session_state.splitted_docs)} chunks")

        # Generate Embeddings
        st.session_state.embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        # Build Chroma Index
        st.session_state.db = Chroma.from_documents(st.session_state.splitted_docs, st.session_state.embedder)
        st.write("Chroma database created successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

user_input = st.text_input("write your query from the paper")

if st.button("Start session"):
    startSession()
    st.write("Things are ready to go")
import time
if user_input:
    documents_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.db.as_retriever()
    chain = create_retrieval_chain(retriever,documents_chain)
    start_time = time.process_time()
    response = chain.invoke({"input":user_input})
    st.write(f"response time {time.process_time()-start_time}")
    st.write(response["answer"])
    with st.expander("Context"):
        for i,doc in enumerate(response["context"]):
            st.write(f"{doc.page_content}")
            st.write("-------------------")