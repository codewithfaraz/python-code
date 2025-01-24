import streamlit as st
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKKEN"] = os.getenv("HF_TOKKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
embedder = OpenAIEmbeddings(model="text-embedding-3-large")
st.title("COnversation RAG with PDF Uploads and chat History")

user_key = st.text_input("Enter Your Groq API Key",type="password")
if user_key:
    llm = ChatGroq(model_name="gemma2-9b-it")


    if "store" not in st.session_state:
        st.session_state.store={}
    #chat interface
    user_session_id = st.text_input("Session ID", value="default_session")
    user_files = st.file_uploader("Upload Your PDF", type="pdf", accept_multiple_files=True)
    if user_files:
        documents=[]
        for user_file in user_files:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(user_file.getvalue())
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
        splitted_docs = text_splitter.split_documents(documents)
        db = Chroma.from_documents(splitted_docs,embedder)
        retriever = db.as_retriever()
        contextualize_system_prompt = (
            "You ar an ai assistant to answer the given questions "
            "You have given a context and cosidering this you to have answer the question "
            "Do not answer without any context and "
            "keep th answer concize"
        )
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_system_prompt),
                MessagesPlaceholder("chat_history"),
                ('human',"{input}")
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_prompt)
        system_prompt = (
            "Use the following given piece of information to answer the question "
            "Do not answer the queston if context is not provided "
            "Keep the answer concise"
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ('system',system_prompt),
                MessagesPlaceholder("chat_history"),
                ('human',"{input}")
            ]
        )
        qa_chain = create_stuff_documents_chain(llm,prompt)
        chain = create_retrieval_chain(history_aware_retriever,qa_chain)
        def getSession(session:str)-> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]
        rag_chain = RunnableWithMessageHistory(
            chain,
            getSession,
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        user_input = st.text_input("Enter Your Question")
        if user_input:
            session_history = getSession(user_session_id)
            response = rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":user_session_id}
                }
            )
            st.write(st.session_state.store)
            st.write("Chat History:",session_history.messages)
            st.write("Assistant: ",response['answer'])
else:
    st.warning("Please Enter Your Groq API Key")