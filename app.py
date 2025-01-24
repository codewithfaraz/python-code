import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT_NAME"] = os.getenv("LANGCHAIN_PROJECT_NAME")

#prompt
prompt = ChatPromptTemplate.from_messages([
    ("system","You are an assistance, so answer any given question."),
    ("user","{input}")
])

st.title("Chatbot using LLama2")
user_question = st.text_input("What is your question?")


llm = Ollama(model = "gemma2:2b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if user_question:
    st.write(chain.invoke({"input":user_question}))