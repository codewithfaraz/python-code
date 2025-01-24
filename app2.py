import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import openai
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT_NAME"] = "Q&A Chatbot"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are an ai assistance, so respond to the user's questions."),
        ("user","Question:{question}")
    ]
)

def generateResponse(question,llm,tokensize,temperature,api_key):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question":question})
    return answer

st.title("Q&A Chatbot with OpenAI Models")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Your OpenAI api key",type="password")
selected_model = st.sidebar.selectbox("Select Model",["gpt-4o","gpt-4-turbo","gpt-4"])
selected_temperature = st.sidebar.slider("Select temperatre",min_value=0.0,max_value=1.0,value=0.7)
selected_tokensize = st.sidebar.slider("Token Size",min_value=50,max_value=300,value=150)
user_input = st.text_input("Prompt:")

if user_input:
    response = generateResponse(user_input,selected_model,selected_tokensize,selected_temperature,api_key)
    st.write(response)
else:
    st.write("Please enter a prompt")