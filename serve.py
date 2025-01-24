import os
from fastapi import FastAPI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="gemma2-9b-it")
outputParser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([
    ("system","convert the following into {language} language"),
    ("user","{text}")
])
chain = prompt|model|outputParser
data = chain.invoke({"language":"Urdu","text":"Hello, how are you?"})
print(data)


app = FastAPI(title="Simple Generative AI application",version="1.0.0",description="My first LLM application with GROQ")
add_routes(app,chain,path="/chain")

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)