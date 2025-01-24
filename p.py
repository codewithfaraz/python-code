from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKKEN"] = os.getenv("HF_TOKKEN")
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeded_text = embedder.embed("Hello, world!")
print(embeded_text)