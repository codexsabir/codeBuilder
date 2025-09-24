from langchain_groq import ChatGroq
from dotenv import load_dotenv
from prompts import *

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
response = llm.invoke("who invented openai?")
print(response.content)