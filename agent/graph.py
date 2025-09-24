from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from dotenv import load_dotenv
from prompts import *
from states import *
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
user_prompt = 'create a simple calculator web application'

def planner_agent(state:dict)-> dict:
    users_prompt = state['user_prompt']
    response = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))
    return {"plan":response}

graph = StateGraph(dict)
graph.add_node('planner',planner_agent)
graph.set_entry_point('planner')
agent = graph.compile()

result = agent.invoke({'user_prompt':user_prompt})
print(result)