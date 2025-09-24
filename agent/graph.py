from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from langchain.agents import create_react_agent

from dotenv import load_dotenv
from prompts import *
from states import *
from tools import *
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
user_prompt = 'create a simple calculator web application'

def planner_agent(state:dict)-> dict:
    users_prompt = state['user_prompt']
    response = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))
    return {"plan":response}
def architect_agent(state:dict)-> dict:
    plan = state['plan']
    response = llm.with_structured_output(TaskPlan ).invoke(architect_prompt(plan))
    if response is None:
        raise ValueError('Architect did not return a valid response')
    response.plan = plan
    return {"task_plan":response}
def coder_agent(state:dict)-> dict:
    steps = state['task_plan'].implementation_steps
    current_step_idx = 0
    current_task  = steps[current_step_idx]
    existing_content = read_file(current_task.filepath)
    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"file:{current_task.filepath}\n"
        f"Existing content:\n{existing_content}\n"
        "User write_file(path, content) to save your changes"
    )
    system_prompt = coder_system_prompt()
    coder_tools = [read_file,write_file,list_files,get_current_directory]
    react_agent = create_react_agent(llm,coder_tools)
    react_agent.invoke({"messages":[{
        "role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]})
    return {}


graph = StateGraph(dict)
graph.add_node('planner',planner_agent)
graph.add_node('architect',architect_agent)
graph.add_node('coder',coder_agent)
graph.add_edge('planner','architect')
graph.add_edge('architect','coder')
graph.set_entry_point('planner')
agent = graph.compile()

result = agent.invoke({'user_prompt':user_prompt})
print(result)