from langchain.agents import create_tool_calling_agent
from query_vectordb import chat_model
from tools import retrieve_tool, calculator_tool
from prompts import agent_prompt
def rag_agent():
    llm=chat_model()
    tools = [retrieve_tool(), calculator_tool()]
    prompt=agent_prompt()
    agent = create_tool_calling_agent(llm, tools, prompt)
    return agent

