from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor
from agents import rag_agent
from tools import retrieve_tool, calculator_tool
chat_history = ChatMessageHistory()
agent_executor = AgentExecutor(agent=rag_agent(),tools=[retrieve_tool(),calculator_tool()], verbose=True)
while True:
    response=agent_executor.invoke({"input": input("Enter the query:"),"chat_history":chat_history.messages})
    chat_history.add_ai_message(response['input'])
    chat_history.add_ai_message(response['output'])
    print(response)