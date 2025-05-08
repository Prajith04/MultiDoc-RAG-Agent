import gradio as gr
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor
from agents import rag_agent
from tools import retrieve_tool, calculator_tool

chat_history_obj = ChatMessageHistory()

agent_executor = AgentExecutor(
    agent=rag_agent(), 
    tools=[retrieve_tool(), calculator_tool()], 
    verbose=True,
    return_intermediate_steps=True,
)

def chat_interface(user_input,history_list):
    response = agent_executor.invoke({"input": user_input, "chat_history": chat_history_obj.messages})
    chat_history_obj.add_user_message(user_input)
    chat_history_obj.add_ai_message(response['output'])
    print(response)
    if len(response['intermediate_steps']) > 0:
        final_response ="Final Output:\n\n"+response['output']+'\n\nTool Used:'+response['intermediate_steps'][0][0].tool+'\n\nTool output:\n'+response['intermediate_steps'][0][1]
        return final_response
    response = "Final Output:\n\n"+response['output']
    return response

iface = gr.ChatInterface(
    fn=chat_interface,
    examples=["how to turn on dark mode in Samsung S25","what is 23*56-67+99*78"],
    cache_examples=False,
)

if __name__ == "__main__":
    iface.launch()