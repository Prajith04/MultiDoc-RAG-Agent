import gradio as gr
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor
from agents import rag_agent
from tools import retrieve_tool, calculator_tool
from langchain.callbacks import get_openai_callback
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.callbacks.manager import CallbackManager
from io import StringIO
import sys
from contextlib import redirect_stdout

chat_history_obj = ChatMessageHistory()

# Create a custom callback handler to capture verbose output
class VerboseOutputCapture:
    def __init__(self):
        self.output = ""
        
    def capture_output(self):
        # Capture stdout to get verbose output
        buffer = StringIO()
        with redirect_stdout(buffer):
            yield
        self.output = buffer.getvalue()

# Create the agent executor with callbacks to capture verbose output
callback_manager = CallbackManager([ConsoleCallbackHandler()])
agent_executor = AgentExecutor(
    agent=rag_agent(), 
    tools=[retrieve_tool(), calculator_tool()], 
    verbose=True,
    callback_manager=callback_manager
)

def chat_interface(user_input):
    # Prepare to capture verbose output
    output_capture = StringIO()
    
    # Redirect stdout to capture verbose output
    with redirect_stdout(output_capture):
        response = agent_executor.invoke({"input": user_input, "chat_history": chat_history_obj.messages})
    
    # Get the captured verbose output
    verbose_output = output_capture.getvalue()
    
    chat_history_obj.add_user_message(user_input)
    chat_history_obj.add_ai_message(response['output'])
    
    # Return both the final answer and the reasoning
    return response['output'], verbose_output

iface = gr.Interface(
    fn=chat_interface,
    inputs=["text"],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Agent Reasoning", lines=10)
    ]
)

if __name__ == "__main__":
    iface.launch()