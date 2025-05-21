"""
gradio_demo.py - Provides a web interface for the MultiDoc RAG Agent using Gradio.

This file:
1. Sets up a chat interface using Gradio
2. Configures and initializes the RAG agent
3. Handles user input processing and response formatting
4. Displays tool usage and intermediate steps for transparency
"""
import gradio as gr
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor
from agents import rag_agent
from tools import retrieve_tool, calculator_tool

# Initialize chat history to maintain conversation context
chat_history_obj = ChatMessageHistory()

# Create the agent executor with tool tracking enabled
agent_executor = AgentExecutor(
    agent=rag_agent(), 
    tools=[retrieve_tool(), calculator_tool()], 
    verbose=True,  # Enable verbose output for debugging
    return_intermediate_steps=True,  # Track intermediate steps to show tool usage
)

def chat_interface(user_input, history_list):
    """
    Process user input and generate responses with the RAG agent.
    
    This function:
    1. Invokes the agent with user input and conversation history
    2. Updates the chat history with new exchanges
    3. Formats the response with tool usage information when available
    
    Args:
        user_input (str): The user's message or question
        history_list (list): The conversation history from Gradio
    
    Returns:
        str: Formatted response including agent output and tool usage details
    """
    # Process the query using the agent
    response = agent_executor.invoke({"input": user_input, "chat_history": chat_history_obj.messages})
    
    # Update chat history
    chat_history_obj.add_user_message(user_input)
    chat_history_obj.add_ai_message(response['output'])
    
    # Print complete response object for debugging
    print(response)
    
    # Format the response with tool usage details if available
    if len(response['intermediate_steps']) > 0:
        final_response ="Final Output:\n\n"+response['output']+'\n\nTool Used:'+response['intermediate_steps'][0][0].tool+'\n\nTool output:\n'+response['intermediate_steps'][0][1]
        return final_response
    
    # Return simple response if no tools were used
    response = "Final Output:\n\n"+response['output']
    return response

# Create Gradio chat interface
iface = gr.ChatInterface(
    fn=chat_interface,
    examples=["how to turn on dark mode in Samsung S25","what is 23*56-67+99*78"],  # Example queries for users to try
    cache_examples=False,  # Disable caching to ensure fresh responses
)

# Launch the web interface when script is run directly
if __name__ == "__main__":
    iface.launch()