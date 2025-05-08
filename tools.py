from langchain.tools.retriever import create_retriever_tool
from query_vectordb import chat_model,init_vector_store,small_chat_model
from langchain_community.agent_toolkits.load_tools import load_tools
def retrieve_tool():
    doc_store=init_vector_store()
    retriever = doc_store.as_retriever(search_type="similarity", search_kwargs={"k": 3,})
    retriever_tool = create_retriever_tool(
    retriever,
    "VectorDB_search",
    "Use this tool when you need to answer questions about Samsung mobile phones, including their features, settings, or troubleshooting. For example: how to enable dark mode, battery saving tips, or camera settings.",)
    return retriever_tool
def calculator_tool():
    return load_tools(["llm-math"],llm=small_chat_model())[0]