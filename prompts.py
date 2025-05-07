from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate,MessagesPlaceholder
from langchain import hub
def retriever_prompt():
    return ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Use the context to answer the question:\nContext: {context}"
        "these are the titles of manuals you have:\nManuals: {docs}"
    ),
    HumanMessagePromptTemplate.from_template("{query}"),
])  
def agent_prompt():
    prompt = hub.pull("hwchase17/openai-functions-agent")
    return prompt
 
