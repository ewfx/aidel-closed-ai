
from langchain.tools import tool, StructuredTool
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain import hub
from langchain_groq import ChatGroq
from agenthub_tools.duckduckgo import search, news

def groq_entity_query(query: str) -> str:
  print(query)
  """A get request to Look about the entity on the internet to find about their political influence or business industry it is involved in"""
  content = search(query)
#   content = DDGS().news(query)
  return content
def chat_agent(transaction):
    search = StructuredTool.from_function(func=groq_entity_query,name="groq_entity_query",description="A get request to Look about the entity on the internet to find about their political influence or business industry it is involved in",handle_tool_error=True)
    llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.3,max_tokens=1000,max_retries=2,verbose=0)
    agent = create_structured_chat_agent(llm,tools=[search],prompt=hub.pull("hwchase17/structured-chat-agent"))
    agent_executor = AgentExecutor(agent=agent, tools=[search], verbose=True,handle_parsing_errors=True)

    if transaction["Notes"] != "No Addon Information":
        transaction["inference_add_info"]=agent_executor.invoke({'input':f"Go through latest and major historic details on financial fraud incidents with patterns in the given information {transaction['Notes']}"})
    transaction["internet_info"]=[]
    for entity in transaction['Entity']:
        if entity["Type"].upper() == "PERSON":
            transaction["internet_info"].append({"description":agent_executor.invoke({'input':f"""Give me brief into about {entity['Name']}.Is this person {entity['Name']} influencialis this person{entity['Name']} and also see if entity is involved in any financial fraud give the year and magnitude of the incident\n. 
                                                                                      Instructions select an full person name is matched If you suspect not able to match full name and transaction is of very high value then give your inferance and add a note telling full name not matched with news and transaction record  """})['output']})
        else:
            transaction["internet_info"].append({"description": agent_executor.invoke({'input':f'''Tell me about {entity['Name']}. Is this {entity['Name']} located at {entity['Place']}, also search for is this company {entity['Name']} a shell company or involved in financial transaction/block list by any nation or organization\n
                                                                                       Instructions try find the company recent news and is it part of any fraud by the management or their relatives'''})['output']})
    return transaction