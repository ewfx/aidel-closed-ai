import tenacity
from langchain.text_splitter import CharacterTextSplitter
import json
from groq import Groq
from dotenv import load_dotenv
import os
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain.tools import tool, StructuredTool
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain import hub
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS
from agenthub_tools.duckduckgo import search, news
from langchain_community.document_loaders.csv_loader import CSVLoader
import shutil


last_chunk_index =0
def text_input_reader(filePath):
    with open(filePath, 'r') as file:
        data = file.read()
        text_splitter = CharacterTextSplitter(separator="---",chunk_size=1000,chunk_overlap=0)
        chunks = text_splitter.split_text(data)
    return chunks
def csv_input_reader(filepath):
    chunk_size=10
    loader = CSVLoader(filepath)
    data = loader.load()
    rowdata=[]
    for row in data:
        record="{"
        record+=row.page_content
        record+="}"
        rowdata.append(record)
    
    if len(rowdata)> chunk_size:
        chunks= [rowdata[i:i + chunk_size] for i in range(0, len(rowdata), chunk_size)]
    else:
        chunks = [rowdata]

    return chunks
def extract_entities(chunks):
    entities = []
    i=0
    for chunk in chunks:
        entities_str=entity_extractor_llm(chunk=chunk,filepath="prompt.txt")
        entities.extend(json.loads(entities_str))
    return entities

def entity_extractor_llm(chunk,filepath=None,temperature=0.6,top_p=1):
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    json_prompt = []
    if filepath:
        with open(filepath,'r') as f:
            # prompt = f.read()
            json_prompt=json.load(f)
    
    user_input = {
                "role": "user",
                "content":f"""{chunk}"""
            }
    json_prompt.append(user_input)
    # print(json_prompt)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
            
        messages=json_prompt,
        temperature= temperature,
        max_completion_tokens=32768,
        top_p= top_p,
        stream=True,
        stop=None,
    )
    extracted_entities=""
    for chunk in completion:
        # print(chunk.choices[0].delta.content or "",end="")
        extracted_entities+=chunk.choices[0].delta.content or ""
    return extracted_entities
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
def llm_analyzer(ai_agent_inf,ofac_input=None,graph_input=None,wikidata_input=None):
    llm_input = f"""
             ai agent inferences:{ai_agent_inf}
             OFAC input : {ofac_input}
             Graph database input : {graph_input}
             Wikidata input : {wikidata_input}"""
    print(entity_extractor_llm(llm_input,"prompt_llm2.txt"))

def start(txtfile):
    load_dotenv()
    # txtfile="test.txt"
    # txtfile="test.txt"
    file_path = f"./{txtfile.filename}"  # Save location
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(txtfile.file, buffer)
    if txtfile.filename.endswith(".txt"):
        chunks = text_input_reader(file_path)
    if txtfile.filename.endswith(".csv"):
        chunks=csv_input_reader(file_path)

    extracted_entities = extract_entities(chunks)
    return extracted_entities
