import tenacity
from langchain.text_splitter import CharacterTextSplitter
import json
from groq import Groq
from dotenv import load_dotenv
import os
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

def start(txtfile):
    load_dotenv()
    # txtfile="test.txt"
    # txtfile="test.txt"
    file_path = f"./{txtfile.filename}"  # Save location
    txtfile.file.seek(0)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(txtfile.file, buffer)
    if txtfile.filename.endswith(".txt"):
        chunks = text_input_reader(file_path)
    if txtfile.filename.endswith(".csv"):
        chunks=csv_input_reader(file_path)
    extracted_entities = extract_entities(chunks)
    return extracted_entities
