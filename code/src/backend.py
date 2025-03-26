from fastapi import FastAPI, File, UploadFile
from typing import List
import uvicorn
from entity_extractor import chat_agent, start
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv
import json

from get_transaction_risk import compute_transaction_risk
from llm_reasoner import llm_reasoner

load_dotenv()
db_uri = "bolt://localhost:7689"
db_user = "neo4j"
db_password = os.environ.get('NEO4J_RISK_DB_PASSWORD')
driver = GraphDatabase.driver(db_uri, auth=(db_user, db_password))
print("Established connection with the database...")


model = SentenceTransformer("all-MiniLM-L6-v2")
print("Sentence transformer loaded...")


client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

app = FastAPI()

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    file_details = []

    results = []
    for file in files:
        content = await file.read()  # Read file content (Modify as needed)
        file_details.append({"filename": file.filename, "size": len(content)})
        transactions = start(file)
        for transaction in transactions:
            extracted_entities = []
            for entity in transaction["Entity"]:
                extracted_entities.append({
                    "name": entity["Name"],
                    "type": entity["Type"],
                    "place": entity["Place"] if entity["Place"] else None,
                })
            transaction_risks = compute_transaction_risk(driver, model, extracted_entities)
            # reasoning = extract_reasoning(client, transaction_risks)
            print("Starting agentic web search...")
            search_agent_response = chat_agent(transaction)

            result = llm_reasoner(
                search_agent_response,
                transaction_risks["network_results"],
                transaction_risks["ofac_results"],
                transaction_risks["wiki_results"]
            )

            # result = {
            #     "Transaction ID": transaction["Transaction ID"],
            #     "Extracted Entity": transaction_risks["entities"],
            #     "Entity Types": transaction_risks["entity_types"],
            #     "Risk Score": transaction_risks["risk_score"],
            #     "Supporting Evidence": transaction_risks["supporting_evidence"],
            #     "Confidence Score": transaction_risks["confidence_score"],
            #     # "reasoning": reasoning
            # }
            results.append(json.loads(result))
            # print(result)

    return {"message": "Files uploaded successfully!", "files": file_details, "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)