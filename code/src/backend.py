from fastapi import FastAPI, File, UploadFile
from typing import List
import uvicorn
from entity_extractor import start
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from get_transaction_risk import compute_transaction_risk

db_uri = "bolt://localhost:7689"
db_user = "neo4j"
db_password = "password"
driver = GraphDatabase.driver(db_uri, auth=(db_user, db_password))
print("Established connection with the database...")


model = SentenceTransformer("all-MiniLM-L6-v2")
print("Sentence transformer loaded...")

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
            results.append(transaction_risks)
            print(transaction_risks)

    return {"message": "Files uploaded successfully!", "files": file_details, "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)