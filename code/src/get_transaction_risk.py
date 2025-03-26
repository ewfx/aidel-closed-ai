import pandas as pd
from network_risk import compute_risk_score_with_details, match_entity
from ofac_risk import compute_normalized_risk_score
from wiki_risk import EntityRiskScorer
import json


node_label_map = {
        "organization": "Entity",
        "shell_company": "Entity",
        "intermediary": "Intermediary",
        "individual": "Officer",
        "person": "Officer",
        "location": "Address",
        "address": "Address"
}

def calculate_overall_risk(entity_risks):
    overall_risk_scores = {}
    
    # Define weights
    weight_network = 0.5
    weight_ofac = 0.35
    weight_wiki = 0.15
    
    for entity, risk_data in entity_risks.items():
        network_risk = risk_data.get("network_risk", 0)
        ofac_risk = risk_data.get("ofac_risk", 0)
        wiki_risk = risk_data.get("wiki_risk", 0)
        
        # Compute overall risk using weighted sum
        overall_risk = (
            weight_network * network_risk +
            weight_ofac * ofac_risk +
            weight_wiki * wiki_risk / 100  # Normalize Wiki risk (since it's 0-100)
        )
        
        overall_risk_scores[entity] = round(overall_risk, 3)
    
    return overall_risk_scores

def calculate_overall_confidence(entity_risks):
    overall_confidence_scores = {}
    
    # Define weights
    for entity, risk_data in entity_risks.items():
        network_confidence = risk_data.get("network_confidence", 0)
        ofac_confidence = risk_data.get("ofac_confidence", 0)
        wiki_confidence = risk_data.get("wiki_confidence", 0)
        
        # Compute overall risk using weighted sum
        overall_confidence = (
           network_confidence +
            ofac_confidence +
            (wiki_confidence / 100)  # Normalize Wiki risk (since it's 0-100)
        ) / 3
        
        overall_confidence_scores[entity] = round(float(overall_confidence), 3)
    
    return overall_confidence_scores

def compute_transaction_risk(driver, model, extracted_entities):
    print("Computing network risk...")
    matched_entities = []
    for _, entity in enumerate(extracted_entities):
        matches = match_entity(driver, model, node_label_map, entity_name=entity["name"], entity_type=entity["type"].lower())
        matched_entities.append({
            "name": entity["name"],
            "type": entity["type"],
            "matched_name": matches[0][0] if len(matches) else None,
            "matched_type": node_label_map.get(entity["type"].lower(), "Entity"),
            "confidence_score": matches[0][1] if len(matches) else 1
    })



    entity_risks = {}
    network_risk_results = []
    for entity in matched_entities:
        risk_score, relationships_summary = compute_risk_score_with_details(driver, entity["matched_name"], entity["matched_type"])
        network_risk_results.append({
            "name": entity["name"],
            "type": entity["type"],
            "matched_name": entity["matched_name"],
            "matched_type": entity["matched_type"],
            "risk_score": risk_score,
            "relationships_summary": relationships_summary,
            "confidence_score": float(entity["confidence_score"])
        })
        entity_risks[entity["name"]] = {
            "network_entity": entity["matched_name"],
            "network_risk": risk_score,
            "network_relationships_summary": relationships_summary,
            "network_confidence": float(entity["confidence_score"])
        }


    print("Computing ofac risk...")
    ofac_risk_results = []
    ofac_df = pd.read_pickle("./ofac_embeddings.pkl")
    for _, entity in enumerate(extracted_entities):
        risk_result = compute_normalized_risk_score(model, entity["name"], ofac_df)
        ofac_risk_results.append(risk_result)
        entity_risks[entity["name"]]["ofac_entity"] = risk_result["entity"]
        entity_risks[entity["name"]]["ofac_risk"] = risk_result["risk_score"]
        entity_risks[entity["name"]]["ofac_reason"] = risk_result["reason"]
        entity_risks[entity["name"]]["ofac_confidence"] = float(risk_result["confidence_score"])

    NEWS_API_KEY = ""
    
    scorer = EntityRiskScorer(NEWS_API_KEY)

    cases = [(e["name"], e["place"]) for e in extracted_entities]
    
    wiki_results = []
    for entity, jurisdiction in cases:
        result = scorer.get_risk_score(entity, jurisdiction)
        wiki_results.append(result)
        entity_risks[entity]["wiki_entity"] = result["entity"]
        entity_risks[entity]["wiki_risk"] = result["risk_score"]
        entity_risks[entity]["wiki_risk_breakdown"] = result["risk_breakdown"]
        entity_risks[entity]["wiki_confidence"] = float(result["confidence"])

    overall_risk_scores = calculate_overall_risk(entity_risks)
    overall_confidence_score = calculate_overall_confidence(entity_risks)
    transaction_risk = {
        "risk_score": max(list(overall_risk_scores.values())),
        "entities": [],
        "confidence_score": sum(list(overall_confidence_score.values())) / len(list(overall_confidence_score.values())),
        "network_results": network_risk_results,
        "ofac_results": ofac_risk_results,
        "wiki_results": wiki_results
    }
    for entity, risk_data in entity_risks.items():
        transaction_risk["entities"].append(entity)
        # transaction_risk["risk_data"][entity] = risk_data

    print(json.dumps(transaction_risk, indent=4))

    return transaction_risk