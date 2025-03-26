def match_entity(driver, model, node_label_map, entity_name, entity_type, threshold=0.75):
    """
    Match an entity based on its type using full-text search for speed.
    """
    node_label = node_label_map.get(entity_type, "Entity")  # Default to 'Entity' if type is unknown

    if node_label == "Officer":
        threshold = 0.9

    index_name = {
        "Entity": "entity_name_index",
        "Officer": "officer_name_index",
        "Address": "address_name_index",
        "Intermediary": "intermediary_name_index"
    }[node_label]

    # Use full-text search to find top candidates
    query = f"""
    CALL db.index.fulltext.queryNodes("{index_name}", $name) 
    YIELD node, score 
    RETURN {'node.name' if node_label != 'Address' else 'node.address'} AS matched_name, score ORDER BY score DESC LIMIT 5
    """


    with driver.session() as session:
        results = session.run(query, name=entity_name)
        
        # ✅ Store the results in a list before processing
        matches = [(record["matched_name"], record["score"]) for record in results]

    # If no matches, return empty list
    if not matches:
        return []

    # Use Sentence Transformers only on the retrieved results
    candidate_names = [match[0] for match in matches]
    name_embeddings = model.encode(candidate_names)
    entity_embedding = model.encode([entity_name])

    similarities = [
        (name, model.similarity(entity_embedding, emb)) 
        for name, emb in zip(candidate_names, name_embeddings)
    ]
    
    # Return the best matches above a similarity threshold
    return [(match[0], match[1]) for match in sorted(similarities, key=lambda x: x[1], reverse=True) if match[1] > threshold]


BASE_WEIGHTS = {
    "officer_of": 2.0,           # High risk if shared officers exist
    "registered_address": 1.5,   # Medium risk if multiple entities share an address
    "intermediary_of": 2.5,      # Very high risk if intermediaries are involved
    "similar": 1.0               # Lower risk but still relevant
}

def compute_risk_score_with_details(driver, entity_name, entity_type):
    risk_score = 0
    related_entities = []
    relationships_summary = []

    query = f"""
    MATCH path = (a:{entity_type.capitalize()} {{name:$entity}})-[r:officer_of|intermediary_of|registered_address|similar*..10]-(b)
    WITH a, b, b.sourceID as source, labels(b) AS node_labels, [rel IN RELATIONSHIPS(path) | TYPE(rel)] AS relationship_types, length(path) AS depth
    RETURN 
        a.name AS entity, 
        CASE 
            WHEN 'Address' IN node_labels THEN b.address
            ELSE b.name 
        END AS connected_entity,
        source,
        relationship_types, 
        labels(b) as label,
        depth
     LIMIT 20;
    """

    with driver.session() as session:
        records = session.run(query, entity=entity_name)

        # print(records.data())
        for record in records:
            relationships = record["relationship_types"]
            depth = record["depth"]
            connected_entity = record["connected_entity"]
            label = record["label"]
            source = record["source"]

            related_entities.append(connected_entity)

            for rel in relationships:
                weight = BASE_WEIGHTS.get(rel, 1)  # Default weight = 1
                adjusted_weight = weight / (depth + 1)  # Reduce impact as depth increases
                risk_score += adjusted_weight
            relationships_summary.append(f"""Entity: {connected_entity} 
                                        Label: {label} 
                                        Source: {source}
                                        Relationship Path: {relationships}
                                        Depth: {depth}""")

    # Normalize risk score
    max_risk_score = sum(BASE_WEIGHTS.values()) * 5 # Max depth = 5 (assuming > 5 means a layered network)
    normalized_risk_score = min(risk_score / max_risk_score, 1)

    return round(normalized_risk_score, 3), relationships_summary

