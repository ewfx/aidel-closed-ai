import pandas as pd
from fuzzywuzzy import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from textblob import TextBlob
import re

def find_best_match(model, entity_name, ofac_df, top_n=3, threshold=0.75):
    """
    Finds the top N closest OFAC matches for a given entity using sentence transformer embeddings.
    """
    entity_embedding = model.encode(entity_name, convert_to_numpy=True)
    
    # Compute cosine similarity
    similarities = cosine_similarity([entity_embedding], np.stack(ofac_df["embedding"].values))[0]



    # Get top matches
    top_indices = np.argsort(similarities)[::-1][:top_n]


    matches = []
    for idx in top_indices:
        match_name = ofac_df.iloc[idx]["Name"]
        print(match_name)
        match_score = similarities[idx]
        match_data = ofac_df.iloc[[idx]]  # Keep as DataFrame
        if(similarities[idx] > threshold):
            matches.append((match_name, float(match_score), match_data))
    
    return matches

def min_max_normalize(value, min_val, max_val):
    """Normalize a value using min-max scaling to range [0,1]."""
    if max_val == min_val:  # Avoid division by zero
        return 0
    return (value - min_val) / (max_val - min_val)


SANCTION_WEIGHTS = {
    "SDGT": 0.95,  # Specially Designated Global Terrorist (Highest risk)
    "RUSSIA-EO14024": 0.95,
    "IRGC": 0.9,   # Islamic Revolutionary Guard Corps
    "IFSR": 0.85,  # Iran Financial Sanctions Regulations
    "OFAC": 0.8,   # Office of Foreign Assets Control List
    "NS-PLC": 0.7, # Non-SDN Palestinian Legislative Council List
    "FSE": 0.6,    # Foreign Sanctions Evaders List
    "NS-ISA": 0.5, # Non-SDN Iranian Sanctions Act List
    "CAPTA": 0.4,  # Correspondent Account Sanctions
    "None": 0.0    # No sanctions
}
# Define high-risk keywords for additional info
HIGH_RISK_KEYWORDS = {
    "terrorist": 30,
    "fraud": 25,
    "money laundering": 25,
    "criminal": 20,
    "drug": 20,
    "weapons": 20,
    "russia": 20,
    "china": 20,
    "pakistan": 25
}

def analyze_sentiment(text):
    """
    Performs sentiment analysis on the additional info field.
    Negative sentiment indicates a higher risk score.
    """
    if pd.isna(text) or text == "-0-":
        return 0  # No information available

    sentiment_score = TextBlob(text).sentiment.polarity  # -1 (negative) to 1 (positive)
    # Convert sentiment score to risk factor (negative sentiment = higher risk)
    if sentiment_score <= -0.1:
        return 20  # High-risk sentiment
    elif sentiment_score < 0:
        return 10  # Moderate risk
    return 0  # Neutral or positive sentiment

def check_high_risk_keywords(text):
    """
    Checks if high-risk words (e.g., fraud, terrorist, money laundering) exist in the additional info.
    """
    risk_score = 0
    if pd.isna(text) or text == "-0-":
        return risk_score  # No data available

    text = text.lower()
    for keyword, score in HIGH_RISK_KEYWORDS.items():
        if re.search(r"\b" + keyword + r"\b", text):
            risk_score += score  # Add weight based on keyword match

    return risk_score



def compute_sanction_risk(sanction_str):
    """
    Computes a weighted sanction risk score if multiple sanctions are imposed.
    The risk is a sum of all applicable sanction weights.
    """
    if pd.isna(sanction_str) or not sanction_str.strip():
        return 0.0  # No sanctions
    
    # Extract individual sanctions
    sanctions = sanction_str.replace("[", "").replace("]", "").split()
    
    # Get the highest sanction weight
    max_risk = max((SANCTION_WEIGHTS.get(s, 0.75) for s in sanctions), default=0.0)

    return round(max_risk, 3)  # Normalize between 0-1


def compute_normalized_risk_score(model, entity_name, ofac_df):
    """
    Computes a normalized risk score (0 to 1) based on:
    - Sentence Transformer Name Match
    - Sanction program severity
    - Sentiment risk
    - Keyword-based risk
    """
    matches = find_best_match(model, entity_name, ofac_df)

    if not matches:
        return {"entity": entity_name, "risk_score": 0, "reason": "No OFAC match found", "confidence_score": 1}

    # Define max values for normalization
    MAX_MATCH_SCORE = 1  # Cosine similarity is already between [0,1]
    MAX_SANCTION_RISK = max(SANCTION_WEIGHTS.values())  
    MAX_SENTIMENT_RISK = 20  
    MAX_KEYWORD_RISK = sum(HIGH_RISK_KEYWORDS.values())  

    max_normalized_risk = 0
    reasons = []

    for match_name, match_score, match_data in matches:
        match_data = match_data.iloc[0]  

        sanction_risk = compute_sanction_risk(match_data["Sanction_Program"])

        info_text = match_data["Additional_Info"] + match_data["Other_Info"]
        sentiment_risk = analyze_sentiment(info_text)
        keyword_risk = check_high_risk_keywords(info_text)

        normalized_match = match_score  # Already in [0,1]
        normalized_sanction = min_max_normalize(sanction_risk, 0, MAX_SANCTION_RISK)
        normalized_sentiment = min_max_normalize(sentiment_risk, 0, MAX_SENTIMENT_RISK)
        normalized_keywords = min_max_normalize(keyword_risk, 0, MAX_KEYWORD_RISK)

        final_risk_score = (0.4 * normalized_match + 
                            0.35 * normalized_sanction + 
                            0.15 * normalized_sentiment + 
                            0.1 * normalized_keywords)

        max_normalized_risk = max(max_normalized_risk, final_risk_score)

        reasons.append(f"(Entity: {match_name}) " f"(Info: {info_text}) " f"(Match Score: {normalized_match:.2f}, Sanction Risk: {normalized_sanction:.2f}, "
                       f"Sentiment Risk: {normalized_sentiment:.2f}, Keyword Risk: {normalized_keywords:.2f})")

    return {
        "entity": entity_name,
        "risk_score": round(float(max_normalized_risk), 3),
        "confidence_score": float(sum([m[1] for m in matches]) / len(matches)),
        "reason": "; ".join(reasons)
    }