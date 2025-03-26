import pandas as pd
from fuzzywuzzy import fuzz, process
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

# Define column names based on OFAC data structure
columns = [
    "ID", "Name", "Type", "Sanction_Program", "Additional_Info",
    "Call_Sign", "Vess_Type", "Tonnage", "GRT", "Vess_Flag", "Vess_Owner", "Other_Info"
]

# Load the CSV (assuming it has no column names)
ofac_df = pd.read_csv("./data/sdn.csv", names=columns, index_col=False)

ofac_df["embedding"] = ofac_df["Name"].apply(lambda x: model.encode(str(x).strip()) if isinstance(x, str) and x.strip() else [0.0] * 384)

ofac_df.to_pickle("ofac_embeddings.pkl")