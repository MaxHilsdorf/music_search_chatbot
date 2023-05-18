import pandas as pd
import openai
import os
import numpy as np
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load data
df = pd.read_csv("../data/musiccaps-public.csv")

captions = df["caption"].tolist()
names = df["ytid"].tolist()

# Compute embeddings

for i, caption in tqdm(enumerate(captions)):
    
    target_path = f"individual_embeddings/embedding_{names[i]}.npy"
    if os.path.exists(target_path):
        continue
    response = openai.Embedding.create(
        input=captions[i],
        model="text-embedding-ada-002"
    )
    embedding = np.array(response["data"][0]["embedding"])
    
    # Save individual embeddings
    target_path = f"individual_embeddings/embedding_{names[i]}.npy"
    if os.path.exists(target_path):
        continue
    np.save(target_path, embedding)

print("\nAll embeddings computed!")
    
# Aggregate embeddings
embeddings = np.zeros((len(captions), 1536)) # 1536 is the embedding size
for i, name in enumerate(names):
    embedding = np.load(f"individual_embeddings/embedding_{name}.npy")
    embeddings[i] = embedding
# Convert to float16
embeddings = embeddings.astype(np.float16)
np.save("aggregated_embeddings.npy", embeddings)
    
print("\nEmbeddings aggregated!")