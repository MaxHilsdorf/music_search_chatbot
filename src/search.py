from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

##################
## SEARCH ALGOS ##
##################

class SearchAlgorithm(ABC):
 
    def __init__(self):
        ...
        
    def read_database(self, embeddings: np.ndarray, captions: List[str], track_names: List[str]) -> None:
        self.embeddings = embeddings
        self.track_names = track_names
        self.captions = captions

    @abstractmethod
    def find_similar(self, input_text: str, n: int=5) -> List[str]:
        ...
        
        
class SimpleCosineSimilarity(SearchAlgorithm):
    
    def find_similar(self, input_text: str, n: int=5) -> Tuple[List[int], List[str], List[str]]:
        
        # Get embedding from openai api
        response = openai.Embedding.create(
            input=input_text,
            model="text-embedding-ada-002"
            )
        
        input_embedding = np.array(response["data"][0]["embedding"])
        
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, input_embedding) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(input_embedding))
        
        # Return most similar indices, captions, and names
        most_similar_indices = similarities.argsort()[-n:][::-1]
        most_similar_captions = [self.captions[i] for i in most_similar_indices]
        most_similar_names = [self.track_names[i] for i in most_similar_indices]
        return most_similar_indices, most_similar_names, most_similar_captions