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
        """
        Initializes a new instance of the class. No parameters are required.
        This method does not return anything.
        """
        ...
        
    def read_database(self, embeddings: np.ndarray, captions: List[str], track_names: List[str]) -> None:
        """
        Reads a database and sets the attributes of the object with the given parameters.

        Args:
            embeddings (numpy.ndarray): An array of embeddings.
            captions (List[str]): A list of captions for the embeddings.
            track_names (List[str]): A list of track names.

        Returns:
            None
        """
        self.embeddings = embeddings
        self.track_names = track_names
        self.captions = captions

    @abstractmethod
    def find_similar(self, input_text: str, n: int=5) -> List[str]:
        """
        This is an abstract method that finds similar text given an input text and returns a list of the most similar n strings. 

        :param input_text: a string representing the input text for which we want to find similar strings.
        :type input_text: str

        :param n: an integer representing the number of similar strings to be returned. Default is 5.
        :type n: int

        :return: a list of the most similar n strings.
        :rtype: List[str]
        """
        ...
        
        
class SimpleCosineSimilarity(SearchAlgorithm):
    
    def find_similar(self, input_text: str, n: int=5) -> Tuple[List[int], List[str], List[str]]:
        """
        Finds the n most similar track names and captions to the given input text, based on cosine similarity of their embeddings.

        Args:
            input_text (str): The text to compare with the track captions and names.
            n (int, optional): The number of most similar tracks to return. Defaults to 5.

        Returns:
            Tuple[List[int], List[str], List[str]]: A tuple containing the indices, names, and captions of the n most similar tracks.
        """
        
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