import openai
import os
import sys
import numpy as np
import pandas as pd

from chat_bot import ReceptionBouncerBot, ReceptionChatBot, ReceptionSummarizerBot, RecommenderChatBot
from search import SimpleCosineSimilarity

# Read openai api key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

N_RSEARCH_RESULTS = 5

if __name__ == "__main__":


    #################
    ## PREPARATION ##
    #################
    
    # Read data & embeddings
    df = pd.read_csv("data/musiccaps-public.csv")
    embeddings = np.load("embeddings/aggregated_embeddings.npy")
    
    # Instantiate search algo
    search_algo = SimpleCosineSimilarity()
    search_algo.read_database(
        embeddings=embeddings,
        captions=df["caption"].tolist(),
        track_names=df["ytid"].tolist()
    )
    
    # Instantiate chat bots
    receptionist = ReceptionChatBot()
    bouncer = ReceptionBouncerBot()
    summarizer = ReceptionSummarizerBot()
    
    
    ##################
    ## CONVERSATION ##
    ##################
    
    while True:
        
        
        ###############
        ## RECEPTION ##
        ###############

        while True:
            
            # Get user msg
            receptionist.get_user_input()
            
            # Check if conversation is done
            bouncer.read_conversation(receptionist.messages)
            reception_job_done = bouncer.is_job_done()
            if reception_job_done:
                break
            
            # Get response from chat bot
            print(f"Assistant: {receptionist.get_response()}")
            
            
        print("\nConversation closed by bouncer.")

        ############
        ## SEARCH ##
        ############
        
        # Get summary
        summarizer.read_conversation(receptionist.messages)
        summary = summarizer.summarize()
        print("\nSummary:", summary)
        
        
        # Do search
        indices, names, captions = search_algo.find_similar(summary, n=N_RSEARCH_RESULTS)
        print("Search done. Starting conversation with recommender.")
        
        
        ####################
        ## RECOMMENDATION ##
        ####################
        
        # Instantiate recommender
        recommender = RecommenderChatBot(
            names=names,
            descriptions=captions,
            user_input=summary
        )
        
        print(f"\nAssistant: {recommender.get_response()}\n")

        while True:
            
            # Get user msg
            recommender.get_user_input()
            
            # Get response from chat bot
            print(f"\nAssistant: {recommender.get_response()}\n")
    