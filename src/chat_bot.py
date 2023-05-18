import openai
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List

openai.api_key = os.getenv("OPENAI_API_KEY")


##################
## BOUNCER BOTS ##
##################

class BouncerBot(ABC):
    
    @abstractmethod
    def __init__(self):
        ...
    
    @abstractmethod
    def read_conversation(self, messages: List[dict]) -> None:
        ...
        
    @abstractmethod
    def is_job_done(self) -> bool:
        ...
        
class ReceptionBouncerBot(BouncerBot):
    
    def __init__(self):
        self.name = "ReceptionBouncerBot"
        self.prompt_start = """
        A chatbot is talking to a user looking for music recommendations. The conversation should be closed when either:
        - the user has nothing more to say.
        - the chat bot ends the conversation, politely.
        - the chat bot tells the user he will be directed to a recommender AI.
        """
        self.prompt_end = """
        Should the conversation be closed? Respond with 'Yes" or "No".
        """
        self.prompt_mid = ""

    def read_conversation(self, messages: List[dict]) -> None:
        for entry in messages:
            self.prompt_mid = f"\n{entry['role']}: {entry['content']}"
    
    def is_job_done(self) -> bool:

        i = 0
        while True:
            response = openai.Completion.create(
                model = "text-davinci-003",
                prompt = f"{self.prompt_start}\n{self.prompt_mid}\n{self.prompt_end}\n",
                max_tokens = 10,
                temperature=0.2
            )
            response_text = response["choices"][0]["text"].lower()
            
            if any([answer in response_text for answer in ["true", "yes", "1"]]):
                return True
            elif any([answer in response_text for answer in ["false", "no", "0"]]):
                return False
            else:
                print("Could not derive bool from", response_text)
                i += 1
                if i > 5:
                    print("Too many attempts. Returning False.")
                    return False
                continue
            
            
#####################
## SUMMARIZER BOTS ##
#####################

class SummarizerBot(ABC):
    
    @abstractmethod
    def __init__(self):
        ...
    
    @abstractmethod
    def read_conversation(self, messages: List[dict]) -> None:
        ...
        
    @abstractmethod
    def summarize(self, text: str) -> str:
        ...

class ReceptionSummarizerBot(SummarizerBot):
    
    def __init__(self):
        self.name = "ReceptionSummarizerBot"
        self.prompt_start = """
        The following is a conversation between a user looking for music and a chatbot
        """
        self.prompt_end = """
        Briefly summarize the kind of music the user is looking for.
        The user is looking for the following kind of music:
        
        """
        self.prompt_mid = ""
        
    def read_conversation(self, messages: List[dict]) -> None:
        for entry in messages:
            self.prompt_mid += f"\n{entry['role']}: {entry['content']}"
    
    def summarize(self) -> str:
        response = openai.Completion.create(
            model = "text-davinci-003",
            prompt = f"{self.prompt_start}\n{self.prompt_mid}\n{self.prompt_end}\n",
            temperature = 0.5,
            max_tokens = 100
        )
        response_text = response["choices"][0]["text"]
        return response_text
        

###############
## CHAT BOTS ##
###############

class ChatBot(ABC):
    
    @abstractmethod
    def __init__(self):
        ...
    
    def get_user_input(self) -> str:
        user_input = input("You: ")
        self.messages.append({"role": "user", "content": user_input})
        return user_input
    
    @abstractmethod
    def get_response(self) -> str:
        ...
        
    @abstractmethod
    def job_done(self, bouncer: BouncerBot) -> bool:
        ...
    
    @abstractmethod
    def close_conversation(self) -> None:
        ...
        
class ReceptionChatBot(ChatBot):
    
    def __init__(self):
        self.name = "ReceptionBot"
        self.system_msg = """
        You are a music discovery receptionist AI. Your listen to the user and ask him if he has anything else to say.
        The user will ask you to find or recommend music. You never do that, but you act as if you can.
        Another AI will take over and do that once you have collected the users thoughts.
        You are not very talkative bring the conversation to an end as soon as possible.
        """
        self.messages = [{"role": "system", "content": f"{self.system_msg}"}]
        
    def get_response(self) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            temperature = 0.7,
            max_tokens = 100
            )
        response_text = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": response_text})
        return response_text
    
    def job_done(self, bouncer: BouncerBot) -> bool:
        bouncer.read_conversation(self.messages)
        return bouncer.is_job_done()
    
    def close_conversation(self) -> None:
        ...


class RecommenderChatBot(ChatBot):
    
    def __init__(self, names: List[str], descriptions: List[str], user_input: str, max_caption_length: int = 500):
        self.name = "RecommenderBot"
        self.max_caption_length = max_caption_length
        self.system_msg = """
        A search algorithm provided you with music that the user may like. You are an assistant that recommends music to the user based on their request.
        The user will start the conversation by repeating his request. Be brief.
        Search results:
        
        """
        for name, description in zip(names, descriptions):
            # Truncate description if needed
            if len(description) > self.max_caption_length:
                description = description[:self.max_caption_length] + "..."
            self.system_msg += f"\n{name}: {description}"

        self.messages = [
            {"role": "system", "content": f"{self.system_msg}"},
            {"role": "user", "content": f"I am looking for the following kind of music: {user_input}"}
            ]
        
    def get_response(self) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            temperature = 0.7,
            max_tokens = 250
            )
        response_text = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": response_text})
        return response_text
    
    def job_done(self, bouncer: BouncerBot) -> bool:
        raise NotImplementedError("RecommenderBot has not implemented job_done yet.")
    
    def close_conversation(self) -> None:
        raise NotImplementedError("RecommenderBot has not implemented close_conversation yet.")