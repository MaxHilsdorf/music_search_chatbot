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
        """
        Initializes the object. This method is an abstract method and must be
        implemented by subclasses. It does not take any parameters and does not
        return anything.
        """
        ...
    
    @abstractmethod
    def read_conversation(self, messages: List[dict]) -> None:
        """
        This abstract method defines the behavior for reading a conversation. It takes in a list of message dictionaries 
        'messages' and returns None. Subclasses must implement this method to define how the conversation should be read.

        :param messages: A list of message dictionaries.
        :type messages: List[dict]
        :return: None
        :rtype: None
        """
        ...
        
    @abstractmethod
    def is_job_done(self) -> bool:
        """
        Determines whether the job is done or not.

        :return: A boolean indicating whether the job is done or not.
        """
        ...

class ReceptionBouncerBot(BouncerBot):
    
    def __init__(self):
        """
        Initializes the ReceptionBouncerBot object with default values for its prompts. This chatbot is designed 
        to talk to users looking for music recommendations, and will close the conversation when either the user has 
        nothing more to say, the chat bot ends the conversation politely, or the chat bot tells the user he will 
        be directed to a recommender AI. This function takes no parameters and returns nothing.
        """
        
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
        """
        Reads a conversation by iterating over each message in the given list of messages.
        For each message, it sets the `prompt_mid` attribute of the instance to a string
        representation of the message.
        
        :param messages: A list of dictionaries representing the messages in the conversation.
        :type messages: List[dict]
        :return: None
        :rtype: None
        """
        
        for entry in messages:
            self.prompt_mid += f"\n{entry['role']}: {entry['content']}"
    
    def is_job_done(self) -> bool:
        """
        This function checks if a job is done by sending a prompt to the OpenAI 
        text completion API and waiting for a boolean response. It takes no 
        parameters but uses instance variables self.prompt_start, self.prompt_mid
        and self.prompt_end to form the prompt. It returns a boolean value True 
        if the response is "true", "yes" or "1" and False if the response is 
        "false", "no" or "0". If a boolean value cannot be derived from the 
        response text, the function retries up to 5 times before returning False.
        """

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
            
            
class HardCodedBouncerBot(BouncerBot):
    
    def __init__(self, stop_phrases: List[str]):
        """
        Initializes an instance of the HardCodedBouncerBot class.

        :param stop_phrases: A list of strings representing the stop phrases for the bot.
        :type stop_phrases: List[str]

        :return: None
        :rtype: None
        """
        
        self.name = "HardCodedBouncerBot"
        self.stop_phrases = stop_phrases
        self.final_message = ""
        
    def read_conversation(self, messages: List[dict]) -> None:
        """
        Reads a conversation and sets the final message as the 'content' of the last message in the conversation.

        :param messages: A list of dictionaries containing messages in a conversation. Each dictionary should have a 'content' key.
        :type messages: List[dict]
        :return: None
        """
        
        self.final_message = messages[-1]["content"]
        
    def is_job_done(self) -> bool:
        """
        Returns a boolean value indicating whether the job is done or not. The function searches for each phrase from the stop_phrases list in the final_message attribute and returns True if any of these phrases is found, otherwise False. 

        :return: A boolean value indicating whether the job is done or not.
        :rtype: bool
        """
        
        for phrase in self.stop_phrases:
            if phrase in self.final_message.lower():
                return True
        return False
        
            
#####################
## SUMMARIZER BOTS ##
#####################

class SummarizerBot(ABC):
    
    @abstractmethod
    def __init__(self):
        """
        Initializes the object. This is an abstract method and must be
        implemented by subclasses. It takes no parameters and returns nothing.
        """
        ...
    
    @abstractmethod
    def read_conversation(self, messages: List[dict]) -> None:
        """
        An abstract method that defines the structure of reading a conversation given a list of messages.
        :param messages: A list of dictionaries representing the messages in the conversation.
        :type messages: List[dict]
        :return: None
        :rtype: None
        """
        ...
        
    @abstractmethod
    def summarize(self, text: str) -> str:
        """
        This is an abstract method for summarizing text. It takes in a string parameter
        named `text` and returns a string. This method must be implemented by any class that
        inherits from the abstract class that contains it.

        :param text: A string representing the text to summarize.
        :type text: str
        :return: A string representing the summary of the input text.
        :rtype: str
        """
        ...

class ReceptionSummarizerBot(SummarizerBot):
    
    def __init__(self):
        """
        Initializes a ReceptionSummarizerBot object.
        This function does not take any parameters.

        Parameters:
        None

        Return:
        None
        """
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
        """
        Given a list of message dictionaries, reads the conversation by concatenating each message's role and content
        into a prompt_mid string attribute. 

        :param messages: A list of message dictionaries with keys 'role' and 'content'
        :type messages: List[dict]
        :return: None
        :rtype: None
        """
        
        for entry in messages:
            self.prompt_mid += f"\n{entry['role']}: {entry['content']}"
    
    def summarize(self) -> str:
        """
        Returns a summarized response using OpenAI's "text-davinci-003" model.
        Takes no parameters. Returns a string containing the generated response.
        """
        
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
        """
        Initializes the object. This is an abstract method and must be overridden by the subclass.
        """
        ...
    
    def get_user_input(self) -> str:
        """
        Returns a string representing the user input obtained through standard input. 
        This method also adds the user's input message to the messages list. 

        :return: a string representing the user's input
        :rtype: str
        """ 
        user_input = input("You: ")
        self.messages.append({"role": "user", "content": user_input})
        return user_input
    
    @abstractmethod
    def get_response(self) -> str:
        """
        This is an abstract method that should be implemented by subclasses. It returns a string.
        """
        ...
        
    @abstractmethod
    def job_done(self, bouncer: BouncerBot) -> bool:
        """
        This is an abstract method that should be implemented in a subclass. It takes a bouncer instance 
        as a parameter and returns a boolean value indicating whether the chatbot's job is already done or not. 
        
        :param bouncer: An instance of the BouncerBot class representing the bouncer object.
        :type bouncer: BouncerBot
        
        :return: A boolean value indicating whether the job is done or not.
        :rtype: bool
        """
        ...
        
class ReceptionChatBot(ChatBot):
    
    def __init__(self):
        """
        Initializes a new instance of the ReceptionBot class.

        Parameters:
        None.

        Returns:
        None.

        Description:
        Sets the name of the bot to "ReceptionBot" and initializes a system message that explains the bot's purpose to
        the user. The message follows a clear structure that consists of:
        1. repeating the user's request in a summarized way
        2. asking whether the user has anything to add
        3. telling the user to type 'start search' to start the search
        The message is then added to a list of messages that the bot can send to the user.
        """
        self.name = "ReceptionBot"
        self.system_msg = """
        You are a music discovery receptionist AI. The user tells you what music he is looking for. Your response follows a clear structure
        1. repeat the users request in a summarized way
        2. ask whether the user has anything to add
        3. tell the user to type 'start search' to start the search
        """
        
        self.messages = [{"role": "system", "content": f"{self.system_msg}"}]
        
    def get_response(self) -> str:
        """
        Returns a string generated by OpenAI's GPT-3.5-turbo model for the given messages.
        Parameter:
            self (object): Instance of the class.
        Returns:
            response_text (str): The response generated by the model.
        """
        
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
        """
        This function takes in a BouncerBot object and reads a conversation from the messages attribute.
        It then returns a boolean indicating whether the job is done or not.

        :param bouncer: A BouncerBot object used to read the conversation from messages.
        :type bouncer: BouncerBot
        :return: A boolean indicating whether the job is done or not.
        :rtype: bool
        """
        bouncer.read_conversation(self.messages)
        return bouncer.is_job_done()


class RecommenderChatBot(ChatBot):
    
    def __init__(self, names: List[str], descriptions: List[str], user_input: str, max_caption_length: int = 500):
        """
        Initializes a RecommenderBot instance with a given list of music names and descriptions, a user input string, and a maximum caption length.
        :param names: A list of strings representing music names.
        :param descriptions: A list of strings representing music descriptions.
        :param user_input: A string representing the user's request.
        :param max_caption_length: An optional integer representing the maximum length of the music descriptions. Defaults to 500.
        """
        
        self.name = "RecommenderBot"
        self.max_caption_length = max_caption_length
        self.system_msg = """
        A search algorithm send you some music that the user may like. You are an assistant that recommends music to the user based on their request.
        The user will start the conversation by repeating his request. Be brief and stick exclusively to the exact search results.
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
        """
        Retrieves a response from the GPT-3.5-Turbo model using the provided messages as context.
        :return: A string representing the response generated by the model.
        """
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
        """
        Checks if a job is done by the RecommenderBot.

        :param bouncer: a BouncerBot object
        :type bouncer: BouncerBot
        :return: a boolean indicating if the job is done or not
        :rtype: bool
        """
        
        raise NotImplementedError("RecommenderBot has not implemented job_done yet.")