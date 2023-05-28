import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

import openai
import os
import numpy as np
import pandas as pd

from chat_bot import HardCodedBouncerBot, ReceptionChatBot, ReceptionSummarizerBot, RecommenderChatBot
from search import SimpleCosineSimilarity

# Read OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

N_RSEARCH_RESULTS = 5

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
bouncer = HardCodedBouncerBot(stop_phrases=["start search"])
summarizer = ReceptionSummarizerBot()

# Instantiate the Dash app
external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css",
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the layout of the app
app.layout = html.Div(
    [
        html.H1("Music Discovery Chatbot", className="app-title"),
        html.Div(id="conversation", className="conversation-container"),
        html.Div(
            [
                dcc.Input(
                    id="user-input",
                    type="text",
                    placeholder="Enter your message...",
                    className="user-input-container"
                ),
                html.Button("Send", id="send-button", n_clicks=0, className="send-button"),
            ],
            className="input-container"
        ),
    ],
    className="app-container"
)

# Store conversation history
conversation_history = []

# Define the callback function
@app.callback(
    [Output("conversation", "children"), Output("user-input", "value")],
    [Input("send-button", "n_clicks")],
    [State("user-input", "value")]
)
def handle_user_interaction(n_clicks, user_input):
    if user_input:
        receptionist.messages.append({"role": "user", "content": user_input})
        # Check if conversation is done
        bouncer.read_conversation(receptionist.messages)
        reception_job_done = bouncer.is_job_done()

        if not reception_job_done:
            # Get response from chat bot
            response = receptionist.get_response()
            # Append user input and assistant response to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})

            return [render_message(message) for message in conversation_history], ""
        else:
            # Get summary
            summarizer.read_conversation(receptionist.messages)
            summary = summarizer.summarize()

            # Do search
            indices, names, captions = search_algo.find_similar(summary, n=N_RSEARCH_RESULTS)

            # Instantiate recommender
            recommender = RecommenderChatBot(
                names=names,
                descriptions=captions,
                user_input=summary
            )

            response = recommender.get_response()

            # Append user input, assistant response, and summary to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})

            return [render_message(message) for message in conversation_history], ""

    return "", ""

def render_message(message):
    role = message["role"]
    content = message["content"]

    message_box_style = {
        "border": "1px solid #ddd",
        "border-radius": "5px",
        "padding": "10px",
        "margin-bottom": "10px"
    }

    if role == "user":
        return html.Div(
            [
                html.Div(
                    [
                        html.Span("User: ", className="message-role", style={"font-weight": "bold"}),
                        html.Span(content, className="message-content")
                    ],
                    className="message-text",
                    style=message_box_style
                )
            ],
            className="message-box user-message"
        )
    elif role == "assistant":
        return html.Div(
            [
                html.Div(
                    [
                        html.Span("Assistant: ", className="message-role", style={"font-weight": "bold"}),
                        html.Span(content, className="message-content")
                    ],
                    className="message-text",
                    style=message_box_style
                )
            ],
            className="message-box assistant-message"
        )
    elif role == "summary":
        return html.Div(
            [
                html.Div(
                    [
                        html.Span("Summary: ", className="message-role"),
                        html.Span(content, className="message-content")
                    ],
                    className="message-text",
                    style=message_box_style
                )
            ],
            className="message-box summary-message"
        )



if __name__ == "__main__":
    app.run_server(debug=True)