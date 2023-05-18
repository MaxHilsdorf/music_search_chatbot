# Music Search Chatbot

A ChatGPT-based chatbot that recommends music to the user based on their stated interests.

## Main Features
* Obtain and discuss music recommendations through a natural conversation.
* Ask follow-up questions or change your mind about what music you want.
* The chatbot searches a predefined and exchangable music database.
* If no music fits the request, the chatbot communicates that to the user.

## Conversation Flow
1. "Receptionist" bot helps user to describe what music they are looking for.
2. "Bouncer" bot closes the conversation when all has been said.
3. "Summarizer" bot summarizes the user request.
4. "Search" bot searches the music database and identifies the best fits.
5. "Recommender" bot presents recommendations and discusses them with the user.

## Setup
1. Clone the repository:
   ```shell
   $ git clone https://github.com/MaxHilsdorf/music_search_chatbot
   ```

2. Navigate into local repository:
    ```shell
    $ cd music_search_chatbot
    ```

2. Install the dependencies:
   ```shell
   $ pip install -r requirements.txt
   ```

3. Set your OpenAI API key as environment variable:
    ```
    $ export OPENAI_API_KEY="your_api_key"
    ```

## How to Run

1. Switch to src directory
   ```shell
   $ cd src
   ```

2. Run main script
   ```shell
   $ python main.py
   ```

3. Talk to the chat bot!

## Music Database
By default, the [MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps) dataset is implemented for search. The recommendations are given as YouTube IDs (e.g. "65KYS3lIRII") which can be accessed with ```www.youtube.com/watch?v=65KYS3lIRII```.

Any other database can be implemented as long as you have a local dataset with the following information:
* a unique name for each track
* a caption (description) for each track

How you construct the names (song title, url, ...) and the descriptions (full-text, list of tags, ...) is up to you. This tools works solely with text information and no audio signals are processed. To implement a new database, make the following changes to the repository:
* Adjust the beginning of ```src/compute_embeddings.py``` to fit your dataset.
* Run ```src/compute_embeddings.py``` to overwrite ```src/aggregated_embeddings.py```.
* Adjust ```src/main.py``` "PREPARATION" section to fit your dataset.

## License

The MIT License (MIT)

Copyright (c) 2023 Max Hilsdorf

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.