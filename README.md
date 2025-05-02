# Voice-AI-workflow

Create this code with Python 3.11 version

## App Flow: 

> User records the question --> Whisper generates text from user speech --> sends it to Conversational RAG --> checks if the chroma vector database is created or not, if not, makes the chroma database with embedding models dependent on your Chat models (openAI or hugging face) --> answers user question --> sends the answer to TTS

## Tools used :
1. Langchain : For conversational RAG
2. Chroma : For the vector database
3. OpenAI : as a chat model (optional, you can completely just use HuggingFace)
4. Hugginface :
  - Mistral 7B chat as a chat model
  - Whisper as Speech-To-Text models
  - Bark as Text-To-Speech models

## Steps to install and run this repo : 
1. Install the library and packages we need (for pytorch, better you exclude it and install the version that supports your device)
```
   pip install -r requirements.txt
```

2. Store your OpenAI API and Hugging Face tokens in .envs and store your data (for now, only support .txt) to folders call data

3. (Optional) Run Test code (test_conv_rag.py and test_tts_sst.py)
4. Use this syntax to run the app.py
```
streamlit run app.py
```

Steps 3 and 4 will create another folder to save our Chroma vector database locally.

5. By default, we use Huggingface models, but if you plan to use OpenAI in conversational RAG, change this code in app.py : 

![image](https://github.com/user-attachments/assets/85c78a73-1d2b-4975-8d4c-1fd743e45445)

to 
```
model_source="openai",
embedding_source="openai",
```

6. By default, I use quantization for loading the Mistral model from Huggingface. You can delete it and change the models with what you like (deepseek, llama, etc), but I recommend using chat models, not base models.

 ## Next step: 
 1. Include machine translation to support any languages (right now, just supports English)
 2. Enable question filtering (currently it is already in the conversational RAG code, but I turned it off because it is less effective, for example, the first question is what is Alara?. and the code has answered it, then the second question is how to use it?. Then my current filtering will ignore that question.)
