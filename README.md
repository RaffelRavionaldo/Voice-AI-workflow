# Voice-AI-Agent

Create this code with Python 3.11 version

Tools used :
1. Langchain : For conversational RAG
2. Chroma : For the vector database
3. OpenAI : as a chat model (optional, you can completely just use HuggingFace)
4. Hugginface :
  - Mistral 7B chat as a chat model
  - Whisper as Speech-To-Text models
  - Bark as Text-To-Speech models

Steps to install and run this repo : 
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
5. By default, we use Huggingface models, but if you plan to use OpenAI in conversational RAG, change this code in app.py : 

![image](https://github.com/user-attachments/assets/85c78a73-1d2b-4975-8d4c-1fd743e45445)

to 
```
model_source="openai",
embedding_source="openai",
```

6. By default too, i use quantization for load mistral model, you can delete it and change the models with what you like (deepseek, llama or etc), but i recommend to use chat models, not base

App Flow: 

> User records the question --> Whisper generate text from user speech --> sends it to Conversational RAG --> checks if the chroma vector database is created or not, if not, create the chroma database --> answers user question --> sends the answer to TTS
