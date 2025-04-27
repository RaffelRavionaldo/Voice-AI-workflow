from conversational_rag import ConversationalRAG
import re

def extract_helpful_answer(response_str):
    parts = response_str.split("Helpful Answer: ")
    if len(parts) > 1:
        return parts[-1].split("\n")[0].strip()
    return response_str  # Fallback

# Initialize with OpenAI
rag = ConversationalRAG(model_source="openai", embedding_source="openai")

# Or with HuggingFace
# rag = ConversationalRAG(model_source="hf", embedding_source="hf")

rag.initialize_vectorstore(data_dir="data")

response = rag.chat("What is Alara?")
print(extract_helpful_answer(response['answer']))
print("\n -----------------------------------------")

response = rag.chat("1+1 = ")
print(extract_helpful_answer(response['answer']))
print("\n -----------------------------------------")

response = rag.chat("who create you?")
print(extract_helpful_answer(response['answer']))
print("\n -----------------------------------------")

response = rag.chat("how can i use you?")
print(extract_helpful_answer(response['answer']))
print("\n -----------------------------------------")