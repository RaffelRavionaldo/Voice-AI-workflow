import os
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional, Dict, Any, Tuple
import numpy as np

load_dotenv()

class ConversationalRAG:
    def __init__(self, model_source="openai", embedding_source="openai", chroma_dir="chroma_store"):
        """
        Initialize Conversational RAG System
        
        Args:
            model_source: "openai" or "hf" (HuggingFace)
            embedding_source: "openai" or "hf"
            chroma_dir: Directory to store/load Chroma vectorstore
        """
        self.model_source = model_source
        self.embedding_source = embedding_source
        self.chroma_dir = f"{chroma_dir}_{embedding_source}"
        self.llm = self._load_llm()
        self.embeddings = self._load_embeddings()
        self.db = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        self.qa_chain = None

    def _load_embeddings(self):
        """Load embeddings model"""
        if self.embedding_source == "openai":
            return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def _load_llm(self):
        """Load LLM"""
        if self.model_source == "openai":
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # Load HuggingFace model with quantization
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto", 
            offload_buffers=True
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )
        
        hf_pipeline = HuggingFacePipeline(pipeline=pipe)
        
        return hf_pipeline

    def initialize_vectorstore(self, data_dir="data", chunk_size=1000, chunk_overlap=200):
        """
        Initialize or load vectorstore from documents
        
        Args:
            data_dir: Directory containing text files
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        if os.path.exists(self.chroma_dir):
            self.db = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embeddings
            )
        else:
            # Load and process documents
            loader = DirectoryLoader(data_dir, glob="**/*.txt")
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create and persist vectorstore
            self.db = Chroma.from_documents(
                chunks,
                self.embeddings,
                persist_directory=self.chroma_dir
            )
            self.db.persist()
        
        # Initialize QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.db.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )

    def _calculate_relevance(self, query: str, document: Document) -> float:
        query_embedding = self.embeddings.embed_query(query)
        doc_embedding = self.embeddings.embed_query(document.page_content)
        return np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
    
    def _retrieve_with_threshold(self, query: str, k: int = 3, threshold: float = 0.6) -> Tuple[list, bool]:
        docs = self.db.similarity_search(query, k=k)
        relevant_docs = []
        for doc in docs:
            score = self._calculate_relevance(query, doc)
            if score >= threshold:
                relevant_docs.append(doc)
        return relevant_docs, len(relevant_docs) > 0

    def chat(self, question, default_message="Sorry, i can't answer it. please booking a call with our team"):
        """
        Process a question and return the answer
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("Vectorstore not initialized. Call initialize_vectorstore() first.")
        
        result = self.qa_chain({"question": question})

        # relevant_docs, is_relevant = self._retrieve_with_threshold(question)

        return {
            "answer": result["answer"],
            "sources": result["source_documents"]
        }

    def clear_memory(self):
        """Clear conversation history"""
        self.memory.clear()