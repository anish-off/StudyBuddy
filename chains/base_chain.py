# ai/chains/base_chain.py
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from ai.tools.ollama_manager import OllamaManager

# Load environment variables
load_dotenv()

class BaseChain:
    """
    Core LLM and embeddings setup with Ollama.
    Handles model initialization and basic prompt templating.
    """
    
    def __init__(self):
        # Verify required models are available
        self._verify_models()
        
        # Initialize LLM with custom settings
        self.llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "mistral"),
            temperature=0.7,  # Balance creativity/focus
            top_k=50,        # Broader token sampling
            num_ctx=1024,  # Smaller context = faster 
            stop=["<|endoftext|>"]  # Custom stop sequence
        )
        
        # Initialize embeddings
        self.embedder = OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDINGS", "nomic-embed-text")
        )
        
        # Base prompt template for all chains
        self.base_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI Study Buddy. Respond helpfully and concisely."),
            ("human", "{input}")
        ])

    def _verify_models(self):
        """Ensure required Ollama models are pulled and available"""
        OllamaManager().pull_model(os.getenv("OLLAMA_MODEL", "mistral"))
        OllamaManager().pull_model(os.getenv("OLLAMA_EMBEDDINGS", "nomic-embed-text"))

    def generate(self, input_text: str, **kwargs) -> str:
        """Base generation method with templating"""
        chain = self.base_prompt | self.llm
        return chain.invoke({"input": input_text, **kwargs})

# Quick test
if __name__ == "__main__":
    chain = BaseChain()
    test_query = "Explain quantum entanglement to a 5th grader"
    print(chain.generate(test_query))