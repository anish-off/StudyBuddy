# ai/agents/qa_agent.py
from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from ai.chains.base_chain import BaseChain

class QAAgent(BaseChain):
    """
    Enhanced Q&A Agent with:
    - Document ingestion capability
    - Dual-path answering (RAG + general knowledge)
    """
    
    def __init__(self, persist_dir: str = "chroma_db"):
        super().__init__()
        self.persist_dir = persist_dir
        
        # Initialize ChromaDB
        self.vector_db = Chroma(
            embedding_function=self.embedder,
            persist_directory=persist_dir
        )
        
        # Setup retrievers
        self.retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3}
        )
        
        # Prompt templates
        self.general_prompt = ChatPromptTemplate.from_template("""
        Answer the question to the best of your ability.
        Question: {question}
        Answer:""")
        
        self.rag_prompt = ChatPromptTemplate.from_template("""
        Use these excerpts from our materials to answer:
        {context}
        
        Question: {question}
        If the context doesn't help, say "I don't see this in our materials, but generally..." 
        and provide a general answer.""")

    def ingest_documents(self, texts: List[str], metadata: Optional[List[Dict]] = None):
        """Add documents to the knowledge base"""
        if metadata is None:
            metadata = [{}] * len(texts)
        self.vector_db.add_texts(texts=texts, metadatas=metadata)
        self.vector_db.persist()
        print(f"✅ Ingested {len(texts)} documents")

    def should_use_rag(self, question: str) -> bool:
        """Determine if question should use RAG"""
        triggers = ["book", "text", "chapter", "page", "lecture", "note"]
        return any(trigger in question.lower() for trigger in triggers)

    def answer(self, question: str) -> str:
        """Smart answering with automatic fallback"""
        try:
            if self.should_use_rag(question):
                docs = self.retriever.invoke(question)
                if docs:
                    rag_chain = (
                        {"context": lambda _: docs, "question": RunnablePassthrough()}
                        | self.rag_prompt
                        | self.llm
                        | StrOutputParser()
                    )
                    return rag_chain.invoke(question)
            
            # Fallback to general knowledge
            general_chain = (
                {"question": RunnablePassthrough()}
                | self.general_prompt
                | self.llm
                | StrOutputParser()
            )
            return general_chain.invoke(question)
            
        except Exception as e:
            return f"⚠️ Error processing your question: {str(e)}"

# Test cases
if __name__ == "__main__":
    qa = QAAgent()
    
    # Test ingestion
    test_docs = [
        "The mitochondria is the powerhouse of the cell",
        "Photosynthesis occurs in chloroplasts",
        "DNA replication happens during S-phase"
    ]
    qa.ingest_documents(test_docs)
    
    # Test queries
    print(qa.answer("Where does photosynthesis occur?"))  # From documents
    print(qa.answer("Explain general relativity"))  # General knowledge