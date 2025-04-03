# Lecture 1: Introduction to AI  
**Key Concepts:**  
1. **Machine Learning vs AI**  
   - AI: Systems mimicking human intelligence  
   - ML: Subset using statistical learning (e.g., classifiers)  

2. **Neural Networks**  
   - Layers: Input → Hidden → Output  
   - Activation: ReLU (Rectified Linear Unit) formula:  
     ```math  
     f(x) = max(0, x)  
     ```  

3. **Training Process**  
   - Loss Function: Cross-entropy for classification  
   - Backpropagation: Chain rule applied to gradients  

# Lecture 2: LLMs and RAG  
1. **Transformer Architecture**  
   - Self-attention mechanism  
   - Query/Key/Value vectors  

2. **Retrieval-Augmented Generation**  
   - Steps:  
     1. Retrieve relevant documents  
     2. Inject context into LLM prompt  
   - Benefits: Reduces hallucinations  

3. **Example Pipeline**  
   ```python  
   from langchain import RetrievalQA  
   qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db)  