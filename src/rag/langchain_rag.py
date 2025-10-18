
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# LangChain imports
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Transformers for LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from src.config import config

logger = logging.getLogger(__name__)


class LangChainRAGSystem:
    """
    RAG system using LangChain for credit risk assessment
    
    Integrates:
    - Document loading (PyPDFLoader)
    - Text splitting (RecursiveCharacterTextSplitter)
    - Embeddings (HuggingFaceEmbeddings)
    - Vector store (FAISS)
    - LLM (HuggingFacePipeline)
    - RAG chain (RetrievalQA)
    """
    
    def __init__(
        self,
        policies_dir: str = None,
        vector_db_path: str = None,
        load_llm: bool = True
    ):
        """
        Initialize LangChain RAG system
        
        Args:
            policies_dir: Directory containing bank policy PDFs
            vector_db_path: Path to save/load vector database
            load_llm: Whether to load LLM (set False for faster testing)
        """
        self.policies_dir = policies_dir or config.bank_policy.policies_dir
        self.vector_db_path = vector_db_path or str(Path(config.rag.vector_db_path).parent)
        
        logger.info("🦜 Initializing LangChain RAG System...")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.rag.embedding_model,
            model_kwargs={'device': config.llm.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✅ Embeddings initialized")
        
        # Initialize vector store
        self.vectorstore = None
        self.retriever = None
        
        # Initialize LLM (optional)
        self.llm = None
        if load_llm:
            self.llm = self._initialize_llm()
        
        # Initialize RAG chain
        self.qa_chain = None
    
    def _initialize_llm(self) -> HuggingFacePipeline:
        """Initialize Llama 3.2 with LangChain wrapper"""
        logger.info("🤖 Loading Llama 3.2 3B...")
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                config.llm.model_name,
                cache_dir=config.llm.cache_dir
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                config.llm.model_name,
                torch_dtype=torch.float16 if config.llm.device == "cuda" else torch.float32,
                device_map="auto" if config.llm.device == "cuda" else None,
                cache_dir=config.llm.cache_dir,
                low_cpu_mem_usage=True
            )
            
            if config.llm.device == "cpu":
                model = model.to(config.llm.device)
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                top_p=config.llm.top_p,
                do_sample=True
            )
            
            # Wrap in LangChain
            llm = HuggingFacePipeline(pipeline=pipe)
            
            logger.info("✅ LLM initialized")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return None
    
    def load_and_process_documents(self) -> List[Document]:
        """
        Load bank policy PDFs and process them
        
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"📄 Loading documents from {self.policies_dir}...")
        
        documents = []
        
        # Define loan type mapping
        loan_type_mapping = {
            'professional_loan.pdf': 'professional',
            'vehicle_loan.pdf': 'vehicle',
            'housing_loan.pdf': 'housing'
        }
        
        for filename, loan_type in loan_type_mapping.items():
            filepath = Path(self.policies_dir) / filename
            
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                continue
            
            try:
                # Load PDF
                loader = PyPDFLoader(str(filepath))
                pages = loader.load()
                
                # Add metadata
                for page in pages:
                    page.metadata['loan_type'] = loan_type
                    page.metadata['source'] = filename
                    page.metadata['page'] = page.metadata.get('page', 0)
                
                documents.extend(pages)
                logger.info(f"  ✓ Loaded {filename}: {len(pages)} pages")
                
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
        
        if not documents:
            raise ValueError("No documents were loaded successfully")
        
        logger.info(f"✅ Loaded {len(documents)} pages from {len(loan_type_mapping)} PDFs")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using LangChain
        
        Args:
            documents: List of Document objects
        
        Returns:
            List of chunked Document objects
        """
        logger.info("✂️ Splitting documents into chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        
        logger.info(f"✅ Created {len(splits)} chunks")
        return splits
    
    def build_vectorstore(self, documents: List[Document] = None):
        """
        Build FAISS vector store from documents
        
        Args:
            documents: Document chunks (if None, loads from disk)
        """
        if documents is None:
            documents = self.load_and_process_documents()
            documents = self.split_documents(documents)
        
        logger.info("🔨 Building FAISS vector store...")
        
        # Create FAISS vectorstore
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.rag.top_k}
        )
        
        logger.info(f"✅ Vector store built with {len(documents)} documents")
    
    def save_vectorstore(self):
        """Save vector store to disk"""
        if self.vectorstore is None:
            raise ValueError("No vector store to save")
        
        save_path = Path(self.vector_db_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.vectorstore.save_local(str(save_path))
        logger.info(f"💾 Vector store saved to {save_path}")
    
    def load_vectorstore(self):
        """Load vector store from disk"""
        load_path = Path(self.vector_db_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        logger.info(f"📂 Loading vector store from {load_path}...")
        
        self.vectorstore = FAISS.load_local(
            str(load_path),
            self.embeddings
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.rag.top_k}
        )
        
        logger.info("✅ Vector store loaded")
    
    def create_qa_chain(self):
        """
        Create RetrievalQA chain
        
        Combines retriever + LLM for question answering
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Build or load vector store first.")
        
        if self.llm is None:
            logger.warning("LLM not available. QA chain will not work.")
            return
        
        # Create prompt template
        prompt_template = """You are a credit risk analyst. Use the following bank policy documents to answer the question.

Context from bank policies:
{context}

Question: {question}

Provide a concise, professional answer based on the policies. If the answer is not in the context, say "I don't have that information in the policies."

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("✅ QA chain created")
    
    def query(
        self,
        question: str,
        loan_type: Optional[str] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: Question to answer
            loan_type: Optional filter by loan type
            return_sources: Whether to return source documents
        
        Returns:
            Dictionary with answer and sources
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized")
        
        # Retrieve relevant documents
        if loan_type:
            # Filter by loan type
            retriever_kwargs = {
                "k": config.rag.top_k,
                "filter": {"loan_type": loan_type}
            }
            docs = self.vectorstore.similarity_search(question, **retriever_kwargs)
        else:
            docs = self.retriever.get_relevant_documents(question)
        
        result = {
            'question': question,
            'documents': []
        }
        
        # Format documents
        for doc in docs:
            result['documents'].append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        # If QA chain available, generate answer
        if self.qa_chain:
            try:
                response = self.qa_chain({"query": question})
                result['answer'] = response['result']
                
                if return_sources and 'source_documents' in response:
                    result['source_documents'] = [
                        {
                            'content': doc.page_content,
                            'metadata': doc.metadata
                        }
                        for doc in response['source_documents']
                    ]
            except Exception as e:
                logger.error(f"QA chain failed: {e}")
                result['answer'] = None
        
        return result
    
    def check_policy_compliance(
        self,
        loan_application: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if loan application meets policy requirements
        
        Args:
            loan_application: Application data
        
        Returns:
            Compliance check results
        """
        loan_type = loan_application.get('loan_type', 'professional')
        
        # Build query
        query = f"""For a {loan_type} loan application with:
- Amount: ${loan_application.get('loan_amount', 0):,.2f}
- Income: ${loan_application.get('income', 0):,.2f}
- Credit Score: {loan_application.get('credit_score', 0)}
- Age: {loan_application.get('age', 0)}
- Debt-to-Income: {loan_application.get('debt_to_income_ratio', 0):.1%}

What are the eligibility requirements and is this application likely to be compliant?"""
        
        # Query RAG system
        result = self.query(query, loan_type=loan_type)
        
        # Rule-based compliance checks
        policy_config = config.bank_policy.loan_types.get(loan_type, {})
        
        age = loan_application.get('age', 0)
        dti = loan_application.get('debt_to_income_ratio', 0)
        credit_score = loan_application.get('credit_score', 0)
        
        age_range = policy_config.get('age_range', (18, 60))
        dti_threshold = policy_config.get('dti_threshold', 0.60)
        min_credit = policy_config.get('min_credit_score', 600)
        
        violations = []
        
        if not (age_range[0] <= age <= age_range[1]):
            violations.append(f"Age {age} outside acceptable range {age_range}")
        
        if dti > dti_threshold:
            violations.append(f"DTI {dti:.1%} exceeds maximum {dti_threshold:.1%}")
        
        if credit_score < min_credit:
            violations.append(f"Credit score {credit_score} below minimum {min_credit}")
        
        compliance = {
            'overall_compliant': len(violations) == 0,
            'violations': violations,
            'age_compliant': age_range[0] <= age <= age_range[1],
            'dti_compliant': dti <= dti_threshold,
            'credit_score_compliant': credit_score >= min_credit,
            'retrieved_policies': result['documents'][:3],
            'rag_answer': result.get('answer')
        }
        
        return compliance


# Convenience function
def build_langchain_rag_system(
    policies_dir: str = None,
    output_dir: str = None,
    load_llm: bool = False
) -> LangChainRAGSystem:
    """
    Build complete LangChain RAG system
    
    Args:
        policies_dir: Directory with bank policy PDFs
        output_dir: Output directory for vector store
        load_llm: Whether to load LLM
    
    Returns:
        Configured RAG system
    """
    logger.info("="*70)
    logger.info("BUILDING LANGCHAIN RAG SYSTEM")
    logger.info("="*70)
    
    # Initialize system
    rag = LangChainRAGSystem(
        policies_dir=policies_dir,
        vector_db_path=output_dir,
        load_llm=load_llm
    )
    
    # Load and process documents
    documents = rag.load_and_process_documents()
    splits = rag.split_documents(documents)
    
    # Build vector store
    rag.build_vectorstore(documents=splits)
    
    # Save
    rag.save_vectorstore()
    
    # Create QA chain if LLM available
    if rag.llm:
        rag.create_qa_chain()
    
    logger.info("="*70)
    logger.info("✅ LANGCHAIN RAG SYSTEM BUILT SUCCESSFULLY")
    logger.info("="*70)
    
    return rag


# Example usage
if __name__ == "__main__":
    
    # Build system
    rag = build_langchain_rag_system(
        policies_dir="data/raw/bank_policies",
        output_dir="models/vector_db",
        load_llm=False  # Set True to include LLM
    )
    
    # Test queries
    print("\n" + "="*70)
    print("TESTING RAG SYSTEM")
    print("="*70)
    
    test_queries = [
        ("What are the eligibility criteria for professional loans?", "professional"),
        ("What is the maximum tenure for vehicle loans?", "vehicle"),
        ("What are the interest rates for housing loans?", "housing")
    ]
    
    for query, loan_type in test_queries:
        print(f"\nQuery: {query}")
        print(f"Loan Type: {loan_type}")
        
        result = rag.query(query, loan_type=loan_type)
        
        print(f"\nFound {len(result['documents'])} relevant documents:")
        for i, doc in enumerate(result['documents'][:2]):
            print(f"\n{i+1}. {doc['content'][:150]}...")
            print(f"   Source: {doc['metadata']['source']}")
    
    print("\n" + "="*70)
    print("✅ Testing complete!")
    print("="*70)