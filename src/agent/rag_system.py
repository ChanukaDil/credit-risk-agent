

"""
RAG System - Retrieval Augmented Generation
Handles document storage, embedding, and retrieval for the credit risk agent
"""

import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader
)

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    RAG system using ChromaDB and sentence-transformers
    Manages knowledge base for credit risk policies and regulations
    """
    
    def __init__(
        self,
        knowledge_base_path: str = "knowledge_base",
        vector_db_path: str = "models/vector_db/chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5
    ):
        """
        Initialize RAG system
        
        Args:
            knowledge_base_path: Path to documents
            vector_db_path: Path to store vector database
            embedding_model: HuggingFace embedding model
            chunk_size: Text chunk size for splitting
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.vector_db_path = Path(vector_db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        logger.info(f"Initializing RAG system with model: {embedding_model}")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Load or create vector store
        if self.vector_db_path.exists():
            logger.info("Loading existing vector database")
            self.vectorstore = self._load_vectorstore()
        else:
            logger.info("Creating new vector database")
            self.vectorstore = None
    
    def build_knowledge_base(self, force_rebuild: bool = False) -> None:
        """
        Build or rebuild the knowledge base from documents
        
        Args:
            force_rebuild: Force rebuild even if DB exists
        """
        if self.vectorstore is not None and not force_rebuild:
            logger.info("Vector store already exists. Use force_rebuild=True to rebuild.")
            return
        
        logger.info(f"Building knowledge base from {self.knowledge_base_path}")
        
        # Load documents
        documents = self._load_documents()
        
        if not documents:
            logger.warning("No documents found to build knowledge base")
            return
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.vector_db_path)
        )
        
        logger.info(f"Vector database created at {self.vector_db_path}")
    
    def _load_documents(self) -> List[Document]:
        """Load all documents from knowledge base directory"""
        documents = []
        
        # Define document loaders for different file types
        loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
        }
        
        # Walk through knowledge base directory
        for category in self.knowledge_base_path.iterdir():
            if not category.is_dir():
                continue
            
            logger.info(f"Loading documents from {category.name}")
            
            for file_path in category.glob("*"):
                if file_path.suffix in loaders:
                    try:
                        loader_class = loaders[file_path.suffix]
                        loader = loader_class(str(file_path))
                        docs = loader.load()
                        
                        # Add metadata
                        for doc in docs:
                            doc.metadata.update({
                                'source': str(file_path),
                                'category': category.name,
                                'filename': file_path.name
                            })
                        
                        documents.extend(docs)
                        logger.info(f"  Loaded: {file_path.name}")
                        
                    except Exception as e:
                        logger.error(f"  Error loading {file_path}: {e}")
        
        return documents
    
    def _load_vectorstore(self) -> Chroma:
        """Load existing vector store"""
        return Chroma(
            persist_directory=str(self.vector_db_path),
            embedding_function=self.embeddings
        )
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve (default: self.top_k)
            filter_metadata: Filter by metadata (e.g., {'category': 'regulations'})
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        if self.vectorstore is None:
            logger.warning("Vector store not initialized. Building knowledge base...")
            self.build_knowledge_base()
        
        k = k or self.top_k
        
        # Retrieve with scores
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_metadata
        )
        
        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        
        return results
    
    def retrieve_by_category(
        self,
        query: str,
        category: str,
        k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents from specific category
        
        Args:
            query: Search query
            category: Category name (e.g., 'bank_policies', 'regulations')
            k: Number of documents to retrieve
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        return self.retrieve(
            query=query,
            k=k,
            filter_metadata={'category': category}
        )
    
    def format_retrieved_docs(
        self,
        results: List[Tuple[Document, float]],
        include_scores: bool = False
    ) -> str:
        """
        Format retrieved documents into a string
        
        Args:
            results: Retrieved documents with scores
            include_scores: Include similarity scores in output
            
        Returns:
            Formatted string
        """
        if not results:
            return "No relevant information found."
        
        formatted_parts = []
        
        for i, (doc, score) in enumerate(results, 1):
            content = doc.page_content
            source = doc.metadata.get('filename', 'Unknown')
            category = doc.metadata.get('category', 'Unknown')
            
            if include_scores:
                header = f"[{i}] {category}/{source} (Score: {score:.3f})"
            else:
                header = f"[{i}] {category}/{source}"
            
            formatted_parts.append(f"{header}\n{content}\n")
        
        return "\n".join(formatted_parts)
    
    def get_context_for_query(
        self,
        query: str,
        categories: Optional[List[str]] = None
    ) -> str:
        """
        Get formatted context for a query
        
        Args:
            query: User query
            categories: Filter by specific categories
            
        Returns:
            Formatted context string
        """
        if categories:
            # Retrieve from each category and combine
            all_results = []
            for category in categories:
                results = self.retrieve_by_category(query, category, k=2)
                all_results.extend(results)
        else:
            # Retrieve from all categories
            all_results = self.retrieve(query)
        
        # Sort by score and take top-k
        all_results.sort(key=lambda x: x[1], reverse=True)
        top_results = all_results[:self.top_k]
        
        return self.format_retrieved_docs(top_results, include_scores=False)
    
    def add_document(
        self,
        text: str,
        metadata: Dict[str, str]
    ) -> None:
        """
        Add a new document to the knowledge base
        
        Args:
            text: Document text
            metadata: Document metadata
        """
        if self.vectorstore is None:
            logger.error("Vector store not initialized")
            return
        
        # Split text
        doc = Document(page_content=text, metadata=metadata)
        chunks = self.text_splitter.split_documents([doc])
        
        # Add to vector store
        self.vectorstore.add_documents(chunks)
        logger.info(f"Added document: {metadata.get('filename', 'Unknown')}")
    
    def search_similar_cases(
        self,
        customer_profile: str,
        k: int = 3
    ) -> str:
        """
        Search for similar customer cases
        
        Args:
            customer_profile: Customer description
            k: Number of cases to retrieve
            
        Returns:
            Formatted similar cases
        """
        results = self.retrieve_by_category(
            query=customer_profile,
            category='case_studies',
            k=k
        )
        
        return self.format_retrieved_docs(results)


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem(
        knowledge_base_path="knowledge_base",
        vector_db_path="models/vector_db/chroma_db"
    )
    
    # Build knowledge base (first time only)
    rag.build_knowledge_base()
    
    # Test retrieval
    query = "What are the lending criteria for high-risk customers?"
    context = rag.get_context_for_query(query)
    
    print("Retrieved Context:")
    print(context)