"""
Setup Script: Initialize Vector Database for RAG System
Creates and populates ChromaDB with knowledge base documents
"""

import sys
from pathlib import Path
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agent.rag_system import RAGSystem
from src.utils.document_loader import create_sample_knowledge_base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main setup function"""
    
    print("\n" + "="*70)
    print("Credit Risk Agent - Vector Database Setup")
    print("="*70 + "\n")
    
    # Step 1: Create sample knowledge base (if not exists)
    knowledge_base_path = Path("knowledge_base")
    if not knowledge_base_path.exists():
        print("📄 Creating sample knowledge base...")
        create_sample_knowledge_base()
        print("✅ Sample knowledge base created\n")
    else:
        print("✅ Knowledge base directory found\n")
    
    # Step 2: Initialize RAG system
    print("🔧 Initializing RAG system...")
    rag = RAGSystem(
        knowledge_base_path=str(knowledge_base_path),
        vector_db_path="models/vector_db/chroma_db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ RAG system initialized\n")
    
    # Step 3: Build vector database
    print("🔨 Building vector database...")
    print("   This may take a few minutes...\n")
    
    try:
        rag.build_knowledge_base(force_rebuild=True)
        print("\n✅ Vector database built successfully!")
        
    except Exception as e:
        print(f"\n❌ Error building vector database: {e}")
        return False
    
    # Step 4: Test retrieval
    print("\n" + "="*70)
    print("🧪 Testing Retrieval")
    print("="*70 + "\n")
    
    test_queries = [
        "What are the lending criteria for high-risk customers?",
        "What documents are required for loan approval?",
        "How should I handle customer defaults?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 70)
        
        try:
            results = rag.retrieve(query, k=2)
            
            if results:
                for i, (doc, score) in enumerate(results, 1):
                    print(f"[{i}] Score: {score:.3f}")
                    print(f"    Source: {doc.metadata.get('filename', 'Unknown')}")
                    print(f"    Content: {doc.page_content[:100]}...")
            else:
                print("No results found")
                
        except Exception as e:
            print(f"Error: {e}")
        
        print()
    
    # Final summary
    print("="*70)
    print("✅ Setup Complete!")
    print("="*70)
    print("\nYour vector database is ready at: models/vector_db/chroma_db")
    print("\nNext steps:")
    print("1. Run the agent: python src/agent/credit_risk_agent.py")
    print("2. Test the API: uvicorn src.api.fastapi_app:app --reload")
    print("3. Interactive mode: python scripts/run_agent_cli.py")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)