"""
Download Models Script
Downloads LLM and embedding models from HuggingFace
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_llm_model(model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
    """
    Download LLM model from HuggingFace
    
    Args:
        model_name: HuggingFace model identifier
    """
    print("\n" + "="*70)
    print("📥 Downloading LLM Model")
    print("="*70)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Create directory
        model_dir = Path("models/llm") / model_name.split("/")[-1]
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading model: {model_name}")
        logger.info(f"Destination: {model_dir}")
        logger.info("This may take 10-20 minutes (model is ~7GB)...")
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(model_dir),
            trust_remote_code=True
        )
        tokenizer.save_pretrained(str(model_dir))
        logger.info("✅ Tokenizer downloaded")
        
        # Download model
        logger.info("Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(model_dir),
            trust_remote_code=True
        )
        model.save_pretrained(str(model_dir))
        logger.info("✅ Model downloaded")
        
        print(f"\n✅ LLM model downloaded successfully to: {model_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading LLM: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (~8GB)")
        print("3. Try setting HuggingFace token:")
        print("   export HUGGINGFACE_TOKEN='your_token'")
        print("4. Get token from: https://huggingface.co/settings/tokens")
        return False


def download_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Download embedding model from HuggingFace
    
    Args:
        model_name: HuggingFace model identifier
    """
    print("\n" + "="*70)
    print("📥 Downloading Embedding Model")
    print("="*70)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Create directory
        model_dir = Path("models/embeddings")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading model: {model_name}")
        logger.info(f"Destination: {model_dir}")
        logger.info("This should take 1-2 minutes (~90MB)...")
        
        # Download model
        model = SentenceTransformer(model_name)
        model.save(str(model_dir / model_name.split("/")[-1]))
        
        logger.info("✅ Embedding model downloaded")
        
        print(f"\n✅ Embedding model downloaded successfully to: {model_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading embeddings: {e}")
        return False


def verify_models():
    """Verify that models can be loaded"""
    print("\n" + "="*70)
    print("🔍 Verifying Model Installation")
    print("="*70)
    
    try:
        # Test LLM
        print("\n1. Testing LLM...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        llm_path = Path("models/llm")
        if llm_path.exists():
            # Find the model directory
            model_dirs = list(llm_path.glob("Llama*"))
            if model_dirs:
                tokenizer = AutoTokenizer.from_pretrained(str(model_dirs[0]))
                print("   ✅ LLM loads successfully")
            else:
                print("   ⚠️ LLM directory not found")
        else:
            print("   ⚠️ LLM not downloaded yet")
        
        # Test embeddings
        print("\n2. Testing Embeddings...")
        from sentence_transformers import SentenceTransformer
        
        emb_path = Path("models/embeddings")
        if emb_path.exists():
            model_dirs = list(emb_path.glob("all-MiniLM*"))
            if model_dirs:
                model = SentenceTransformer(str(model_dirs[0]))
                print("   ✅ Embeddings load successfully")
            else:
                print("   ⚠️ Embedding directory not found")
        else:
            print("   ⚠️ Embeddings not downloaded yet")
        
        print("\n✅ Verification complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Verification error: {e}")
        return False


def check_disk_space():
    """Check if there's enough disk space"""
    try:
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)  # Convert to GB
        
        logger.info(f"Free disk space: {free_gb} GB")
        
        if free_gb < 10:
            logger.warning("⚠️ Low disk space! Need at least 10GB free")
            logger.warning(f"   Current free space: {free_gb} GB")
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Continue anyway


def main():
    """Main download function"""
    
    print("\n" + "="*70)
    print("🤖 Credit Risk Agent - Model Downloader")
    print("="*70)
    print("\nThis script will download:")
    print("1. Llama 3.2 3B Instruct (~7GB)")
    print("2. all-MiniLM-L6-v2 embeddings (~90MB)")
    print("\nTotal size: ~8GB")
    print("Estimated time: 10-25 minutes")
    
    # Check disk space
    print("\n" + "-"*70)
    if not check_disk_space():
        response = input("\n⚠️ Low disk space. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("❌ Download cancelled")
            return
    
    # Confirm download
    print("-"*70)
    response = input("\nProceed with download? (y/n): ")
    if response.lower() != 'y':
        print("❌ Download cancelled")
        return
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download models
    success = True
    
    # 1. Download LLM
    llm_success = download_llm_model()
    if not llm_success:
        success = False
    
    # 2. Download embeddings
    emb_success = download_embedding_model()
    if not emb_success:
        success = False
    
    # Verify installation
    if success:
        verify_models()
    
    # Final summary
    print("\n" + "="*70)
    if success:
        print("✅ All models downloaded successfully!")
        print("\nNext steps:")
        print("1. Set up knowledge base: python src/utils/document_loader.py")
        print("2. Initialize vector DB: python scripts/setup_vector_db.py")
        print("3. Test the agent: python scripts/run_agent_cli.py")
    else:
        print("⚠️ Some models failed to download")
        print("\nPlease check the errors above and try again")
        print("\nCommon solutions:")
        print("- Check internet connection")
        print("- Free up disk space")
        print("- Set HuggingFace token if needed")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)