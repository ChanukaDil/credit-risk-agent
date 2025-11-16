

import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def authenticate_huggingface():
    """Authenticate with HuggingFace"""
    
    # Try to get token from environment
    token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    
    if token:
        try:
            from huggingface_hub import login
            login(token=token)
            logger.info("✅ Authenticated with HuggingFace using token")
            return token
        except Exception as e:
            logger.warning(f"Token authentication failed: {e}")
            return None
    else:
        # Check if already logged in via CLI
        from huggingface_hub import HfFolder
        stored_token = HfFolder.get_token()
        
        if stored_token:
            logger.info("✅ Using stored HuggingFace credentials")
            return stored_token
        else:
            logger.warning("⚠️ No HuggingFace token found!")
            logger.warning("Please set HUGGINGFACE_TOKEN environment variable")
            logger.warning("Or run: huggingface-cli login")
            return None


def download_llm_model(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    token: str = None
):
    """Download LLM model with authentication"""
    
    print("\n" + "="*70)
    print("📥 Downloading LLM Model")
    print("="*70)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Create directory
        model_dir = Path("models/llm") / model_name.split("/")[-1]
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading: {model_name}")
        logger.info(f"Destination: {model_dir}")
        logger.info("Size: ~7GB, Time: 10-20 minutes")
        
        # Download with authentication
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,  # Pass token
            trust_remote_code=True,
            cache_dir=str(model_dir)
        )
        tokenizer.save_pretrained(str(model_dir))
        logger.info("✅ Tokenizer downloaded")
        
        logger.info("Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,  # Pass token
            trust_remote_code=True,
            cache_dir=str(model_dir)
        )
        model.save_pretrained(str(model_dir))
        logger.info("✅ Model downloaded")
        
        print(f"\n✅ LLM downloaded successfully to: {model_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        
        # Provide helpful error messages
        if "401" in str(e) or "403" in str(e):
            print("\n❌ AUTHENTICATION ERROR!")
            print("\nPossible causes:")
            print("1. Invalid or expired token")
            print("2. Haven't accepted Llama license agreement")
            print("3. Token doesn't have required permissions")
            print("\nSolutions:")
            print("1. Get token: https://huggingface.co/settings/tokens")
            print("2. Accept license: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
            print("3. Set token: export HUGGINGFACE_TOKEN=your_token")
            
        elif "gated" in str(e).lower():
            print("\n❌ MODEL IS GATED!")
            print("\nYou need to:")
            print("1. Go to: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
            print("2. Click 'Agree and access repository'")
            print("3. Wait for approval (usually instant)")
            print("4. Try downloading again")
        
        return False


def download_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """Download embedding model (no auth required)"""
    
    print("\n" + "="*70)
    print("📥 Downloading Embedding Model")
    print("="*70)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_dir = Path("models/embeddings")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading: {model_name}")
        logger.info("Size: ~90MB, Time: 1-2 minutes")
        
        model = SentenceTransformer(model_name)
        model.save(str(model_dir / model_name.split("/")[-1]))
        
        logger.info("✅ Embedding model downloaded")
        print(f"\n✅ Embeddings downloaded to: {model_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False


def main():
    """Main download function"""
    
    print("\n" + "="*70)
    print("🤖 Credit Risk Agent - Model Downloader")
    print("="*70)
    
    # Step 1: Authenticate
    print("\n🔐 Authenticating with HuggingFace...")
    token = authenticate_huggingface()
    
    if not token:
        print("\n⚠️ WARNING: No authentication token found!")
        print("You may not be able to download gated models like Llama.")
        print("\nTo authenticate:")
        print("1. Get token: https://huggingface.co/settings/tokens")
        print("2. Set token: export HUGGINGFACE_TOKEN=your_token")
        print("3. Or login: huggingface-cli login")
        
        response = input("\nContinue without authentication? (y/n): ")
        if response.lower() != 'y':
            print("❌ Download cancelled")
            return
    
    # Step 2: Download models
    print("\n" + "-"*70)
    print("📦 Starting Downloads")
    print("-"*70)
    
    # Download LLM
    llm_success = download_llm_model(token=token)
    
    # Download embeddings
    emb_success = download_embedding_model()
    
    # Summary
    print("\n" + "="*70)
    if llm_success and emb_success:
        print("✅ ALL MODELS DOWNLOADED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Create knowledge base: python src/utils/document_loader.py")
        print("2. Initialize vector DB: python scripts/setup_vector_db.py")
        print("3. Test agent: python scripts/run_agent_cli.py")
    elif llm_success:
        print("⚠️ LLM downloaded, but embeddings failed")
    elif emb_success:
        print("⚠️ Embeddings downloaded, but LLM failed")
    else:
        print("❌ DOWNLOAD FAILED")
        print("\nPlease check errors above and try again")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Download interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)