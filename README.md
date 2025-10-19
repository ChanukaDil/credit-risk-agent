# 🏦 Credit Risk Assessment Agent

AI-powered credit risk assessment system combining deep learning autoencoder, RAG (Retrieval-Augmented Generation), and LLM for intelligent decision-making.

## 🎯 Features

- **Autoencoder-based Anomaly Detection**: Deep learning model for risk scoring
- **RAG System**: Policy retrieval from knowledge base using ChromaDB
- **LLM Agent**: Natural language interface with Llama 3.2 3B
- **Multi-stage Validation**: Autoencoder → RAG → LLM → Decision pipeline
- **REST API**: FastAPI-based production-ready API
- **Interactive CLI**: Beautiful command-line interface


## 📊 Architecture

Customer Input → Autoencoder (Risk Score) → RAG (Policy Retrieval) → LLM (Explanation) → Decision

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, Scikit-learn
- **LLM**: Llama 3.2 3B (HuggingFace)
- **RAG**: LangChain, ChromaDB, sentence-transformers
- **API**: FastAPI, Uvicorn
- **Data Processing**: Pandas, NumPy, SMOTE-Tomek

## 📁 Project Structure

credit-risk-agent/
├── src/ # Source code
│ ├── agent/ # AI agent components
│ ├── api/ # FastAPI application
│ └── utils/ # Utilities
├── models/ # Model artifacts (not in repo)
├── data/ # Data files (not in repo)
├── knowledge_base/ # Policy documents
├── scripts/ # Setup scripts
├── tests/ # Test suite
└── docs/ # Documentation

## 📖 Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [API Documentation](docs/API_DOCUMENTATION.md)
- [Architecture Overview](docs/AGENT_ARCHITECTURE.md)

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_agent.py -v
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f agent-api
```

## 📊 Performance

- **Accuracy**: 94.14%
- **ROC-AUC**: 0.6748
- **Inference Time**: <1 second per customer
- **Batch Processing**: ~1000 customers/second

## 🙏 Acknowledgments

- Original dataset: [Bank Data Source]
- LangChain for agent framework
- HuggingFace for models
