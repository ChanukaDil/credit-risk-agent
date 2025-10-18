# 🏦 Credit Risk Assessment Agent

AI-powered credit risk assessment system combining deep learning autoencoder, RAG (Retrieval-Augmented Generation), and LLM for intelligent decision-making.

## 🎯 Features

- **Autoencoder-based Anomaly Detection**: Deep learning model for risk scoring
- **RAG System**: Policy retrieval from knowledge base using ChromaDB
- **LLM Agent**: Natural language interface with Llama 3.2 3B
- **Multi-stage Validation**: Autoencoder → RAG → LLM → Decision pipeline
- **REST API**: FastAPI-based production-ready API
- **Interactive CLI**: Beautiful command-line interface

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/credit-risk-agent.git
cd credit-risk-agent
```

2. **Create virtual environment:**

```bash
python -m venv venv_agent
source venv_agent/bin/activate  # Linux/Mac
# venv_agent\Scripts\activate  # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
pip install -r requirements_agent.txt
```

4. **Download models:**

```bash
python scripts/download_models.py
```

5. **Initialize knowledge base:**

```bash
python scripts/setup_vector_db.py
```

6. **Run the agent:**

```bash
# CLI Interface
python scripts/run_agent_cli.py

# Or API Server
uvicorn src.api.fastapi_app:app --reload
```

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original dataset: [Bank Data Source]
- LangChain for agent framework
- HuggingFace for models

## ⚠️ Important Notes

**Model Files**: Large model files are not included in this repository due to size constraints. Download them using:

```bash
python scripts/download_models.py
```

**Data Files**: Sample data structure is provided, but actual data files must be obtained separately.

**Environment Variables**: Copy `.env.example` to `.env` and configure your settings.

---

**Star ⭐ this repository if you find it helpful!**
